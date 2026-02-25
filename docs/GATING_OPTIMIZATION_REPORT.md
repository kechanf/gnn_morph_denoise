# 门控优化开发报告：Conflict-Aware Gating for Mamba+GNN Fusion

## 1. 背景与动机

### 1.1 问题

在 **Baseline A（10+10）** 中，每层 GPSLayer 包含两条并行分支：

- **GNN（GatedGCN）**：局部消息传递，刻画边与局部几何。
- **Mamba**：序列建模，依赖节点顺序，刻画长程依赖。

默认融合方式为 **简单相加**：

```text
h = h_local (GNN) + h_attn (Mamba)
```

当两路输出在部分维度上**不一致**（例如 Mamba 因顺序或长程噪声给出与局部结构矛盾的表示）时，直接相加会混入冲突信息，可能损害节点分类效果。希望在不改动 baseline 结构的前提下，对「冲突」做显式建模，在融合时**按维度、按样本**调节两路权重。

### 1.2 目标

- 在 **Logit 层面** 引入可学习的门控，得到逐维凸组合，避免不可解释的复杂融合。
- 用 **冲突度量**（两路逐维差异）调制门控：冲突大时偏向 GNN（局部更可靠），冲突小时可更信任 Mamba。
- **可选开关**：默认保持 `h = sum(...)`，与现有 baseline 完全一致；仅当配置启用时才接入门控模块。

---

## 2. 思路与设计

### 2.1 冲突感知（Conflict-Aware）

- 定义逐维冲突：**Δ_diff = (H_Mamba − H_GNN)²**（逐元素平方差）。
- 含义：某维上两路输出越接近，Δ_diff 越小；差异越大，冲突越大。
- 设计原则：**冲突大 → 降低 Mamba 的权重（即减小 α）→ 更依赖 GNN**。

### 2.2 逐维门控（Channel-wise Gating）

- 不采用单一标量 α，而是 **每个通道一个 α**，即 α ∈ ℝ^d，与隐藏维度 d 一致。
- 这样不同维度可以有不同的 Mamba/GNN 权重，适应不同语义（如局部几何 vs 长程上下文）。

### 2.3 Logit 偏置公式

门控采用 **Sigmoid( gate_logit − β·Δ_diff )**：

- **gate_logit**：由小型 MLP 根据 `z = [H_Mamba; H_GNN]` 得到，可学习。
- **β·Δ_diff**：冲突惩罚项；β > 0 时，Δ_diff 越大，logit 被减得越多，α 越小，越偏 GNN。
- **α = Sigmoid(gate_logit − β·Δ_diff)**，保证 α ∈ (0,1)，得到逐维凸组合：
  - **H_Fused = α ⊙ H_Mamba + (1−α) ⊙ H_GNN**（⊙ 为逐元素乘）。

这样既保留可学习能力（gate_net），又用冲突项做了显式偏置，便于调参和复现。

---

## 3. 实现方法

### 3.1 公式与流程

1. **拼接**：`z = concat(H_Mamba, H_GNN)`，形状 `[N, 2*dim]`。
2. **门控 logit**：`gate_logit = gate_net(z)`，形状 `[N, dim]`，`gate_net` 为 `Linear(2*dim, dim)`。
3. **冲突**：`diff = (H_Mamba - H_GNN)^2`，形状 `[N, dim]`。
4. **门控**：`alpha = sigmoid(gate_logit - beta * diff)`，β 可标量可学习。
5. **融合**：`H_Fused = alpha * H_Mamba + (1 - alpha) * H_GNN`。

### 3.2 模块实现：ConflictAwareFusion

**文件**：`external/Graph-Mamba/graphgps/layer/fusion_gating.py`

```python
class ConflictAwareFusion(nn.Module):
    def __init__(self, dim: int, beta: float = 1.0, learnable_beta: bool = True, gate_init_zero: bool = False):
        # gate_net: [N, 2*dim] -> [N, dim]
        # gate_init_zero=True 时权重与偏置零初始化，使初始 alpha ≈ 0.5（两端等权）
        # beta 可为可学习参数或固定 buffer
```

- **forward(h_mamba, h_gnn)**：按上述 5 步计算并返回 `h_fused`，与 `h_mamba`/`h_gnn` 同形状 `[N, dim]`。
- **参数**：
  - `dim`：隐藏维度，与 GPS 层一致。
  - `beta`：冲突惩罚强度，默认 1.0。
  - `learnable_beta`：是否将 β 设为可学习参数。
  - `gate_init_zero`：是否对 `gate_net` 做零初始化，使训练初期 α≈0.5，避免过早偏向一侧。

### 3.3 配置与默认值

**配置定义**（`external/Graph-Mamba/graphgps/config/gt_config.py`）：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `cfg.gt.fusion` | str | `'sum'` | 融合方式：`'sum'` 或 `'conflict_aware'` |
| `cfg.gt.fusion_beta` | float | 1.0 | 冲突项系数 β |
| `cfg.gt.fusion_gate_init_zero` | bool | False | 是否零初始化 gate_net（初始 α≈0.5） |

**YAML 示例**（`external/Graph-Mamba/configs/Mamba/morphology-node-EX.yaml`）：

```yaml
gt:
  fusion: sum
  fusion_beta: 1.0
  # fusion_gate_init_zero 未写则用代码默认 False
```

### 3.4 接入点：GPSLayer

**文件**：`external/Graph-Mamba/graphgps/layer/gps_layer.py`

- **初始化**（在 `GPSLayer.__init__` 末尾）：
  - 读取 `cfg.gt.fusion`，若为 `'conflict_aware'` 则构造 `self.fusion_module = ConflictAwareFusion(dim_h, ...)`；
  - 否则 `self.fusion_module = None`。
- **前向**（在 local + global 两支得到 `h_out_list` 后）：
  - 若 `self.fusion_module is not None` 且 `len(h_out_list) == 2`：
    - `h_out_list[0]` = GNN 输出，`h_out_list[1]` = Mamba 输出；
    - 调用 `h = self.fusion_module(h_out_list[1], h_out_list[0])`（注意参数顺序：Mamba, GNN）。
  - 否则：`h = sum(h_out_list)`，与原有 baseline 行为一致。

这样 **未启用门控时**（默认 `fusion: sum`）不创建额外模块、不改变计算图，与两种 baseline 完全兼容。

---

## 4. 调用方法

### 4.1 通过配置文件（YAML）

在使用的 YAML（如 `morphology-node-EX.yaml`）中设置：

```yaml
gt:
  fusion: conflict_aware
  fusion_beta: 1.0
  fusion_gate_init_zero: false   # 可选，true 时初始 α≈0.5
```

或保持 `fusion: sum` 即关闭门控。

### 4.2 通过命令行 override

不修改 YAML 时，可在运行脚本时覆盖：

```bash
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0 \
  --override gt.fusion_gate_init_zero True
```

**完整示例**（与 Baseline A 同数据、同超参，仅开启门控）：

```bash
export GNN_DATA_ROOT=/path/to/your/data_root
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --name_tag full_gating_run \
  --repeat 1 \
  --override gt.layers 10 \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0
```

### 4.3 在代码中读取配置

- 配置来自 `cfg.gt`（GraphGym 的 global config），在 `set_cfg_gt()` 中已注册上述键。
- 若在自定义脚本中需要判断是否使用门控，可写：
  - `getattr(cfg.gt, 'fusion', 'sum') == 'conflict_aware'`。

### 4.4 直接调用模块（仅作参考）

若在其他地方复用融合逻辑，可直接实例化并前向：

```python
from graphgps.layer.fusion_gating import ConflictAwareFusion

fusion = ConflictAwareFusion(dim=96, beta=1.0, gate_init_zero=False)
h_fused = fusion(h_mamba, h_gnn)  # h_mamba, h_gnn: [N, 96]
```

---

## 5. 实验与参考结果

- **设定**：与 Baseline A 一致（全量 2000 图，200 epoch，batch 32，dim_inner 96，lr 0.001，wd 0.01）。
- **门控配置**：`fusion=conflict_aware`，`fusion_beta=1.0`。
- **参考结果**（单次运行）：Val acc ≈ 88.9%，Test acc ≈ 88.6%；与 Baseline A（Val 88.21%，Test 88.14%）同量级或略高，说明门控在未破坏 baseline 的前提下带来小幅提升或持平。

---

## 6. 文件与代码索引

| 内容 | 路径 |
|------|------|
| 门控模块定义 | `external/Graph-Mamba/graphgps/layer/fusion_gating.py` |
| GPS 层接入（创建 + forward 分支） | `external/Graph-Mamba/graphgps/layer/gps_layer.py` |
| 配置默认值（fusion / fusion_beta / fusion_gate_init_zero） | `external/Graph-Mamba/graphgps/config/gt_config.py` |
| 示例 YAML（gt.fusion 等） | `external/Graph-Mamba/configs/Mamba/morphology-node-EX.yaml` |
| 使用说明与 baseline 对照 | `docs/BASELINES.md`（门控可选小节） |
| 训练入口与 override 用法 | `scripts/run_graph_mamba.py` |

---

## 7. 小结

- **思路**：用逐维冲突 (H_Mamba−H_GNN)² 调制可学习门控，在 Logit 上做偏置，得到 α = Sigmoid(gate_logit − β·Δ_diff)，再做凸组合融合，冲突大时偏 GNN。
- **实现**：独立模块 `ConflictAwareFusion`，通过 `gt.fusion` 在 GPSLayer 中可选接入；默认 `fusion=sum` 时行为与原有两种 baseline 完全一致。
- **调用**：YAML 中设置 `gt.fusion` / `gt.fusion_beta` / `gt.fusion_gate_init_zero`，或通过 `--override gt.fusion conflict_aware` 等命令行参数启用。

---

## 8. Gate structure variants (tried and reverted)

The current **gate_net** is a single **Linear(2*dim, dim)** as in §3.1. Two variants were tried and reverted:

- **Two-layer MLP** (2*dim → dim/2 → dim): In experiments such as `bfs_gating_depth_tau0.5_mlp`, Val/Test degraded (e.g. val_loss ↑, test_acc ↓) and overfitting increased; reverted. **Do not use** this variant.
- **ReLU + Linear + ReLU**: Reverted by request; no full comparison. Current code uses **single Linear only**.
