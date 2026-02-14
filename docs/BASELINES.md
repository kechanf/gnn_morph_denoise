# 两种统一 Baseline 配置

对比实验固定以下**两种 baseline**，训练/优化超参一致，仅结构不同。

## 统一训练超参（两者相同）

| 参数 | 值 |
|------|-----|
| train.batch_size | 32 |
| gnn.dim_inner | 96 |
| gnn.dropout | 0.0 |
| optim.base_lr | 0.001 |
| optim.weight_decay | 0.01 |
| optim.max_epoch | 200 |
| split | [0.8, 0.1, 0.1] |

## Baseline A：10+10（GatedGCN + Mamba）

- **结构**：10 层 GPSLayer，每层 = 1 个 GatedGCN + 1 个 Mamba。
- **配置**：`morphology-node-EX.yaml`，override `gt.layers=10`。
- **依赖**：需安装 `mamba-ssm`。

```bash
export GNN_DATA_ROOT=/path/to/data_root
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --name_tag baseline_10_10 \
  --repeat 1 \
  --override gt.layers 10
```

## Baseline B：20 层纯 GatedGCN（对齐版）

- **结构**：20 层 GatedGCN，无 Mamba。
- **配置**：`morphology-node-GatedGCN-only.yaml`，override 层数与上述统一超参。
- **依赖**：无需 mamba-ssm（`--no-mamba`）。

```bash
export GNN_DATA_ROOT=/path/to/data_root
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --name_tag baseline_20_aligned \
  --repeat 1 \
  --no-mamba \
  --override gnn.layers_mp 20 \
  --override gnn.dim_inner 96 \
  --override gnn.dropout 0.0 \
  --override optim.base_lr 0.001 \
  --override optim.weight_decay 0.01 \
  --override train.batch_size 32
```

## 参考结果（全量 2000 图，200 epoch）

| Baseline | 最佳 epoch | Val acc | Test acc |
|----------|------------|---------|----------|
| A (10+10) | 43 | 88.21% | 88.14% |
| B (20 层对齐) | 40 | 85.87% | 84.73% |

详见 `config.py` 中 `GRAPH_MAMBA_BASELINE_*` 常量。

### 可选：门控融合（Conflict-Aware Gating）

在 Baseline A（EX）上可启用逐维门控融合，替代默认的 `h = sum(h_local, h_attn)`：

- **配置**：`gt.fusion=conflict_aware`，`gt.fusion_beta`（默认 1.0），可选 `gt.fusion_gate_init_zero=True` 使初始 α≈0.5。
- **实现**：`external/Graph-Mamba/graphgps/layer/fusion_gating.py`（ConflictAwareFusion），在 `gps_layer.py` 中按配置分支调用；未启用时仍为 `sum`，不影响 baseline。
- **运行示例**（完整数据、结果落本地）：
  ```bash
  GNN_DATA_ROOT=/path/to/your/data_root python scripts/run_graph_mamba.py \
    --data_dir /path/to/synthesis_data --wandb False --name_tag full_gating_run --repeat 1 \
    --override gt.layers 10 --override gt.fusion conflict_aware --override gt.fusion_beta 1.0
  ```
- **参考结果**（2000 图，200 epoch）：Val acc ≈ 88.9%，Test acc ≈ 88.6%（与 Baseline A 同量级或略高）。

---

## 供其他 Agent 查阅：Baseline 与模型代码位置

### 两种 Baseline 简述

| 名称 | 结构 | 配置 YAML | 入口行为 |
|------|------|------------|----------|
| **Baseline A（10+10）** | 10 层 GPSLayer，每层 = 1×GatedGCN + 1×Mamba | `morphology-node-EX.yaml`，override `gt.layers=10` | `scripts/run_graph_mamba.py`，不传 `--no-mamba`，需 mamba-ssm |
| **Baseline B（20 层对齐）** | 20 层纯 GatedGCN，无 Mamba | `morphology-node-GatedGCN-only.yaml`，override 层数及统一超参 | `scripts/run_graph_mamba.py --no-mamba`，无需 mamba-ssm |

两者训练超参一致：batch 32，dim_inner 96，dropout 0，lr 0.001，weight_decay 0.01，200 epoch。

### 模型与配置代码位置（均相对于项目根或 `external/Graph-Mamba`）

- **训练入口**：`scripts/run_graph_mamba.py`（解析 `--no-mamba`、`--override`，调 `external/Graph-Mamba/main.py`）。
- **配置**：`external/Graph-Mamba/configs/Mamba/morphology-node-EX.yaml`（A）、`morphology-node-GatedGCN-only.yaml`（B）。
- **整体模型**  
  - **A（10+10）**：`external/Graph-Mamba/graphgps/network/gps_model.py` 中的 `GPSModel`，堆叠 `GPSLayer`。  
  - **B（20 层 GatedGCN）**：`external/Graph-Mamba/graphgps/network/custom_gnn.py` 中的 `CustomGNN`，堆叠 `GatedGCNLayer`。
- **单层实现**  
  - GatedGCN：`external/Graph-Mamba/graphgps/layer/gatedgcn_layer.py` → `GatedGCNLayer`。  
  - GPS 层（GatedGCN + Mamba）：`external/Graph-Mamba/graphgps/layer/gps_layer.py` → `GPSLayer`（内用 `GatedGCNLayer` 与 `mamba_ssm.Mamba`，Mamba 可选 try/except）。
- **数据**：`external/Graph-Mamba/graphgps/loader/dataset/morphology_node.py`（MorphologyNode），loader 在 `graphgps/loader/master_loader.py`。
- **Baseline 常量**：项目根 `config.py` 中 `GRAPH_MAMBA_BASELINE_10_10_*`、`GRAPH_MAMBA_BASELINE_20_ALIGNED_*`。
