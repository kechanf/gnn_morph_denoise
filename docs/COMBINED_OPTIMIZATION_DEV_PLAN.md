# 组合优化实验开发计划：BFS 扫描 + 门控融合

本文档描述将 **启发式扫描优化（Mamba_GNNPriorityBFS）** 与 **门控融合（Conflict-Aware Gating）** 组合使用的开发计划与实验设计。

---

## 一、背景与目标

### 1.1 两个优化模块回顾

| 模块 | 作用阶段 | 核心思想 | 独立效果 |
|------|----------|----------|----------|
| **Mamba_GNNPriorityBFS** | Mamba 输入（序列排序） | 用 GNN 输出预测树/噪声得分，优先扫描主干、噪声靠后 | Val +4%，Test +3.8%（相对 Baseline A） |
| **Conflict-Aware Gating** | GNN+Mamba 输出融合 | 用逐维冲突 (H_Mamba−H_GNN)² 调制门控，冲突大时偏 GNN | Val/Test 略升或持平（+0.7% 量级） |

### 1.2 组合动机

- **扫描优化**：改善 Mamba 的输入序列质量，使长程建模更聚焦主干；
- **门控融合**：在输出端抑制两路冲突，融合时按冲突调节权重。
- 两者作用于**不同阶段**（输入 vs 输出），理论上可叠加：更好的输入 → 更少噪声的 Mamba 输出 → 门控进一步抑制剩余冲突。

### 1.3 目标

- 验证组合在架构与运行上的**兼容性**；
- 提供一键运行的**预设与命令行入口**；
- 完成**全量实验**并对比：Baseline A / 仅 BFS / 仅门控 / **BFS+门控**；
- 产出组合实验报告与可选消融分析。

---

## 二、架构兼容性分析

### 2.1 数据流（单层 GPSLayer）

```
h (上一层输出)
    │
    ├─→ GNN 分支 ──→ h_local
    │
    └─→ Mamba 分支：
            │  [Mamba_GNNPriorityBFS] 用 h_local 计算 scores → 排序 → permute(h)
            │  [普通 Mamba] 直接 to_dense_batch(h)
            │
            └─→ h_attn (Mamba 输出)

h_local 与 h_attn ──→ [融合]
    │  fusion=sum:           h = h_local + h_attn
    │  fusion=conflict_aware: h = ConflictAwareFusion(h_attn, h_local)
    │
    └─→ h → FFN → 下一层
```

### 2.2 兼容性结论

- **fusion_module** 在 `GPSLayer.__init__` 中根据 `cfg.gt.fusion` 创建，与 `global_model_type` 无关；
- **Mamba_GNNPriorityBFS** 由 `cfg.gt.layer_type` 指定，与 `fusion` 无关；
- 前向流程中：先得到 `h_out_list = [h_local, h_attn]`，再根据 `fusion_module` 是否非空选择融合方式。

**结论**：组合在现有代码中**已兼容**，无需改动 `gps_layer.py` 核心逻辑。只需通过配置同时启用两者：

```yaml
gt:
  layer_type: CustomGatedGCN+Mamba_GNNPriorityBFS
  fusion: conflict_aware
  fusion_beta: 1.0
  gnn_priority_aux_weight: 0.5
```

或命令行：

```bash
--override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
--override gt.fusion conflict_aware
```

---

## 三、开发计划

### 阶段 1：快速验证（预计 0.5 天）

| 任务 | 操作 | 预期 |
|------|------|------|
| 1.1 tiny50 组合跑通 | 用 `synthesis_data_tiny_50` + 3 epoch 跑 BFS+门控 | 无报错，loss 下降 |
| 1.2 对比 4 种配置 | 同 tiny50：A / BFS / 门控 / BFS+门控 | 确认组合不会明显劣化 |

**命令示例**：

```bash
# 组合（BFS + 门控）
python scripts/run_graph_mamba.py \
  --data_dir $GNN_DATA_ROOT/synthesis_data_tiny_50 \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override gt.fusion conflict_aware \
  --override optim.max_epoch 3 \
  --name_tag combo_bfs_gating_tiny50
```

### 阶段 2：预设与入口（预计 0.5 天）

| 任务 | 操作 |
|------|------|
| 2.1 新增 `combo` 预设 | 在 `config.py` 增加 `GRAPH_MAMBA_BASELINE_COMBO_*`，合并 BFS 与门控 overrides |
| 2.2 扩展 `--baseline` 选项 | 在 `run_graph_mamba.py` 的 `choices` 中加入 `combo`，解析时应用 combo overrides |
| 2.3 更新 BASELINES.md | 添加「组合：BFS + 门控」小节及运行示例 |

**config.py 新增**：

```python
# 组合：BFS 扫描 + 门控融合（与 A 参数一致）
GRAPH_MAMBA_BASELINE_COMBO_NAME = "baseline_combo_bfs_gating"
GRAPH_MAMBA_BASELINE_COMBO_OVERRIDES = {
    "gt.layer_type": "CustomGatedGCN+Mamba_GNNPriorityBFS",
    "gt.layers": 10,
    "gt.fusion": "conflict_aware",
    "gt.fusion_beta": 1.0,
}
```

**run_graph_mamba.py**：`--baseline combo` 时使用上述 overrides。

### 阶段 3：全量实验（预计 1–2 天，含训练时间）

| 任务 | 操作 |
|------|------|
| 3.1 统一数据规模 | 建议与现有报告一致：**2000 图**（或 8000 图，需在报告内统一说明） |
| 3.2 4 组对比实验 | Baseline A / BFS / 门控 / **BFS+门控**，每组 repeat≥2，取均值±std |
| 3.3 记录最佳 epoch、Val acc、Test acc | 用于撰写组合实验报告 |

**实验矩阵**：

| 配置 | layer_type | fusion | name_tag |
|------|------------|--------|----------|
| Baseline A | CustomGatedGCN+Mamba | sum | baseline_10_10 |
| BFS | CustomGatedGCN+Mamba_GNNPriorityBFS | sum | baseline_bfs |
| 门控 | CustomGatedGCN+Mamba | conflict_aware | baseline_gating |
| **组合** | CustomGatedGCN+Mamba_GNNPriorityBFS | conflict_aware | baseline_combo |

### 阶段 4：可选消融与超参（预计 0.5–1 天）

| 任务 | 操作 |
|------|------|
| 4.1 门控 β | 在 combo 基础上 sweep `fusion_beta` ∈ {0.5, 1.0, 2.0}，观察对 Val/Test 的影响 |
| 4.2 辅助损失权重 | sweep `gnn_priority_aux_weight` ∈ {0.3, 0.5, 0.7}，可选 |
| 4.3 gate_init_zero | 对比 `True` vs `False`，看组合下是否更稳定 |

### 阶段 5：文档与报告（预计 0.5 天）

| 任务 | 操作 |
|------|------|
| 5.1 组合实验报告 | 新建 `docs/COMBINED_OPTIMIZATION_REPORT.md`，包含：动机、实验设置、结果表、小结 |
| 5.2 更新 BASELINES.md | 添加组合预设说明、命令、参考结果 |
| 5.3 交叉引用 | 在 GATING / HEURISTIC 报告中补充「可与另一优化组合」的说明 |

---

## 四、风险与缓解

| 风险 | 缓解 |
|------|------|
| 组合可能不如单独 BFS | 若 combo < BFS，保留 BFS 为推荐配置，门控作为可选增强 |
| 超参冲突（如 aux_weight 与 fusion_beta 互相影响） | 先固定 BFS 报告中的 aux_weight=0.5，仅调 fusion_beta |
| 训练时间翻倍 | 使用相同 repeat 数量，或优先跑 combo vs BFS 对比 |

---

## 五、验收标准

- [ ] tiny50 下 BFS+门控组合无报错、loss 正常收敛；
- [ ] `--baseline combo` 可一键运行组合配置；
- [ ] 全量实验至少完成 4 组对比（A / BFS / 门控 / combo），并记录 Val/Test acc；
- [ ] 产出 `COMBINED_OPTIMIZATION_REPORT.md`，包含结果表与结论；
- [ ] BASELINES.md 已更新组合小节。

---

## 六、时间估算

| 阶段 | 工时 |
|------|------|
| 阶段 1：快速验证 | 0.5 天 |
| 阶段 2：预设与入口 | 0.5 天 |
| 阶段 3：全量实验 | 1–2 天（含 GPU 训练） |
| 阶段 4：可选消融 | 0.5–1 天 |
| 阶段 5：文档与报告 | 0.5 天 |
| **合计** | **约 3–4.5 天** |

---

## 七、附录：命令速查

```bash
# 组合（手动 override，阶段 1 验证用）
python scripts/run_graph_mamba.py \
  --data_dir $GNN_DATA_ROOT/synthesis_data_tiny_50 \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0 \
  --override optim.max_epoch 3 \
  --name_tag combo_tiny50

# 全量组合（预设生效后）
python scripts/run_graph_mamba.py \
  --data_dir $GNN_DATA_ROOT/synthesis_data \
  --baseline combo \
  --wandb False \
  --repeat 2
```
