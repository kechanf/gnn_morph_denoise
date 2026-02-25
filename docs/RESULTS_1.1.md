# GNN 形态学节点分类 — 1.1 正式结果（当前版本）

本文档记录**全量五模型**在完整数据集上的 **1.1 轮**训练结果（含数值保护与 Mamba 显式 dt 参数），并保存所有**原始文件与五模型保存路径**便于复现与追溯。历史 1.0 见 `docs/RESULTS_1.0.md`。

---

## 1. 实验配置

| 项目 | 说明 |
|------|------|
| 版本 | **1.1**（当前） |
| 数据规模 | 2000 图（train/val/test 按 0.8/0.1/0.1 划分） |
| 每模型 epoch | 200 |
| batch_size | 32 |
| 数值保护 | 主任务 logits `_safe_logits`（clamp + nan_to_num）；Mamba 显式 `dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4` |
| 其他超参 | 与 `docs/GNN_DEVELOPMENT_SUMMARY.md` 中「统一训练超参」一致 |

---

## 2. 五模型结果总表（1.1）

| 模型 | 最佳 epoch | Val acc | Test acc |
|------|------------|---------|----------|
| 1. 纯 GNN Baseline (20_aligned) | 38 | 85.91% | 85.41% |
| 2. Mamba Baseline (10_10) | 41 | 88.53% | 88.28% |
| 3. BFS 优化 (Mamba_GNNPriorityBFS) | 73 | 90.51% | 90.60% |
| 4. 门控优化 (Conflict-Aware Gating) | 33 | 88.99% | 88.75% |
| 5. BFS+门控 (combo) | 68 | 91.66% | **91.19%** |

**说明**：最佳 epoch 按 **validation accuracy** 选取。本轮训练**未出现** “NaN or Inf found in input tensor”，数值保护生效。

---

## 3. 五模型保存地点（完整路径）

以下为 1.1 运行后**五个模型的结果与 checkpoint 保存目录**（绝对路径），同一目录下后续重训会覆盖或与 config 约定一致。

| 模型 | 保存路径（绝对） |
|------|------------------|
| 1. 纯 GNN Baseline | `/home/kfchen/gnn_project_local/graph_mamba_results/morphology-node-GatedGCN-only-baseline_20_aligned` |
| 2. Mamba Baseline | `/home/kfchen/gnn_project_local/graph_mamba_results/morphology-node-EX-baseline_10_10` |
| 3. BFS 优化 | `/home/kfchen/gnn_project_local/graph_mamba_results/morphology-node-EX-baseline_10_10_gnn_priority_bfs` |
| 4. 门控优化 | `/home/kfchen/gnn_project_local/graph_mamba_results/morphology-node-EX-full_gating_run` |
| 5. BFS+门控 | `/home/kfchen/gnn_project_local/graph_mamba_results/morphology-node-EX-baseline_combo_bfs_gating` |

每个目录下包含：
- **`42/`**：单次 run（seed 42）的 checkpoint、`logging.log`、train/val/test 曲线等；
- **`agg/`**：多 run 聚合结果（1.1 为单 run）；
- **`config.yaml`**：该 run 的完整配置。

---

## 4. 其他原始文件与路径

### 4.1 数据

| 用途 | 路径 |
|------|------|
| 数据根目录 | `/home/kfchen/gnn_project_local` |
| 全量合成数据（.pt） | `/home/kfchen/gnn_project_local/synthesis_data` |
| 小数据集（快速试跑用） | `/home/kfchen/gnn_project_local/synthesis_data_tiny_50` |

### 4.2 训练日志

| 用途 | 路径 |
|------|------|
| 五模型全量训练完整日志 | `/home/kfchen/gnn_project/all_five_models.log` |

### 4.3 脚本与文档

| 用途 | 路径 |
|------|------|
| 五模型依次训练脚本 | `/home/kfchen/gnn_project/scripts/train_all_five_models.sh` |
| 单模型训练入口 | `/home/kfchen/gnn_project/scripts/run_graph_mamba.py` |
| 开发总结与训练方法 | `/home/kfchen/gnn_project/docs/GNN_DEVELOPMENT_SUMMARY.md` |
| 门控设计说明 | `docs/GATING_OPTIMIZATION_REPORT.md`、`docs/门控融合开发报告.md` |
| BFS/扫描优化说明 | `docs/HEURISTIC_SCAN_OPTIMIZATION_REPORT.md` |
| NaN 原因与应对 | `docs/NAN_IN_TRAINING.md`、`docs/TRAINING_NAN_FIX.md` |

---

## 5. 复现命令（与 1.1 一致）

```bash
cd /home/kfchen/gnn_project
conda activate graph-mamba  # 或 PATH 指向 graph-mamba 的 bin

GNN_DATA_ROOT=/home/kfchen/gnn_project_local ./scripts/train_all_five_models.sh \
  --data_dir /home/kfchen/gnn_project_local/synthesis_data \
  --log /home/kfchen/gnn_project/all_five_models.log
```

结果写入 `GNN_DATA_ROOT/graph_mamba_results`，与 §3 中五模型保存路径一致（新 run 会覆盖同目录）。

---

## 6. 小结

- **1.1 结果**：BFS+门控 Test **91.19%** 最佳；整轮**无 NaN**，数值保护有效。
- **五模型保存地点**：已全部记录于 §3 绝对路径，便于加载 checkpoint 与复现。
