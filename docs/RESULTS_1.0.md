# GNN 形态学节点分类 — 1.0 正式结果（历史）

> **当前最新为 1.1**（含数值保护、无 NaN、五模型保存路径）：见 **[docs/RESULTS_1.1.md](RESULTS_1.1.md)**。

本文档记录**全量五模型**在完整数据集上的 **1.0** 训练结果（历史基准），并保存当时**原始文件与路径**便于复现与对比。

---

## 1. 实验配置

| 项目 | 说明 |
|------|------|
| 版本 | 1.0 |
| 数据规模 | 2000 图（train/val/test 按 0.8/0.1/0.1 划分） |
| 每模型 epoch | 200 |
| batch_size | 32 |
| 其他超参 | 与 `docs/GNN_DEVELOPMENT_SUMMARY.md` 中「统一训练超参」一致 |

---

## 2. 五模型结果总表（1.0）

| 模型 | 最佳 epoch | Val acc | Test acc |
|------|------------|---------|----------|
| 1. 纯 GNN Baseline (20_aligned) | 38 | 85.85% | 84.71% |
| 2. Mamba Baseline (10_10) | 41 | 88.30% | 87.81% |
| 3. BFS 优化 (Mamba_GNNPriorityBFS) | 81 | 91.23% | 90.99% |
| 4. 门控优化 (Conflict-Aware Gating) | 31 | 88.77% | 88.66% |
| 5. BFS+门控 (combo) | 57 | 91.29% | 91.18% |

**说明**：最佳 epoch 按 **validation accuracy** 选取；训练过程中 Mamba 10_10 与 BFS 曾出现 “NaN or Inf found in input tensor” 警告，已通过跳过 nan loss 的 backward/step 与梯度裁剪保护，表中为各 run 保存的 best checkpoint 对应指标。

---

## 3. 原始文件与路径

以下路径为本次 1.0 运行时的**实际路径**，请勿随意移动或删除，便于复现与对比。

### 3.1 数据

| 用途 | 路径 |
|------|------|
| 数据根目录 | `/home/kfchen/gnn_project_local` |
| 全量合成数据（.pt） | `/home/kfchen/gnn_project_local/synthesis_data` |
| 小数据集（快速试跑用） | `/home/kfchen/gnn_project_local/synthesis_data_tiny_50` |

### 3.2 训练日志

| 用途 | 路径 |
|------|------|
| 五模型全量训练完整日志 | `/home/kfchen/gnn_project/all_five_models.log` |

### 3.3 模型结果目录（checkpoint、曲线、config）

结果根目录：`/home/kfchen/gnn_project_local/graph_mamba_results`

| 模型 | 结果子目录（相对上述根目录） |
|------|-----------------------------|
| 1. 纯 GNN Baseline | `morphology-node-GatedGCN-only-baseline_20_aligned` |
| 2. Mamba Baseline | `morphology-node-EX-baseline_10_10` |
| 3. BFS 优化 | `morphology-node-EX-baseline_10_10_gnn_priority_bfs` |
| 4. 门控优化 | `morphology-node-EX-full_gating_run` |
| 5. BFS+门控 | `morphology-node-EX-baseline_combo_bfs_gating` |

每个子目录下包含：
- `42/`：单次 run（seed 42）的 checkpoint、`logging.log`、train/val/test 曲线等；
- `agg/`：多 run 聚合结果（本次为单 run）；
- `config.yaml`：该 run 的完整配置。

### 3.4 脚本与文档

| 用途 | 路径 |
|------|------|
| 五模型依次训练脚本 | `/home/kfchen/gnn_project/scripts/train_all_five_models.sh` |
| 单模型训练入口 | `/home/kfchen/gnn_project/scripts/run_graph_mamba.py` |
| 开发总结与训练方法 | `/home/kfchen/gnn_project/docs/GNN_DEVELOPMENT_SUMMARY.md` |
| 门控设计说明 | `/home/kfchen/gnn_project/docs/GATING_OPTIMIZATION_REPORT.md`、`docs/门控融合开发报告.md` |
| BFS/扫描优化说明 | `/home/kfchen/gnn_project/docs/HEURISTIC_SCAN_OPTIMIZATION_REPORT.md` |
| NaN 原因与应对 | `/home/kfchen/gnn_project/docs/NAN_IN_TRAINING.md`、`docs/TRAINING_NAN_FIX.md` |

---

## 4. 复现命令（与 1.0 一致）

```bash
cd /home/kfchen/gnn_project
conda activate graph-mamba  # 或 PATH 指向 graph-mamba 的 bin

GNN_DATA_ROOT=/home/kfchen/gnn_project_local ./scripts/train_all_five_models.sh \
  --data_dir /home/kfchen/gnn_project_local/synthesis_data \
  --log /home/kfchen/gnn_project/all_five_models.log
```

结果将写入 `GNN_DATA_ROOT/graph_mamba_results`（即 `/home/kfchen/gnn_project_local/graph_mamba_results`），新 run 会落在与上表相同的子目录名（若存在则覆盖或按时间区分，以实际 config/out_dir 为准）。

---

## 5. 小结

- **1.0 结果**：BFS 与 BFS+门控 Test 约 **91.0%–91.2%**，门控相对 Mamba baseline 约 +0.9% Test，纯 GNN 84.71% 作为对照。
- **原始文件**：数据、日志、各模型结果目录及脚本路径已全部记录于 §3，便于长期保留与复现。
