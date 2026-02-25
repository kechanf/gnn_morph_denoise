# GNN 形态学节点分类开发总结

本文档汇总形态学节点分类任务中**五种模型**的设计要点与**训练方法**，便于复现与对比。统一训练超参见下表，除结构/配置外保持一致。

---

## 统一训练超参

| 参数 | 值 |
|------|-----|
| train.batch_size | 32 |
| gnn.dim_inner | 96 |
| gnn.dropout | 0.0 |
| optim.base_lr | 0.001 |
| optim.weight_decay | 0.01 |
| optim.max_epoch | 200 |
| split | [0.8, 0.1, 0.1] |

**数据**：形态学合成图（MorphologyNode），`.pt` 目录由 `--data_dir` 指定；全量实验通常为 2000 图。**环境**：除纯 GNN baseline 外均需安装 `mamba-ssm`。

---

## 一键依次训练五模型（脚本快速使用）

使用脚本 **`scripts/train_all_five_models.sh`** 可顺序跑完上述五个模型，无需逐条执行命令。

**环境**：需先激活已安装 PyTorch 与 `mamba-ssm` 的 conda 环境（如 `graph-mamba`）。

**基本用法**（数据目录用 `config.DATA_ROOT` 下的 `synthesis_data`）：

```bash
cd /path/to/gnn_project
conda activate graph-mamba
./scripts/train_all_five_models.sh
```

**指定数据根目录与数据目录**（推荐）：

```bash
export GNN_DATA_ROOT=/path/to/data_root   # 例如 /home/user/gnn_project_local
./scripts/train_all_five_models.sh --data_dir $GNN_DATA_ROOT/synthesis_data
```

**常用参数**：

| 参数 | 说明 |
|------|------|
| `--data_dir DIR` | 形态学 `.pt` 数据目录（默认 `$GNN_DATA_ROOT/synthesis_data`） |
| `--log FILE` | 将全部输出追加写入日志文件 |
| `--max_epoch N` | 覆盖训练轮数（如 `3` 用于快速试跑） |
| `--continue` | 某个模型失败后继续跑后续模型（默认失败即退出） |
| `--dry_run` | 只打印将要执行的命令，不实际训练 |

**示例**：

```bash
# 快速试跑（每个模型 3 个 epoch）
./scripts/train_all_five_models.sh --data_dir $GNN_DATA_ROOT/synthesis_data_tiny_50 --max_epoch 3

# 全量训练并写日志，失败也继续
./scripts/train_all_five_models.sh --data_dir $GNN_DATA_ROOT/synthesis_data --log all_five.log --continue
```

脚本会依次执行：1 纯 GNN → 2 Mamba Baseline → 3 BFS → 4 门控 → 5 BFS+门控。查看进度可 `tail -f all_five.log`（若使用了 `--log`）。

---

## 1. 纯 GNN Baseline（Baseline B）

- **结构**：20 层 GatedGCN，无 Mamba，与 10+10 的参数量/感受野大致对齐。
- **配置**：使用 `morphology-node-GatedGCN-only.yaml`，`--baseline 20_aligned` 会自动加上 `--no-mamba`，无需 Mamba 依赖。
- **用途**：作为“无序列建模”的对照，评估 Mamba 分支的贡献。

### 训练方法

```bash
export GNN_DATA_ROOT=/path/to/data_root

python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --baseline 20_aligned
```

或手动指定（等效）：

```bash
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --no-mamba \
  --override gnn.layers_mp 20 \
  --override gnn.dim_inner 96 \
  --override gnn.dropout 0.0 \
  --override optim.base_lr 0.001 \
  --override optim.weight_decay 0.01 \
  --override train.batch_size 32
```

### 参考结果（2000 图，200 epoch）

| 指标 | 值 |
|------|-----|
| 最佳 epoch | 38 |
| Val acc | 85.91% |
| Test acc | 85.41% |

---

## 2. Mamba Baseline（Baseline A，10+10）

- **结构**：10 层 GPSLayer，每层 = 1×GatedGCN + 1×Mamba；融合方式为**简单相加** `h = h_local + h_attn`。
- **配置**：`morphology-node-EX.yaml`，override `gt.layers=10`，`gt.fusion` 保持默认 `sum`。
- **用途**：主 baseline，用于评估 BFS、门控、组合等优化的增益。

### 训练方法

```bash
export GNN_DATA_ROOT=/path/to/data_root

python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --baseline 10_10
```

或手动：

```bash
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --override gt.layers 10 \
  --name_tag baseline_10_10
```

### 参考结果（2000 图，200 epoch）

| 指标 | 值 |
|------|-----|
| 最佳 epoch | 41 |
| Val acc | 88.53% |
| Test acc | 88.28% |

---

## 3. BFS 优化（Mamba_GNNPriorityBFS）

- **结构**：与 Baseline A 相同为 10 层 GPSLayer，但 Mamba 分支改为 **GNN 引导的 Priority BFS/LexSort** 排序后再送入 Mamba；融合仍为 `sum`。
- **要点**：用 GNN 的 `h_local` 经 MLP 得到节点“树似然”得分，按得分（及可选的 `dist_from_root`）排序，使 Mamba 先看到主干、后看到噪声；辅助损失 BCE(scores, node_labels) 监督 MLP。
- **数据**：需 `dist_from_root`、`is_target_root` 时，删除 `$DATADIR/processed/morphology_processed.pt` 以触发重新预处理。

### 训练方法

```bash
export GNN_DATA_ROOT=/path/to/data_root

python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --baseline bfs
```

或手动：

```bash
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override gt.layers 10 \
  --name_tag baseline_10_10_gnn_priority_bfs
```

可选参数：

```bash
# 调整辅助损失权重与 dist 编码维度
--override gt.gnn_priority_aux_weight 0.5 \
--override gt.gnn_priority_pe_dim 16
```

### 参考结果（2000 图，200 epoch）

| 指标 | 值 |
|------|-----|
| 最佳 epoch | 73 |
| Val acc | 90.51% |
| Test acc | 90.60% |

相对 Baseline A 约 **+2.0% Val / +2.3% Test**。详见 `docs/HEURISTIC_SCAN_OPTIMIZATION_REPORT.md`。

---

## 4. 门控优化（Conflict-Aware Gating）

- **结构**：与 Baseline A 相同（10 层 GPSLayer，普通 Mamba），仅在**融合阶段**用逐维门控替代简单相加：  
  `α = σ(gate_logit − β·Δ_diff)`，`h = α⊙h_mamba + (1−α)⊙h_gnn`，冲突大时更偏 GNN。
- **配置**：`gt.fusion=conflict_aware`，`gt.fusion_beta=1.0`（可选 `gt.fusion_gate_init_zero=True`）。  
  可选增强：深度感知 β（`fusion_depth_aware_beta`）、温度 τ（`fusion_tau`），见 `docs/门控融合开发报告.md`。
- **用途**：在不动扫描顺序的前提下，评估“冲突感知融合”带来的收益。

### 训练方法

```bash
export GNN_DATA_ROOT=/path/to/data_root

python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --name_tag full_gating_run \
  --repeat 1 \
  --override gt.layers 10 \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0
```

可选：初始两路等权可加 `--override gt.fusion_gate_init_zero True`。

### 参考结果（2000 图，200 epoch）

| 指标 | 值 |
|------|-----|
| Val acc | 88.99% |
| Test acc | 88.75% |

与 Baseline A 同量级或略高。详见 `docs/GATING_OPTIMIZATION_REPORT.md`、`docs/门控融合开发报告.md`。

---

## 5. BFS + 门控（组合优化）

- **结构**：10 层 GPSLayer，Mamba 分支为 **Mamba_GNNPriorityBFS**，融合为 **Conflict-Aware Gating**；即同时启用 BFS 优化与门控优化。
- **配置**：`gt.layer_type=CustomGatedGCN+Mamba_GNNPriorityBFS`，`gt.fusion=conflict_aware`，`gt.fusion_beta=1.0`，其余与上述统一超参一致。
- **用途**：评估“扫描优化 + 融合优化”联合效果。

### 训练方法

```bash
export GNN_DATA_ROOT=/path/to/data_root

python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --name_tag baseline_combo_bfs_gating \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override gt.layers 10 \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0
```

快速验证（tiny50，3 epoch）：

```bash
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data_tiny_50 \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0 \
  --override optim.max_epoch 3 \
  --name_tag combo_bfs_gating_tiny50
```

### 参考结果（2000 图，200 epoch）

| 指标 | 值 |
|------|-----|
| 最佳 epoch | 68 |
| Val acc | 91.66% |
| Test acc | 91.19% |

相对 Baseline A 约 **+3.1% Val / +2.9% Test**；与单独 BFS 处于同一水平带，门控在 BFS 之上为小幅增强。详见 `docs/COMBINED_OPTIMIZATION_DEV_PLAN.md` 及组合实验记录。

---

## 五模型对比总表

| 模型 | 结构要点 | 训练命令概要 | Val acc | Test acc |
|------|----------|--------------|---------|----------|
| 纯 GNN Baseline | 20 层 GatedGCN，无 Mamba | `--baseline 20_aligned` | 85.91% | 85.41% |
| Mamba Baseline | 10+10，sum 融合 | `--baseline 10_10` | 88.53% | 88.28% |
| BFS 优化 | 10+10，Mamba_GNNPriorityBFS，sum | `--baseline bfs` | 90.51% | 90.60% |
| 门控优化 | 10+10，conflict_aware 融合 | `--override gt.fusion conflict_aware` 等 | 88.99% | 88.75% |
| BFS+门控 | 10+10，BFS + conflict_aware | 同上，并 `gt.layer_type` BFS + `gt.fusion conflict_aware` | 91.66% | 91.19% |

---

## 相关文档索引

| 内容 | 文档/路径 |
|------|-----------|
| **1.1 正式结果与五模型保存路径** | `docs/RESULTS_1.1.md`（当前版本：五模型结果、最佳 epoch、Val/Test acc、五模型保存绝对路径、无 NaN） |
| **1.0 历史结果** | `docs/RESULTS_1.0.md` |
| **一键训练五模型** | `scripts/train_all_five_models.sh`（用法见上文「一键依次训练五模型」） |
| Baseline 与统一超参 | `docs/BASELINES.md` |
| 门控设计与配置 | `docs/GATING_OPTIMIZATION_REPORT.md`、`docs/门控融合开发报告.md` |
| BFS/扫描优化 | `docs/HEURISTIC_SCAN_OPTIMIZATION_REPORT.md` |
| 组合实验计划与结论 | `docs/COMBINED_OPTIMIZATION_DEV_PLAN.md` |
| 训练入口与 config | `scripts/run_graph_mamba.py`、`config.py`（GRAPH_MAMBA_BASELINE_*） |
