# 启发式扫描优化开发报告

本文档描述在 **Mamba + GNN** 形态学节点分类中，对图节点进行**启发式扫描排序**的设计思路、实现方式与调用方法。

---

## 一、背景与动机

### 1.1 问题

- **Mamba** 是序列模型（SSM），输入为**有序序列**；图本身无自然顺序，需要将节点排成一条序列再送入 Mamba。
- 节点顺序会显著影响 Mamba 的建模效果：同一张图、不同顺序会得到不同表示。
- 形态学图近似**树结构**（目标树 + 干扰/噪声节点），希望顺序能体现“从根到叶”或“树优先、噪声靠后”的归纳偏置，以利于分类。

### 1.2 目标

- 在**不改变** GNN 与 Mamba 主体结构的前提下，通过**扫描顺序的优化**提升 Mamba 分支的表达能力。
- 支持从**固定启发式**（如 BFS/DFS、按度）到**数据驱动的 GNN 引导排序**（Priority BFS / LexSort），并可利用形态学数据中的 `dist_from_root`、`is_target_root` 等先验。

---

## 二、设计思路

### 2.1 扫描策略的演进

| 策略 | 说明 | 顺序依据 | 可学习 | 适用场景 |
|------|------|----------|--------|----------|
| **随机**（Mamba_Permute） | 每图内随机排列 | 无 | 否 | 无先验 |
| **按度**（Mamba_Degree） | 按节点度升序/降序 | 度 | 否 | 度与任务相关 |
| **树 BFS/DFS**（Mamba_TreeBFS 等） | 从选定的根做 BFS 或 DFS | 图拓扑 + 根选择 | 否 | 连通树 |
| **GNN 引导**（Mamba_GNNPriorityBFS） | 用 GNN 输出预测树/噪声得分再排序 | GNN 表示 + 可选 dist/root | **是**（MLP + 辅助损失） | 形态学树+噪声 |

### 2.2 核心思想（Mamba_GNNPriorityBFS）

1. **GNN 已能区分树节点与噪声**：局部消息传递后的 `h_local` 适合预测“该节点更像目标树还是噪声”。
2. **用 MLP 将 h_local 映射为标量得分** s_i ∈ [0,1]，表示“树似然”；高分优先扫描，使 Mamba 先看到主干、后看到噪声。
3. **BFS/LexSort 不可导**：用**辅助损失** BCE(scores, node_labels) 监督 MLP，使 scores 与真实节点标签（1=树，0=噪声）一致。
4. **形态学先验**：若有 `dist_from_root`、`is_target_root`，可拼入 MLP 输入并参与 **LexSort**（dist 主导、同层内按 score），兼顾层序与树似然。

### 2.3 排序策略的选择

| 条件 | 排序方式 | 特点 |
|------|----------|------|
| 有 `dist_from_root` 且维度正确 | **LexSort** | GPU 向量化，无 Python 循环，dist 主导、score 微调 |
| 无 `dist_from_root` 或维度不匹配 | **Priority BFS** | 每图根=argmax(score)，最大堆扩展，支持不连通图 |

LexSort 键为 `(batch, dist, -score)`：先按 batch 分组，再按测地距离（层序），同层内按得分降序。

---

## 三、实现方法

### 3.1 代码位置（均相对于项目根）

| 模块 | 路径 | 说明 |
|------|------|------|
| GPS 层与扫描逻辑 | `external/Graph-Mamba/graphgps/layer/gps_layer.py` | LexSort、BFS、MLP、前向分支 |
| 数据集与 dist/root 字段 | `external/Graph-Mamba/graphgps/loader/dataset/morphology_node.py` | `_preprocess_features` 中构造 dist_from_root、is_target_root |
| 辅助损失 | `external/Graph-Mamba/graphgps/train/custom_train.py` | `_add_gnn_priority_aux_loss` |
| 配置 | `external/Graph-Mamba/configs/Mamba/morphology-node-EX.yaml` | 默认 layer_type、gnn_priority_* |
| GT 配置扩展 | `external/Graph-Mamba/graphgps/config/gt_config.py` | gnn_priority_aux_weight、gnn_priority_pe_dim |

### 3.2 固定启发式（无学习）

- **tree_order_within_batch(edge_index, batch, order, root_choice)**  
  每图选根（min_degree/max_degree/first），按 BFS 或 DFS 生成节点顺序，返回全局节点索引的排列。
- **lexsort(keys)**  
  多键稳定排序，用于 Mamba_Degree、Mamba_GNNPriorityBFS 的 LexSort 路径。
- 在 `GPSLayer.forward` 中根据 `global_model_type` 分支：Mamba_TreeBFS、Mamba_TreeBFS_Soma、Mamba_TreeDFS、Mamba_Degree 等，得到 `h_ind_perm` 后对 `h[h_ind_perm]` 做 to_dense_batch 送入 Mamba，再按逆排列还原。

### 3.3 GNN 引导扫描（Mamba_GNNPriorityBFS）

#### 3.3.1 模型结构（GPSLayer.__init__）

- **Mamba 块**：与标准 Mamba 相同（d_model=dim_h, d_state=16, d_conv=4, expand=1）。
- **gnn_priority_mlp**：
  - 输入维度：`dim_h + gnn_priority_pe_dim + 1` = `h_local + dist 正弦 PE + is_target_root`
  - 结构：`Linear(dim_mlp_in, dim_h) → ReLU → Dropout → Linear(dim_h, 1)`
  - 输出：1 维 logit，sigmoid 后为 s_i ∈ [0,1]
  - `gnn_priority_pe_dim` 来自 `cfg.gt.gnn_priority_pe_dim`（默认 16）

#### 3.3.2 正弦距离编码（_sinusoidal_dist_encoding）

- 形式：`[sin(d·w_0), cos(d·w_0), sin(d·w_1), cos(d·w_1), ...]`
- 多尺度编码，使 MLP 能利用测地距离信息。

#### 3.3.3 LexSort（gnn_priority_lexsort）

- 输入：`batch_vec`、`scores`、`dist_from_root`
- 键：`(batch, dist, -scores)`，lexsort 主键为 batch，次键 dist，再次 -scores
- 全 GPU 操作，无 Python 循环，无 `.item()` 同步。

#### 3.3.4 Priority BFS（gnn_priority_bfs_within_batch）

- 每图根 = `argmax(scores)`（最像树的节点）
- 边界为最大堆，按 score 降序扩展
- **不连通图**：BFS 只覆盖根所在连通分量；未访问节点按 score 降序排在末尾，保证 perm 覆盖全部节点，避免维度不匹配。

#### 3.3.5 前向流程

```
1. 取 h_local（GNN 输出）
2. 构造 MLP 输入：
   - 有 dist_from_root、is_target_root：concat(h_local, sinusoidal(dist), is_root)
   - 否则：concat(h_local, zeros_pe, zeros_root)
3. scores_i = sigmoid(gnn_priority_mlp(mlp_in))
4. batch.gnn_priority_scores = scores_i  # 供辅助损失使用
5. 选择排序方式：
   - 有 dist：h_ind_perm = gnn_priority_lexsort(batch, scores, dist)
   - 无 dist：h_ind_perm = gnn_priority_bfs_within_batch(edge_index, batch, scores)
6. h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
7. h_attn = Mamba(h_dense)[mask][argsort(h_ind_perm)]  # 逆排列还原
```

#### 3.3.6 辅助损失

- 在 `custom_train` 的 train_epoch / eval_epoch 中调用 `_add_gnn_priority_aux_loss(loss, batch)`。
- 条件：`layer_type` 含 `Mamba_GNNPriorityBFS` 且 batch 有 `gnn_priority_scores` 和 `y`。
- `loss_total = loss + gnn_priority_aux_weight * BCE(scores, labels)`
- labels 为节点 0/1 标签（1=目标树，0=噪声）。

#### 3.3.7 数据侧

- `morphology_node.py` 的 `_preprocess_features` 中：
  - `dist_from_root`：从 `x[:, 2]`（原始 dist）取，不可达（<0）置 1e9
  - `is_target_root`：`(node_type == 1).float()`
- **首次使用需删除 processed 缓存**：`$DATADIR/processed/morphology_processed.pt`，以触发重新预处理并写入 dist/root 字段。

---

## 四、调用方法

### 4.1 配置项

| 配置键 | 含义 | 默认 |
|--------|------|------|
| gt.layer_type | 如 CustomGatedGCN+Mamba、CustomGatedGCN+Mamba_GNNPriorityBFS | 见 YAML |
| gt.layers | GPS 层数（10 = 10+10） | 10 |
| gt.gnn_priority_aux_weight | 辅助 BCE 权重 | 0.5 |
| gt.gnn_priority_pe_dim | dist 正弦 PE 维度 | 16 |

### 4.2 命令行

#### 使用预设（推荐）

```bash
# BFS 改进（10+10 + Mamba_GNNPriorityBFS，与 Baseline A 参数一致）
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --wandb False \
  --baseline bfs
```

#### 手动指定

```bash
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override gt.layers 10 \
  --name_tag baseline_10_10_gnn_priority_bfs
```

#### 调整 GNN Priority 参数

```bash
python scripts/run_graph_mamba.py --data_dir /path/to/synthesis_data \
  --baseline bfs \
  --override gt.gnn_priority_aux_weight 0.5 \
  --override gt.gnn_priority_pe_dim 16
```

#### 完整示例（tiny50 快速测试）

```bash
export GNN_DATA_ROOT=/path/to/data_root
python scripts/run_graph_mamba.py \
  --data_dir /path/to/synthesis_data_tiny_50 \
  --baseline bfs \
  --override optim.max_epoch 3 \
  --override out_dir /path/to/graph_mamba_results
```

### 4.3 数据要求

- **MorphologyNode 数据集**：需包含 `x`（含 dist 列）、`y`（节点 0/1 标签）、`edge_index`。
- **dist_from_root / is_target_root**：由 `_preprocess_features` 自动构造；若使用旧 processed 缓存，需删除 `$DATADIR/processed/morphology_processed.pt` 以重新生成。

### 4.4 预设与 Baseline 关系

| 预设 | 说明 | 命令 |
|------|------|------|
| `--baseline 10_10` | Baseline A（普通 Mamba） | 默认 |
| `--baseline bfs` | BFS 改进（Mamba_GNNPriorityBFS） | 需 mamba-ssm |
| `--baseline 20_aligned` | Baseline B（20 层纯 GatedGCN） | 自动 --no-mamba |

详见 `config.py` 中 `GRAPH_MAMBA_BASELINE_*` 常量及 `docs/BASELINES.md`。

---

## 五、实验结论

### 5.1 全量训练结果（约 8000 图，200 epoch）

| 方法 | 最佳 epoch | Val acc | Test acc |
|------|------------|---------|----------|
| **BFS 改进（Mamba_GNNPriorityBFS）** | **74** | **92.16%** | **91.95%** |
| Baseline A（10+10） | 43 | 88.21% | 88.14% |
| Baseline B（20 层对齐） | 40 | 85.87% | 84.73% |

BFS 改进相对 Baseline A 提升约 **+4% Val / +3.8% Test**。

### 5.2 实现要点总结

- **LexSort 路径**：有 dist 时使用，GPU 向量化，避免 CPU 瓶颈。
- **BFS 不连通处理**：未访问节点按 score 降序排在末尾，保证 perm 覆盖全部节点。
- **辅助损失**：使 MLP 学会预测树/噪声，弥补 BFS/LexSort 不可导。
- **数据预处理**：dist_from_root、is_target_root 需在 MorphologyNode 中正确构造并清缓存。

---

## 六、小结

- **思路**：通过启发式扫描（固定 BFS/度 → GNN 引导 Priority BFS / LexSort）使 Mamba 输入序列“树优先、噪声靠后”，用辅助 BCE 训练打分 MLP，并用形态学 dist/root 做 LexSort 与 MLP 输入增强。
- **实现**：`gps_layer.py` 中按 `global_model_type` 分支；Mamba_GNNPriorityBFS 含 MLP、LexSort/Priority BFS、辅助损失与数据字段的完整链路；BFS 支持不连通图。
- **调用**：通过 `gt.layer_type`、`gt.gnn_priority_*` 与 `--override` 或 `--baseline bfs` 即可切换 baseline 与 BFS。
