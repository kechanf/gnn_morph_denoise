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
