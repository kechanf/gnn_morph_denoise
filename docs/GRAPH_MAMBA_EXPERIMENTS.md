# Graph-Mamba 超参实验设计

用于在形态学节点分类任务上系统搜索最佳超参数，对照 baseline 已跑通配置。

## 实验分组与变量

| 分组 | 变量 | 预设 name_tag | 说明 |
|------|------|----------------|------|
| Baseline | - | `baseline` | 与当前 YAML 一致（lr=1e-3, dropout=0, dim=96, layers=4, wd=0.01, bs=32, pe_dim=16） |
| 学习率 | `optim.base_lr` | `lr_5e-4`, `lr_2e-3`, `lr_3e-3` | 0.0005, 0.002, 0.003 |
| Dropout | `gnn.dropout` | `drop_0.1` ~ `drop_0.5` | 0.1, 0.2, 0.3, 0.5 |
| 隐藏维 | `gnn.dim_inner` | `dim_64`, `dim_128`, `dim_192` | 64, 128, 192 |
| 层数 | `gnn.layers_mp` | `layers_2`, `layers_3`, `layers_6` | 2, 3, 6 |
| 权重衰减 | `optim.weight_decay` | `wd_0`, `wd_0.001`, `wd_0.05` | 0, 0.001, 0.05 |
| Batch size | `train.batch_size` | `bs_16`, `bs_64` | 16, 64 |
| LapPE 维度 | `posenc_LapPE.dim_pe` | `pe_dim_8`, `pe_dim_32` | 8, 32 |

## 如何运行

### 跑全部预设（约 22 组，每组 200 epoch，耗时长）

```bash
cd /path/to/gnn_project
python scripts/run_graph_mamba_experiments.py
```

### 只跑部分预设

```bash
python scripts/run_graph_mamba_experiments.py --presets baseline drop_0.2 lr_5e-4 dim_128
```

### 多 seed 取平均（推荐用于最终选参）

```bash
python scripts/run_graph_mamba_experiments.py --repeat 3
```

### 指定数据目录与结果 CSV

```bash
python scripts/run_graph_mamba_experiments.py --data_dir /data2/kfchen/.../synthesis_data --csv /data2/kfchen/.../graph_mamba_experiments.csv
```

## 结果输出

- 每个 preset 对应输出子目录：`{GRAPH_MAMBA_OUT_DIR}/morphology-node-GatedGCN-only-{name_tag}/`
- 汇总 CSV 默认：`{DATA_ROOT}/graph_mamba_experiments.csv`，列包括 `name_tag`、`overrides`、`best_val_accuracy`、`best_test_accuracy`、`best_val_auc`、`best_test_auc`、`timestamp`。

## 单次跑带覆盖（不跑批量）

```bash
python scripts/run_graph_mamba.py --no-mamba --wandb False \
  --name_tag my_exp \
  --override gnn.dropout 0.2 \
  --override optim.base_lr 0.0005
```

## 建议流程

1. 先跑 **baseline** 与少数几组（如 `lr_5e-4`, `drop_0.2`, `dim_128`）确认脚本与 CSV 正常。
2. 再跑全部预设，根据 CSV 中 `best_test_accuracy` / `best_test_auc` 挑出最优 name_tag。
3. 对最优方向做细调（例如最优 lr 附近再扫几个点），或对最优组合做 `--repeat 3` 取均值与标准差。
