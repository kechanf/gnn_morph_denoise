# Graph-Mamba 适配与训练说明

## 已完成的修改（适配本项目）

1. **数据读取**
   - `external/Graph-Mamba/graphgps/loader/dataset/morphology_node.py`：从目录加载 `.pt`，预处理与 `utils/features.py` 一致（6 维特征）。
   - `external/Graph-Mamba/graphgps/loader/master_loader.py`：增加 `PyG-MorphologyNode` 分支及 `preformat_MorphologyNode()`，按 0.8/0.1/0.1 划分 train/val/test。
   - 数据集增加 `_data` 别名，兼容 GraphGym 的 `set_dataset_info`。

2. **无需 Mamba-SSM 即可训练**
   - `graphgps/layer/gps_layer.py`：`mamba_ssm` 改为 try/except 延迟导入，未安装时仅不能用 Mamba 层。
   - `graphgps/train/custom_train.py`：`deepspeed` 改为可选。
   - `main.py`：移除已不存在的 `set_agg_dir` 导入。
   - `graphgps/config/defaults_config.py`：增加 `cfg.train.mode`，避免新版 PyG 报错。

3. **配置与入口**
   - `configs/Mamba/morphology-node-EX.yaml`：完整 Graph-Mamba（需 mamba-ssm）。
   - `configs/Mamba/morphology-node-GatedGCN-only.yaml`：仅 GatedGCN + LapPE，**不需 mamba-ssm**。
   - `scripts/run_graph_mamba.py`：`--no-mamba` 使用 GatedGCN-only 配置；依赖缺失时会提示安装。

4. **减配与补全（相对完整 Graph-Mamba）**
   - **GatedGCN-only** 相对 EX 的减配：无 Mamba/gt、`custom_gnn`、`edge_encoder: False`、`node_encoder_bn: False`。
   - **已在 EX 中补上**：`morphology-node-EX.yaml` 现启用 `node_encoder_bn: True`、`edge_encoder: True`、`edge_encoder_name: LinearEdge`、`edge_encoder_bn: True`；`graphgps/encoder/linear_edge_encoder.py` 已支持 `morphology-node` 的 6 维边特征。用 EX 且不传 `--no-mamba` 即使用完整模型 + 上述编码器。

## 环境（已用 medsam 环境跑通）

在 **medsam** 环境中已安装并验证可用的包：

- torch, torch-geometric, torch_scatter  
- ogb, pyyaml, yacs, tensorboardx, torchmetrics  
- performer-pytorch  

未安装：mamba-ssm（可选，仅完整 Mamba 配置需要）。

## 开始训练

在 **gnn_project 根目录** 下执行：

```bash
# 激活含 PyTorch + PyG 的环境（例如 medsam）
conda activate medsam

# 使用 GatedGCN-only（不需 mamba-ssm），数据目录来自 config.TRAIN_DATA_DIR
python scripts/run_graph_mamba.py --no-mamba --wandb False

# 指定数据目录
python scripts/run_graph_mamba.py --no-mamba --data_dir /data2/kfchen/tracing_ws/morphology_seg/synthesis_data --wandb False
```

首次运行会：

1. 在数据目录下生成 `processed/morphology_processed.pt`（预处理并缓存）。
2. 计算 LapPE 等，可能耗时数分钟。
3. 训练 200 个 epoch，结果与 checkpoint 在 `external/Graph-Mamba/results/morphology-node-GatedGCN-only/42/`（seed=42）。

若已安装 **mamba-ssm**，可去掉 `--no-mamba` 使用完整 Graph-Mamba：

```bash
python scripts/run_graph_mamba.py --wandb False
```

## 数据要求

- 目录下为多个 `.pt` 文件，每个为 PyG `Data`：`x`（4 维：Radius, Type, Dist, Angle）、`y`（节点 0/1）、`edge_index`、可选 `pos`。
- 当前 `config.TRAIN_DATA_DIR` 下有 2000 个 `.pt`，已满足要求。
