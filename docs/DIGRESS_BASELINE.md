# DiGress 基线部署与运行说明

[DiGress](https://github.com/cvignac/DiGress)（Discrete Denoising diffusion for graph generation）已作为本项目的另一 baseline，放在 `external/DiGress`。

## 环境要求

- Python 3.9
- Conda（推荐）
- CUDA 11.8（若用 GPU）
- 官方测试组合：PyTorch 2.0.1 + torch_geometrics 2.3.1

## 一键部署

在 **gnn_project 根目录** 下执行：

```bash
bash scripts/install_digress.sh
```

脚本会：

1. 若 `external/DiGress` 不存在，则克隆 DiGress 仓库。
2. 创建 conda 环境 `digress`（含 rdkit、graph-tool）。
3. 安装 PyTorch、`requirements.txt` 及 `pip install -e external/DiGress`。
4. 编译 `src/analysis/orca/orca.cpp`（用于图统计评估）。

若网络导致克隆超时，可先手动克隆再重新运行脚本：

```bash
git clone https://github.com/cvignac/DiGress.git external/DiGress
bash scripts/install_digress.sh
```

## 运行 DiGress

激活环境并进入 DiGress 目录：

```bash
conda activate digress
cd external/DiGress
```

- **调试（建议先跑）**  
  `python3 main.py +experiment=debug.yaml`

- **离散模型（默认）**  
  `python3 main.py`

- **连续模型**  
  `python3 main.py model=continuous`

- **指定数据集**  
  `python3 main.py dataset=guacamol`  
  可用数据集见 `configs/dataset/`。

- **仅跑少量 batch**  
  `python3 main.py general.name=test`

更多参数覆盖见 [Hydra 文档](https://hydra.cc/)。

## 与本项目数据对接（后续）

DiGress 支持自定义数据集：在 `src/datasets` 下新增数据集类（可参考 `moses_dataset.py` 或 `spectre_datasets.py`），实现 `Dataset` 与 `DatasetInfos`，并在 `configs/dataset` 中增加配置。若要把本项目的形态学图数据接进去，需要：

- 将现有图数据转为 PyG `Data` 格式；
- 在 DiGress 中实现对应的 `Dataset` 与 `DatasetInfos`，并写好 config。

## 参考

- 仓库：<https://github.com/cvignac/DiGress>
- 论文：DiGress: Discrete Denoising diffusion for graph generation (ICLR 2023)
