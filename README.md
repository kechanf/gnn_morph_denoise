# GNN 神经元形态去噪项目

基于 GCN 的神经元 SWC 图节点二分类：区分「目标树」与「干扰树」节点，用于形态学去噪。

## 项目结构

```
gnn_project/
├── config.py              # 路径与超参数（可改或通过 GNN_DATA_ROOT 覆盖）
├── data/                  # 数据相关
│   ├── resample.py        # SWC 按步长重采样
│   ├── synthesis.py       # 目标树 + 干扰树合成、合并近距节点、转 GNN Data
│   ├── dataset.py         # PyG TreeDataset（.pt 图 + 预处理）
│   └── swc_io.py          # NetworkX 树 ↔ SWC 文件
├── models/
│   └── gcn.py             # 2 层 GCN 节点分类模型
├── utils/
│   ├── features.py        # 特征预处理（One-Hot + 归一化）
│   └── graph_build.py     # 从预测掩码构建 MST 导出 SWC
├── scripts/               # 入口脚本（需在项目根目录执行）
│   ├── resample_swc.py    # Step 1: 重采样
│   ├── synthesis_data.py # Step 2: 生成合成数据
│   ├── train.py           # Step 3: 训练
│   └── test.py            # Step 4: 推理并导出 SWC
├── requirements.txt
└── README.md
```

## 环境与依赖

- Python 3.8+
- PyTorch、PyTorch Geometric、NetworkX、NumPy、SciPy、scikit-learn、pandas、tqdm
- `neuroutils`（SWC 解析与几何，需自行安装）

安装示例：

```bash
pip install -r requirements.txt
# 再安装 neuroutils（若为私有包则从对应源安装）
```

## 数据路径

默认数据根目录为 `/data2/kfchen/tracing_ws/morphology_seg`。可通过环境变量覆盖：

```bash
export GNN_DATA_ROOT=/path/to/your/data
```

或在 `config.py` 中直接修改 `DATA_ROOT` 及各子路径。

## 使用流程

在**项目根目录**下执行：

```bash
# 1. 对 SWC 按 10μm 重采样
python scripts/resample_swc.py

# 2. 生成合成图数据（目标树 + 多棵干扰树 → .pt）
python scripts/synthesis_data.py

# 3. 训练 GCN，最佳模型保存到 config.MODEL_SAVE_PATH
python scripts/train.py

# 4. 推理并导出 input / pred / gt 三种 SWC
python scripts/test.py

# 额外：生成样本用于观察不同合成策略的效果
python scripts/generate_samples.py
```

测试阶段默认处理 `TEST_DATA_DIR` 下全部 `.pt`；若只跑前 N 个，在 `config.py` 中设置 `TEST_NUM_FILES = N`。

## 配置说明

- **重采样**: `RESAMPLE_INPUT_DIR` / `RESAMPLE_OUTPUT_DIR` / `RESAMPLE_STEP_UM`
- **合成**: `SWC_POOL_DIR` / `SYNTHESIS_OUTPUT_DIR` / `SYNTHESIS_NUM_SAMPLES` / `SYNTHESIS_NUM_INTERFER_RANGE` / `MERGE_DIST_THRESHOLD`
- **训练**: `TRAIN_DATA_DIR` / `MODEL_SAVE_PATH` / `IN_CHANNELS` / `HIDDEN_CHANNELS` / `BATCH_SIZE` / `NUM_EPOCHS` 等
- **测试**: `TEST_DATA_DIR` / `TEST_OUTPUT_DIR` / `TEST_NUM_FILES` / `MST_K_NEIGHBORS`

## 数据合成策略

项目支持多种合成策略（见 `STRATEGIES.md`）：
- **`full_tree`**: 整棵干扰树（默认，原策略）
- **`local_spur`**: 局部短刺状假分支
- **`branch_segment`**: 干扰树的一段路径
- **`small_cluster`**: 小点簇（模拟成像伪影）
- **`mixed`**: 随机混合所有策略

在 `scripts/synthesis_data.py` 或调用 `data.synthesis.generate_dataset()` 时可通过 `strategy` 参数指定。

## 旧版入口（可选）

原单文件入口 `1_resample_swc.py`、`2_synthesis_data.py`、`3_train.py`、`4_test.py` 已由 `scripts/` 下四个脚本替代，逻辑集中在 `data/`、`models/`、`utils/` 中，便于复用与测试。若不再需要可自行删除或移出项目目录。
