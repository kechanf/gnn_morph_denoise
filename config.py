"""
全局配置：路径、数据合成/增强参数、训练与推理超参数。

使用建议：
- **仅改这里** 来控制数据合成和训练过程，不要在代码里硬编码参数。
- 可以通过环境变量 `GNN_DATA_ROOT` 快速切换数据根目录。
"""
import os

# ---------- 项目根目录 ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================
# 路径 & 数据根目录
# =========================

# 数据根目录（所有数据相关路径都基于此构造）：
# - 可以通过环境变量 GNN_DATA_ROOT 覆盖默认路径
# - 用于存放：重采样 SWC、合成 .pt、训练模型、可视化结果等
DATA_ROOT = os.environ.get("GNN_DATA_ROOT", "/data2/kfchen/tracing_ws/morphology_seg")

# ---------- Step 1: SWC 重采样 ----------
# 原始 SWC 输入目录（一般为 1 μm 或原始分辨率）
RESAMPLE_INPUT_DIR = os.path.join(DATA_ROOT, "auto8.4k_0510_resample1um_mergedBranches0712")
# 重采样步长（单位：μm），数值越大，点越稀疏
RESAMPLE_STEP_UM = 10
# 重采样后 SWC 输出目录
RESAMPLE_OUTPUT_DIR = os.path.join(DATA_ROOT, f"auto8k_resampled_{RESAMPLE_STEP_UM}um")

# =========================
# 数据合成 & 噪声/增强参数
# =========================

# ---------- Step 2: 合成数据（基础） ----------
# 合成时使用的 SWC 池（作为 target 与 seed），一般直接用重采样后的目录
SWC_POOL_DIR = os.path.join(DATA_ROOT, f"auto8k_resampled_{RESAMPLE_STEP_UM}um")
# 合成输出的 .pt 图目录
SYNTHESIS_OUTPUT_DIR = os.path.join(DATA_ROOT, "synthesis_data")
# 一次跑 scripts/synthesis_data.py 生成多少个样本
SYNTHESIS_NUM_SAMPLES = 2000
# 每个样本中干扰树（simple negative）的数量范围（闭区间）
SYNTHESIS_NUM_INTERFER_RANGE = (5, 10)
# 合并近距节点的距离阈值（单位：与坐标同尺度，通常是 μm）
MERGE_DIST_THRESHOLD = 1.0
# 单个样本合成超时（秒）：若单次 generate_dataset 超过此时间未返回，则跳过该样本并继续下一个
# 设为 None 或 0 表示不设超时
SYNTHESIS_SAMPLE_TIMEOUT_SEC = 120

# ---------- 各合成/噪声策略的参数 ----------

# local_spur（局部短刺假分支）
# - 在目标树现有节点上“长出”一段线性小分支
# - 用来模拟追踪时误产生的短假分支
LOCAL_SPUR_CFG = {
    # 假分支长度（节点数）的随机范围
    "length_range": (3, 8),
    # 假分支沿方向前进的步长范围（单位：μm）
    "step_um_range": (3.0, 8.0),
    # 根部最大半径（越大越粗），沿分支逐渐衰减
    "max_radius": 2.0,
}

# branch_segment（干扰树分支片段）
# - 从 seed SWC 中截取一段真实路径，贴到目标树附近
BRANCH_SEGMENT_CFG = {
    # 路径长度（节点数）的随机范围
    "length_range": (3, 10),
}

# small_cluster（小点簇伪影）
# - 在目标树节点附近生成小点簇（MST 连接），模拟局部成像伪影
SMALL_CLUSTER_CFG = {
    # 簇中节点数量范围
    "num_points_range": (3, 8),
    # 簇的空间半径范围（单位：μm），值越大簇越“散”
    "radius_um_range": (1.0, 5.0),
}

# break_fragment（断裂 + 碎片噪声，碎片来自真实 SWC 片段）
# - 在目标树内部随机断开多条边，制造“真断点”
# - 从 seed SWC 中裁剪多个真实分支片段，平移/旋转后丢在断点附近
# - 在空间上“较近”的目标节点 / 断裂端点上建立错误连接，模拟复杂错误拓扑
BREAK_FRAGMENT_CFG = {
    # 每次注入时断开的边数量范围（越大断裂越多）
    "num_break_edges_range": (3, 8),
    # 注入的碎片（真实分支片段）数量范围（越大噪声越多）
    "num_fragments_range": (1, 5),
    # 把碎片平移到断点附近时的“额外抖动半径”范围（单位：μm）
    "jitter_radius_um_range": (1.0, 5.0),
    # 碎片根节点在此半径内寻找“近邻目标节点”，建立错误连接（单位：μm）
    "near_origin_radius_um": 5.0,
    # 碎片根节点在此半径内寻找“近邻断点端”，再建立额外错误连接（单位：μm）
    "near_break_radius_um": 5.0,
}

# mixed 策略控制
MIXED_CFG = {
    # 对于一个样本，最少注入多少次“噪声操作”（一次操作 = 选择一个策略并应用）
    "min_injections": 20,
}

# ---------- 新连接几何约束（角度） ----------
# 对“新建立的连接”（分支连接 / 树连接 / 碎片连接等），
# 控制其与已有分支方向的夹角上限（单位：度）。
# 实现方式：在 [0, ANGLE_CONSTRAINT_DEG] 内随机采样一个角度，
#          并旋转新片段，使其与参考方向的夹角等于该随机角度。
ANGLE_CONSTRAINT_DEG = 30.0
# 数值稳定性用的小 epsilon，避免零向量和浮点误差
ANGLE_CONSTRAINT_EPS = 1e-6

# 简单负样本策略（旧策略）使用比值（权重，只有相对大小有意义）
# 用于 mixed 策略下控制 full_tree/local_spur/branch_segment/small_cluster 的采样频率
SIMPLE_NEG_STRATEGY_WEIGHTS = {
    # 这些都是“简单负样本”，权重刻意设得较小，
    # 让它们偶尔出现，主体仍由高级 break_fragment 控制
    "full_tree": 0.1,
    "local_spur": 0.1,
    "branch_segment": 0.1,
    "small_cluster": 0.1,
}

# 高级数据增强策略（例如断裂与碎片噪声）的使用比值
ADV_STRATEGY_WEIGHTS = {
    # break_fragment 占主导地位
    "break_fragment": 1.0,
}

# break_fragment 样本可视化脚本参数
BREAK_FRAGMENT_SAMPLE_NUM = 10        # 生成多少个样本供观察
BREAK_FRAGMENT_SAMPLE_NUM_SEED_SWCS = 10  # 作为碎片 seed 的 SWC 数

# =========================
# 训练 / 测试 / 推理 参数
# =========================

# ---------- Step 3 & 4: 训练与测试（数据路径） ----------
# 训练使用的 .pt 图目录（通常就是合成输出目录）
TRAIN_DATA_DIR = os.path.join(DATA_ROOT, "synthesis_data")
# 仅含 50 个样本的小数据集目录（用于快速验证训练流程，由 scripts/create_tiny_dataset.py 生成）
TRAIN_DATA_DIR_TINY_50 = os.path.join(DATA_ROOT, "synthesis_data_tiny_50")
# 测试/验证使用的 .pt 图目录（可与训练目录不同，用于 cross-dataset 实验）
TEST_DATA_DIR = os.path.join(DATA_ROOT, "synthesis_data")
# 训练得到的最佳模型保存路径（仅保存 state_dict）
MODEL_SAVE_PATH = os.path.join(DATA_ROOT, "best_model.pth")
# 推理阶段导出的 SWC 结果目录（input / pred / gt）
TEST_OUTPUT_DIR = os.path.join(DATA_ROOT, "results_swc")
# Graph-Mamba 训练输出目录（结果、checkpoint、日志等，统一在 DATA_ROOT 下）
GRAPH_MAMBA_OUT_DIR = os.path.join(DATA_ROOT, "graph_mamba_results")

# ---------- Graph-Mamba 两种统一 baseline（对比实验固定） ----------
# 训练/优化超参一致，仅结构不同。详见 docs/BASELINES.md。
# Baseline A：10+10（EX，10 层 GatedGCN+Mamba）
GRAPH_MAMBA_BASELINE_10_10_NAME = "baseline_10_10"
GRAPH_MAMBA_BASELINE_10_10_OVERRIDES = {"gt.layers": 10}  # 使用 morphology-node-EX.yaml，不传 --no-mamba
# Baseline B：20 层纯 GatedGCN（对齐 A 的超参）
GRAPH_MAMBA_BASELINE_20_ALIGNED_NAME = "baseline_20_aligned"
GRAPH_MAMBA_BASELINE_20_ALIGNED_OVERRIDES = {
    "gnn.layers_mp": 20,
    "gnn.dim_inner": 96,
    "gnn.dropout": 0.0,
    "optim.base_lr": 0.001,
    "optim.weight_decay": 0.01,
    "train.batch_size": 32,
}  # 使用 morphology-node-GatedGCN-only.yaml，--no-mamba

# ---------- 旧版：s5_combo_all（20 层 + 单独调参，非对齐 baseline） ----------
GRAPH_MAMBA_FINAL_BASELINE = "s5_combo_all"
GRAPH_MAMBA_FINAL_BASELINE_OVERRIDES = {
    "gnn.layers_mp": 20,
    "gnn.dim_inner": 192,
    "gnn.dropout": 0.15,
    "optim.base_lr": 0.0025,
    "optim.weight_decay": 0.02,
    "train.batch_size": 16,
}

# =========================
# Graph-Mamba 超参实验预设（用于 scripts/run_graph_mamba_experiments.py）
# =========================
# 每项: name_tag 唯一标识（会作为 out_dir 子目录名）;
#       overrides: 覆盖 YAML 的键值对，键为 cfg 路径如 "gnn.dropout"、"optim.base_lr"。
# 分组：学习率 | Dropout | 隐藏维 | 层数 | 权重衰减 | batch_size | LapPE 维度
GRAPH_MAMBA_EXPERIMENT_PRESETS = [
    # ----- Baseline（与当前 YAML 一致） -----
    {"name_tag": "baseline", "overrides": {}},
    # ----- 1. 学习率 -----
    {"name_tag": "lr_5e-4", "overrides": {"optim.base_lr": 0.0005}},
    {"name_tag": "lr_2e-3", "overrides": {"optim.base_lr": 0.002}},
    {"name_tag": "lr_3e-3", "overrides": {"optim.base_lr": 0.003}},
    # ----- 2. Dropout -----
    {"name_tag": "drop_0.1", "overrides": {"gnn.dropout": 0.1}},
    {"name_tag": "drop_0.2", "overrides": {"gnn.dropout": 0.2}},
    {"name_tag": "drop_0.3", "overrides": {"gnn.dropout": 0.3}},
    {"name_tag": "drop_0.5", "overrides": {"gnn.dropout": 0.5}},
    # ----- 3. 隐藏维度 (dim_inner) -----
    {"name_tag": "dim_64", "overrides": {"gnn.dim_inner": 64}},
    {"name_tag": "dim_128", "overrides": {"gnn.dim_inner": 128}},
    {"name_tag": "dim_192", "overrides": {"gnn.dim_inner": 192}},
    # ----- 4. GNN 层数 -----
    {"name_tag": "layers_2", "overrides": {"gnn.layers_mp": 2}},
    {"name_tag": "layers_3", "overrides": {"gnn.layers_mp": 3}},
    {"name_tag": "layers_6", "overrides": {"gnn.layers_mp": 6}},
    # ----- 5. 权重衰减 -----
    {"name_tag": "wd_0", "overrides": {"optim.weight_decay": 0.0}},
    {"name_tag": "wd_0.001", "overrides": {"optim.weight_decay": 0.001}},
    {"name_tag": "wd_0.05", "overrides": {"optim.weight_decay": 0.05}},
    # ----- 6. Batch size -----
    {"name_tag": "bs_16", "overrides": {"train.batch_size": 16}},
    {"name_tag": "bs_64", "overrides": {"train.batch_size": 64}},
    # ----- 7. LapPE 维度 -----
    {"name_tag": "pe_dim_8", "overrides": {"posenc_LapPE.dim_pe": 8}},
    {"name_tag": "pe_dim_32", "overrides": {"posenc_LapPE.dim_pe": 32}},
]

# 测试时最多处理多少个 .pt 文件：
# - None: 全部
# - 整数: 只取前 N 个，便于快速调试
TEST_NUM_FILES = None

# ---------- 模型与训练超参数 ----------
# 输入特征维度（本项目固定为 6：半径+类型 one‑hot+距离+角度）
IN_CHANNELS = 6
# GCN 隐藏层通道数（越大容量越强，但更耗显存）
HIDDEN_CHANNELS = 64
# 输出类别数（本任务是二分类：目标 vs 干扰）
OUT_CHANNELS = 2
# Dropout 比例（0 表示不做 dropout）
DROPOUT = 0.5
# 学习率
LEARNING_RATE = 0.001
# batch 中包含的图数量（越大梯度越稳定，但显存占用更高）
BATCH_SIZE = 4
# 训练集占总数据比例（1 - TRAIN_RATIO 为测试集比例）
TRAIN_RATIO = 0.8
# 训练总轮数（完整 scripts/train.py 中使用）
NUM_EPOCHS = 51
# 以哪个指标选择“最佳模型”：
# - "test": 使用测试集准确率
# - "train": 使用训练集准确率
SAVE_BEST_BASED_ON = "test"

# ---------- 推理 / 图重建 ----------
# 在测试阶段，从预测掩码重建树结构时使用的 KNN 中的 k：
# - k 越大，MST 的连通性越强，但可能引入不自然的远距离连接
MST_K_NEIGHBORS = 15

# =========================
# 对照实验预设 (EXPERIMENT_PRESETS)
# =========================
# 供 scripts/run_experiments.py 批量跑实验；Graph-Mamba 超参实验见 run_graph_mamba_experiments.py。每项会覆盖上面的模型/训练超参，
# 并可将最佳模型保存到 DATA_ROOT/checkpoints/{run_name}.pth。
#
# 字段说明：
#   run_name      : 唯一标识，用于保存模型和日志
#   model_type    : "gcn" | "gat" | "sage" | "gin"
#   num_layers    : GNN 卷积层数 (1~4)
#   hidden_channels : 隐藏维度
#   dropout       : Dropout 比例
#   learning_rate : 可选，不写则用上面 LEARNING_RATE
#
# 分组：深度 | 宽度 | Dropout | 架构 | 学习率

EXPERIMENT_PRESETS = [
    # ----- 1. 深度对照 (Depth)：固定 H64 D0.5，只改层数 -----
    {"run_name": "gcn_L1_H64_D0.5", "model_type": "gcn", "num_layers": 1, "hidden_channels": 64, "dropout": 0.5},
    {"run_name": "gcn_L2_H64_D0.5", "model_type": "gcn", "num_layers": 2, "hidden_channels": 64, "dropout": 0.5},  # Baseline
    {"run_name": "gcn_L3_H64_D0.5", "model_type": "gcn", "num_layers": 3, "hidden_channels": 64, "dropout": 0.5},
    {"run_name": "gcn_L4_H64_D0.5", "model_type": "gcn", "num_layers": 4, "hidden_channels": 64, "dropout": 0.5},
    # ----- 2. 宽度对照 (Width)：固定 2 层 D0.5，只改 hidden -----
    {"run_name": "gcn_L2_H32_D0.5", "model_type": "gcn", "num_layers": 2, "hidden_channels": 32, "dropout": 0.5},
    {"run_name": "gcn_L2_H128_D0.5", "model_type": "gcn", "num_layers": 2, "hidden_channels": 128, "dropout": 0.5},
    # ----- 3. Dropout 对照 (Regularization)：固定 2 层 H64 -----
    {"run_name": "gcn_L2_H64_D0.0", "model_type": "gcn", "num_layers": 2, "hidden_channels": 64, "dropout": 0.0},
    {"run_name": "gcn_L2_H64_D0.3", "model_type": "gcn", "num_layers": 2, "hidden_channels": 64, "dropout": 0.3},
    # ----- 4. 架构对照 (Arch)：2 层 H64 D0.5，GCN / GAT / SAGE / GIN -----
    {"run_name": "gat_L2_H64_D0.5", "model_type": "gat", "num_layers": 2, "hidden_channels": 64, "dropout": 0.5},
    {"run_name": "sage_L2_H64_D0.5", "model_type": "sage", "num_layers": 2, "hidden_channels": 64, "dropout": 0.5},
    {"run_name": "gin_L2_H64_D0.5", "model_type": "gin", "num_layers": 2, "hidden_channels": 64, "dropout": 0.5},
    # ----- 5. 学习率对照 (LR)：固定 GCN 2 层 H64 D0.5 -----
    {"run_name": "gcn_L2_H64_D0.5_LR5e-4", "model_type": "gcn", "num_layers": 2, "hidden_channels": 64, "dropout": 0.5, "learning_rate": 0.0005},
    {"run_name": "gcn_L2_H64_D0.5_LR2e-3", "model_type": "gcn", "num_layers": 2, "hidden_channels": 64, "dropout": 0.5, "learning_rate": 0.002},
]
