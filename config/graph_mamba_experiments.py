# Graph-Mamba 超参实验预设（供 scripts/run_graph_mamba_experiments.py 使用）
# 每项: name_tag 唯一标识（作为 out_dir 子目录名）; overrides: 覆盖 YAML 的 cfg 键值对。
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
