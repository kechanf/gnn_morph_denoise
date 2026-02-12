"""
批量跑 Graph-Mamba 超参实验，确定最佳参数。每个 preset 用不同 name_tag 与 cfg 覆盖跑一次，结果汇总到 CSV。

用法（在项目根目录）:
  python scripts/run_graph_mamba_experiments.py
  python scripts/run_graph_mamba_experiments.py --presets baseline drop_0.2 lr_5e-4   # 只跑指定
  python scripts/run_graph_mamba_experiments.py --repeat 2   # 每 preset 跑 2 个 seed
  # 只跑第三阶段（层数递增，以 combo_layers_7 为基线）:
  # 例如先跑 7~10 层:
  #   python scripts/run_graph_mamba_experiments.py --presets s3_layers_7 s3_layers_8 s3_layers_9 s3_layers_10 --csv <DATA_ROOT>/graph_mamba_experiments_stage3.csv
  # 如需继续加深，可再追加 s3_layers_11 s3_layers_12:
  #   python scripts/run_graph_mamba_experiments.py --presets s3_layers_11 s3_layers_12 --csv <DATA_ROOT>/graph_mamba_experiments_stage3_deep.csv
  # 第四阶段（20 层 baseline 调参）:
  #   python scripts/run_graph_mamba_experiments.py --presets s4_baseline s4_lr_2_5e-3 s4_lr_3_5e-3 s4_drop_0_05 s4_drop_0_15 s4_dim_160 s4_dim_224 s4_wd_0_005 s4_wd_0_02 s4_bs_24 --csv <DATA_ROOT>/graph_mamba_experiments_stage4.csv
  # 第五阶段（20 层 baseline 组合验证）:
  #   python scripts/run_graph_mamba_experiments.py --presets s5_combo_wd0.02_drop0.15 s5_combo_wd0.02_lr2.5e-3 s5_combo_drop0.15_lr2.5e-3 s5_combo_all --csv <DATA_ROOT>/graph_mamba_experiments_stage5_combo.csv
"""
import sys
import os
import argparse
import csv
import json
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_MAMBA_ROOT = os.path.join(PROJECT_ROOT, "external", "Graph-Mamba")
CONFIG_GATEDGCN = "configs/Mamba/morphology-node-GatedGCN-only.yaml"

# 实验预设：name_tag 唯一；overrides 覆盖 YAML 的 cfg 键值对
GRAPH_MAMBA_EXPERIMENT_PRESETS = [
    {"name_tag": "baseline", "overrides": {}},
    {"name_tag": "lr_5e-4", "overrides": {"optim.base_lr": 0.0005}},
    {"name_tag": "lr_2e-3", "overrides": {"optim.base_lr": 0.002}},
    {"name_tag": "lr_3e-3", "overrides": {"optim.base_lr": 0.003}},
    {"name_tag": "drop_0.1", "overrides": {"gnn.dropout": 0.1}},
    {"name_tag": "drop_0.2", "overrides": {"gnn.dropout": 0.2}},
    {"name_tag": "drop_0.3", "overrides": {"gnn.dropout": 0.3}},
    {"name_tag": "drop_0.5", "overrides": {"gnn.dropout": 0.5}},
    {"name_tag": "dim_64", "overrides": {"gnn.dim_inner": 64}},
    {"name_tag": "dim_128", "overrides": {"gnn.dim_inner": 128}},
    {"name_tag": "dim_192", "overrides": {"gnn.dim_inner": 192}},
    {"name_tag": "layers_2", "overrides": {"gnn.layers_mp": 2}},
    {"name_tag": "layers_3", "overrides": {"gnn.layers_mp": 3}},
    {"name_tag": "layers_6", "overrides": {"gnn.layers_mp": 6}},
    {"name_tag": "wd_0", "overrides": {"optim.weight_decay": 0.0}},
    {"name_tag": "wd_0.001", "overrides": {"optim.weight_decay": 0.001}},
    {"name_tag": "wd_0.05", "overrides": {"optim.weight_decay": 0.05}},
    {"name_tag": "bs_16", "overrides": {"train.batch_size": 16}},
    {"name_tag": "bs_64", "overrides": {"train.batch_size": 64}},
    {"name_tag": "pe_dim_8", "overrides": {"posenc_LapPE.dim_pe": 8}},
    {"name_tag": "pe_dim_32", "overrides": {"posenc_LapPE.dim_pe": 32}},
    # ================= 第二阶段：以新 baseline 组合为中心的 12 组精细实验 =================
    # 新 baseline（隐含在每个组合中）：
    #   gnn.layers_mp = 6
    #   gnn.dim_inner = 192
    #   gnn.dropout = 0.1
    #   optim.base_lr = 0.003
    #   optim.weight_decay = 0.01  (保持默认)
    #   train.batch_size = 16
    #
    # 学习率微调
    {
        "name_tag": "combo_lr_2e-3",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.002,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_lr_2_5e-3",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.0025,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_lr_3_5e-3",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.0035,
            "train.batch_size": 16,
        },
    },
    # Dropout 微调
    {
        "name_tag": "combo_drop_0",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.0,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_drop_0_05",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.05,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_drop_0_15",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.15,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    # 模型规模（宽度 / 深度）微调
    {
        "name_tag": "combo_dim_160",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 160,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_dim_224",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 224,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_layers_5",
        "overrides": {
            "gnn.layers_mp": 5,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_layers_7",
        "overrides": {
            "gnn.layers_mp": 7,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    # 正则与 batch size 微调
    {
        "name_tag": "combo_wd_0_005",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "optim.weight_decay": 0.005,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "combo_bs_24",
        "overrides": {
            "gnn.layers_mp": 6,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 24,
        },
    },
    # ================= 第三阶段：层数递增（combo_layers_7 为基线，其余对比暂缓） =================
    # 固定：dim_inner=192, dropout=0.1, base_lr=0.003, batch_size=16，只变 layers_mp
    {
        "name_tag": "s3_layers_7",
        "overrides": {
            "gnn.layers_mp": 7,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_8",
        "overrides": {
            "gnn.layers_mp": 8,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_9",
        "overrides": {
            "gnn.layers_mp": 9,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_10",
        "overrides": {
            "gnn.layers_mp": 10,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_11",
        "overrides": {
            "gnn.layers_mp": 11,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_12",
        "overrides": {
            "gnn.layers_mp": 12,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_13",
        "overrides": {
            "gnn.layers_mp": 13,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_14",
        "overrides": {
            "gnn.layers_mp": 14,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_15",
        "overrides": {
            "gnn.layers_mp": 15,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_16",
        "overrides": {
            "gnn.layers_mp": 16,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_17",
        "overrides": {
            "gnn.layers_mp": 17,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_18",
        "overrides": {
            "gnn.layers_mp": 18,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_19",
        "overrides": {
            "gnn.layers_mp": 19,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s3_layers_20",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    # ================= 第四阶段：以 20 层为新 baseline 的调参实验 =================
    # 基线：layers_mp=20, dim_inner=192, dropout=0.1, base_lr=0.003, batch_size=16
    {
        "name_tag": "s4_baseline",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_lr_2_5e-3",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.0025,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_lr_3_5e-3",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.0035,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_drop_0_05",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.05,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_drop_0_15",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.15,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_dim_160",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 160,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_dim_224",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 224,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_wd_0_005",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "optim.weight_decay": 0.005,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_wd_0_02",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "optim.weight_decay": 0.02,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s4_bs_24",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.003,
            "train.batch_size": 24,
        },
    },
    # ================= 第五阶段：20 层 baseline 的组合验证实验 =================
    # 组合方向来自第四阶段最优单项：wd=0.02、dropout=0.15、lr=0.0025
    {
        "name_tag": "s5_combo_wd0.02_drop0.15",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.15,
            "optim.base_lr": 0.003,
            "optim.weight_decay": 0.02,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s5_combo_wd0.02_lr2.5e-3",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.1,
            "optim.base_lr": 0.0025,
            "optim.weight_decay": 0.02,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s5_combo_drop0.15_lr2.5e-3",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.15,
            "optim.base_lr": 0.0025,
            "optim.weight_decay": 0.01,
            "train.batch_size": 16,
        },
    },
    {
        "name_tag": "s5_combo_all",
        "overrides": {
            "gnn.layers_mp": 20,
            "gnn.dim_inner": 192,
            "gnn.dropout": 0.15,
            "optim.base_lr": 0.0025,
            "optim.weight_decay": 0.02,
            "train.batch_size": 16,
        },
    },
]


def run_one(name_tag, overrides, out_dir, data_dir, repeat, env):
    cfg_path = os.path.join(GRAPH_MAMBA_ROOT, CONFIG_GATEDGCN)
    cmd = [
        sys.executable,
        os.path.join(GRAPH_MAMBA_ROOT, "main.py"),
        "--cfg", cfg_path,
        "--repeat", str(repeat),
        "out_dir", out_dir,
        "dataset.dir", data_dir,
        "wandb.use", "False",
        "name_tag", name_tag,
    ]
    for k, v in overrides.items():
        cmd.extend([k, str(v)])
    return subprocess.run(cmd, cwd=GRAPH_MAMBA_ROOT, env=env)


def read_best_metrics(base_out, run_name):
    """从 agg 或单 seed 目录读取 best val/test 指标。"""
    agg_test = os.path.join(base_out, run_name, "agg", "test", "best.json")
    agg_val = os.path.join(base_out, run_name, "agg", "val", "best.json")
    out = {}
    for split, path in [("test", agg_test), ("val", agg_val)]:
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    out[split] = json.load(f)
            except Exception:
                pass
    if not out:
        seed_dir = os.path.join(base_out, run_name, "42")
        if os.path.isdir(seed_dir):
            for split in ["val", "test"]:
                stats_path = os.path.join(seed_dir, split, "stats.json")
                if os.path.isfile(stats_path):
                    try:
                        with open(stats_path) as f:
                            stats_list = json.load(f)
                        if stats_list:
                            best = max(stats_list, key=lambda x: x.get("accuracy", 0))
                            out[split] = best
                    except Exception:
                        pass
    return out


def main():
    parser = argparse.ArgumentParser(description="Run Graph-Mamba hyperparameter experiments")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--presets", nargs="*", default=None, help="Only run these name_tags")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path (default: DATA_ROOT/graph_mamba_experiments.csv)")
    args = parser.parse_args()

    sys.path.insert(0, PROJECT_ROOT)
    import config as gnn_config
    out_dir = os.path.abspath(gnn_config.GRAPH_MAMBA_OUT_DIR)
    data_dir = os.path.abspath(args.data_dir or gnn_config.TRAIN_DATA_DIR)
    csv_path = args.csv or os.path.join(gnn_config.DATA_ROOT, "graph_mamba_experiments.csv")

    if not os.path.isdir(data_dir):
        print(f"Error: data_dir is not a directory: {data_dir}")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = GRAPH_MAMBA_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    config_basename = os.path.splitext(os.path.basename(CONFIG_GATEDGCN))[0]
    presets = [p for p in GRAPH_MAMBA_EXPERIMENT_PRESETS
               if args.presets is None or p["name_tag"] in args.presets]

    results = []
    for i, preset in enumerate(presets):
        name_tag = preset["name_tag"]
        overrides = preset.get("overrides", {})
        run_name = f"{config_basename}-{name_tag}"
        print(f"[{i+1}/{len(presets)}] Running name_tag={name_tag} overrides={overrides}")
        rc = run_one(name_tag, overrides, out_dir, data_dir, args.repeat, env)
        metrics = read_best_metrics(out_dir, run_name)
        row = {
            "name_tag": name_tag,
            "overrides": str(overrides),
            "exit_code": rc.returncode,
            "best_val_accuracy": metrics.get("val", {}).get("accuracy"),
            "best_test_accuracy": metrics.get("test", {}).get("accuracy"),
            "best_val_auc": metrics.get("val", {}).get("auc"),
            "best_test_auc": metrics.get("test", {}).get("auc"),
            "timestamp": datetime.now().isoformat(),
        }
        results.append(row)
        print(f"  -> val_acc={row['best_val_accuracy']} test_acc={row['best_test_accuracy']}")

    if results:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"Results written to {csv_path}")


if __name__ == "__main__":
    main()
