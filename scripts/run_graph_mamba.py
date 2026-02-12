"""
从 gnn_project 根目录运行 Graph-Mamba baseline（形态学节点分类）。

会使用 config.TRAIN_DATA_DIR 作为数据目录，并调用 external/Graph-Mamba 的 main.py。
需先安装 Graph-Mamba 依赖（见下方说明或 docs/GRAPH_MAMBA_DEPLOY.md）。

用法:
  cd /path/to/gnn_project
  conda activate graph_mamba   # 或已安装依赖的环境
  python scripts/run_graph_mamba.py

  或指定数据目录:
  python scripts/run_graph_mamba.py --data_dir /path/to/synthesis_data
"""
import sys
import os
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_MAMBA_ROOT = os.path.join(PROJECT_ROOT, "external", "Graph-Mamba")
CONFIG_MAMBA = "configs/Mamba/morphology-node-EX.yaml"  # 需 mamba-ssm
CONFIG_GATEDGCN = "configs/Mamba/morphology-node-GatedGCN-only.yaml"  # 无需 mamba-ssm
CONFIG_REL = CONFIG_MAMBA

_DEPS_MSG = """
未检测到 PyTorch。请先安装依赖再运行本脚本。

方式一（推荐，使用 Conda）:
  conda create -n graph_mamba python=3.9 -y
  conda activate graph_mamba
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements_graph_mamba.txt
  # Mamba-SSM: pip install mamba-ssm  （见 https://github.com/state-spaces/mamba）

方式二（使用 Graph-Mamba 官方）:
  cd external/Graph-Mamba && conda create --name graph-mamba --file requirements_conda.txt

然后激活环境并重新运行:
  conda activate graph_mamba
  python scripts/run_graph_mamba.py
"""


def main():
    parser = argparse.ArgumentParser(description="Run Graph-Mamba on morphology node data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to folder of .pt graph files (default: from config.TRAIN_DATA_DIR)",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="False",
        choices=("True", "False"),
        help="Use wandb (default: False)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of runs with different seeds (default: 1)",
    )
    parser.add_argument(
        "--no-mamba",
        action="store_true",
        help="Use GatedGCN-only config (no mamba-ssm), for env without Mamba",
    )
    parser.add_argument(
        "--name_tag",
        type=str,
        default=None,
        help="Experiment name (subdir under out_dir); for hyperparameter runs",
    )
    parser.add_argument(
        "--override",
        nargs=2,
        action="append",
        default=None,
        metavar=("KEY", "VALUE"),
        help="Override cfg key (e.g. gnn.dropout) and value; can be repeated",
    )
    args = parser.parse_args()

    config_rel = CONFIG_GATEDGCN if args.no_mamba else CONFIG_MAMBA

    sys.path.insert(0, PROJECT_ROOT)
    import config as gnn_config
    out_dir = os.path.abspath(gnn_config.GRAPH_MAMBA_OUT_DIR)
    if args.data_dir is None:
        data_dir = os.path.abspath(gnn_config.TRAIN_DATA_DIR)
    else:
        data_dir = os.path.abspath(args.data_dir)

    if not os.path.isdir(data_dir):
        print(f"Error: data_dir is not a directory: {data_dir}")
        sys.exit(1)

    cfg_path = os.path.join(GRAPH_MAMBA_ROOT, config_rel)
    if not os.path.isfile(cfg_path):
        print(f"Error: config not found: {cfg_path}")
        sys.exit(1)

    # 使用子进程前检查 Graph-Mamba 入口能否 import torch（避免子进程报错不直观）
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print(_DEPS_MSG)
        sys.exit(1)

    cmd = [
        sys.executable,
        os.path.join(GRAPH_MAMBA_ROOT, "main.py"),
        "--cfg", cfg_path,
        "--repeat", str(args.repeat),
        "out_dir", out_dir,
        "dataset.dir", data_dir,
        "wandb.use", args.wandb,
    ]
    if getattr(args, "name_tag", None):
        cmd.extend(["name_tag", args.name_tag])
    if getattr(args, "override", None):
        for k, v in args.override:
            cmd.extend([k, str(v)])
    env = os.environ.copy()
    env["PYTHONPATH"] = GRAPH_MAMBA_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    rc = subprocess.run(cmd, cwd=GRAPH_MAMBA_ROOT, env=env)
    sys.exit(rc.returncode)


if __name__ == "__main__":
    main()
