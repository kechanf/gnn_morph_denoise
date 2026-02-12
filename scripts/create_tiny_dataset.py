"""
从完整合成数据目录中复制前 N 个 .pt 样本到小数据集目录，用于快速验证训练流程。

用法（在项目根目录）:
  python scripts/create_tiny_dataset.py           # 默认 50 个样本
  python scripts/create_tiny_dataset.py --n 100   # 指定样本数
"""
import os
import sys
import argparse
import shutil
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Create tiny dataset for training sanity check")
    parser.add_argument("--n", type=int, default=50, help="Number of .pt samples to copy (default: 50)")
    parser.add_argument("--source", type=str, default=None, help="Source dir (default: config.TRAIN_DATA_DIR)")
    parser.add_argument("--target", type=str, default=None, help="Target dir (default: config.TRAIN_DATA_DIR_TINY_50)")
    args = parser.parse_args()

    import config

    source_dir = os.path.abspath(args.source or config.TRAIN_DATA_DIR)
    target_dir = os.path.abspath(args.target or config.TRAIN_DATA_DIR_TINY_50)

    if not os.path.isdir(source_dir):
        print(f"Error: source is not a directory: {source_dir}")
        return 1

    pt_files = sorted(glob.glob(os.path.join(source_dir, "*.pt")))
    if len(pt_files) == 0:
        print(f"Error: no .pt files in {source_dir}")
        return 1

    take = min(args.n, len(pt_files))
    os.makedirs(target_dir, exist_ok=True)

    for i, src in enumerate(pt_files[:take]):
        name = os.path.basename(src)
        dst = os.path.join(target_dir, name)
        shutil.copy2(src, dst)

    print(f"Created tiny dataset: {take} samples in {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
