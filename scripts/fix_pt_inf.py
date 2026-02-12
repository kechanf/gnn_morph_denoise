"""
修复已有 .pt 中 x 的 Inf/NaN：将 dist(第3列)、angle(第4列) 的非有限值改为 -1 和 0，避免训练 NaN。

用法: python scripts/fix_pt_inf.py [--data_dir PATH] [--dry_run]

注意：若用 Graph-Mamba 训练，修复 .pt 后需删除该数据目录下的 processed/ 文件夹，
否则会继续使用旧缓存（仍含 Inf），导致 loss=nan。例如：
  rm -rf /path/to/synthesis_data/processed
"""
import os
import sys
import argparse
import glob
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="只统计不写回")
    args = parser.parse_args()
    import config
    data_dir = os.path.abspath(args.data_dir or config.TRAIN_DATA_DIR)
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not pt_files:
        print(f"未找到 .pt: {data_dir}")
        return 1
    fixed = 0
    for path in pt_files:
        data = torch.load(path, weights_only=False)
        if not hasattr(data, "x") or data.x is None or data.x.size(1) < 4:
            continue
        x = data.x
        bad_dist = ~torch.isfinite(x[:, 2])
        bad_angle = ~torch.isfinite(x[:, 3])
        if bad_dist.any() or bad_angle.any():
            x = x.clone()
            x[bad_dist, 2] = -1.0
            x[bad_angle, 3] = 0.0
            data.x = x
            if not args.dry_run:
                torch.save(data, path)
            fixed += 1
    print(f"处理目录: {data_dir}, 共 {len(pt_files)} 个文件, 修复 {fixed} 个")
    if args.dry_run:
        print("(dry_run 未写回)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
