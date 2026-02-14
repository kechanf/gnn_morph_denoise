#!/usr/bin/env python
"""
批量随机选择若干个样本，在三个 MIP 视角下对比 target / pred，并输出 PNG。

假设：
  - target_dir / pred_dir 下都有若干 .npy 文件，文件名一一对应
  - 每个 .npy 为 3D 体数据 (D,H,W)

用法示例：
  conda activate hb_seg
  python scripts/batch_mip_compare.py \
    --target_dir /path/to/target_npys \
    --pred_dir   /path/to/pred_npys   \
    --out_dir    /path/to/mip_pngs    \
    --num 5
"""

import argparse
import os
import random
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num", type=int, default=5, help="随机选择多少个样本做可视化")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    pred_dir = os.path.abspath(args.pred_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(target_dir):
        raise SystemExit(f"target_dir 不存在: {target_dir}")
    if not os.path.isdir(pred_dir):
        raise SystemExit(f"pred_dir 不存在: {pred_dir}")
    os.makedirs(out_dir, exist_ok=True)

    target_files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]
    if not target_files:
        raise SystemExit(f"target_dir 中未找到 .npy 文件: {target_dir}")

    # 只保留 pred 也存在的样本
    pairs = []
    for fn in target_files:
        t_path = os.path.join(target_dir, fn)
        p_path = os.path.join(pred_dir, fn)
        if os.path.isfile(p_path):
            pairs.append((t_path, p_path))

    if not pairs:
        raise SystemExit("没有找到 target/pred 同名 .npy 对。")

    random.shuffle(pairs)
    pairs = pairs[: args.num]

    print(f"将随机可视化 {len(pairs)} 个样本，输出到: {out_dir}")

    for t_path, p_path in pairs:
        base = os.path.splitext(os.path.basename(t_path))[0]
        out_png = os.path.join(out_dir, f"{base}_mip.png")
        cmd = [
            "python",
            os.path.join(os.path.dirname(__file__), "mip_compare.py"),
            "--target",
            t_path,
            "--pred",
            p_path,
            "--out",
            out_png,
        ]
        print("  ->", os.path.basename(t_path), "=>", out_png)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

