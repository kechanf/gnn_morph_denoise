#!/usr/bin/env python
"""
在三个 MIP 视角下对比 target 与 pred，并输出 PNG。

用法示例：
  conda activate hb_seg
  python scripts/mip_compare.py \
    --target /path/to/target.npy \
    --pred /path/to/pred.npy \
    --out /path/to/out_mip.png

假设 target / pred 都是形状为 (D, H, W) 的 3D 体数据，
值可以是 0/1 掩码或概率图。
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_volume(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} 不是 3D 体数据，shape={arr.shape}")
    return arr


def make_mips(vol: np.ndarray):
    """计算三个方向的最大强度投影 (MIP)。"""
    # vol: (D, H, W)
    mip_xy = vol.max(axis=0)  # H x W  (沿 Z 投影)
    mip_xz = vol.max(axis=1)  # D x W  (沿 Y 投影)
    mip_yz = vol.max(axis=2)  # D x H  (沿 X 投影)
    return mip_xy, mip_xz, mip_yz


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    return img


def overlay_mip(mip_t: np.ndarray, mip_p: np.ndarray) -> np.ndarray:
    """
    把 target / pred 的 MIP 叠成伪彩色：
      - target -> 绿色通道
      - pred   -> 品红通道（红+蓝）
    """
    t = normalize(mip_t)
    p = normalize(mip_p)
    rgb = np.zeros((*t.shape, 3), dtype=np.float32)
    # 绿色通道: target
    rgb[..., 1] = t
    # 红+蓝通道: pred (品红)
    rgb[..., 0] = p
    rgb[..., 2] = p
    return np.clip(rgb, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="target 体数据 .npy, 形状 (D,H,W)")
    parser.add_argument("--pred", required=True, help="pred 体数据 .npy, 形状 (D,H,W)")
    parser.add_argument("--out", required=True, help="输出对比图 PNG 路径")
    args = parser.parse_args()

    vol_t = load_volume(args.target)
    vol_p = load_volume(args.pred)

    if vol_t.shape != vol_p.shape:
        raise ValueError(f"shape 不一致: target={vol_t.shape}, pred={vol_p.shape}")

    # 三轴 MIP
    t_xy, t_xz, t_yz = make_mips(vol_t)
    p_xy, p_xz, p_yz = make_mips(vol_p)

    # 三幅叠加图
    im_xy = overlay_mip(t_xy, p_xy)
    im_xz = overlay_mip(t_xz, p_xz)
    im_yz = overlay_mip(t_yz, p_yz)

    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(im_xy)
    ax1.set_title("MIP XY (Z-proj)")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(im_xz)
    ax2.set_title("MIP XZ (Y-proj)")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(im_yz)
    ax3.set_title("MIP YZ (X-proj)")
    ax3.axis("off")

    # 在第一个子图上添加图例：target / pred 颜色含义
    legend_handles = [
        Patch(color="lime", label="Target (GT)"),
        Patch(color="magenta", label="Pred (Model)"),
    ]
    ax1.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"保存对比图到: {args.out}")


if __name__ == "__main__":
    main()

