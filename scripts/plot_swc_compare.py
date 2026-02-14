import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_swc_edges(path: str) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    简单读取 SWC，返回所有 (parent -> child) 边的 3D 坐标对。
    """
    xs, ys, zs, parents = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            swc_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            parent = int(parts[6])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            parents.append(parent)

    coords = list(zip(xs, ys, zs))
    edges = []
    for idx, parent in enumerate(parents):
        if parent <= 0:
            continue
        child_xyz = coords[idx]
        parent_xyz = coords[parent - 1]  # SWC 下标从 1 开始
        edges.append((parent_xyz, child_xyz))
    return edges


def plot_projection(ax, edges, view: str, color: str, label: str):
    """
    在指定子图上按照给定视角画线段。
    view: "xy" / "xz" / "yz"
    """
    for (x1, y1, z1), (x2, y2, z2) in edges:
        if view == "xy":
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.8)
        elif view == "xz":
            ax.plot([x1, x2], [z1, z2], color=color, linewidth=0.8)
        elif view == "yz":
            ax.plot([y1, y2], [z1, z2], color=color, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(view.upper())


def main():
    parser = argparse.ArgumentParser(
        description="对比两个 SWC（GT vs Pred），在 XY / XZ / YZ 三视图上 overlay。"
    )
    parser.add_argument("--gt", type=str, required=True, help="GT SWC 路径，例如 graph_000_gt.swc")
    parser.add_argument("--pred", type=str, required=True, help="Pred SWC 路径，例如 graph_000_pred.swc")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 PNG 路径；不指定则保存在 pred 同目录下自动命名。",
    )
    args = parser.parse_args()

    gt_edges = load_swc_edges(args.gt)
    pred_edges = load_swc_edges(args.pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = ["xy", "xz", "yz"]

    for ax, view in zip(axes, views):
        plot_projection(ax, gt_edges, view=view, color="green", label="GT")
        plot_projection(ax, pred_edges, view=view, color="magenta", label="Pred")

    # 统一图例放在中间
    handles = [
        plt.Line2D([0], [0], color="green", lw=2, label="GT (Target)"),
        plt.Line2D([0], [0], color="magenta", lw=2, label="Pred (Graph-Mamba)"),
    ]
    fig.legend(handles=handles, loc="upper right")

    fig.tight_layout()

    if args.out is None:
        base_dir = os.path.dirname(os.path.abspath(args.pred))
        base_name = os.path.splitext(os.path.basename(args.pred))[0]
        out_path = os.path.join(base_dir, f"{base_name}_swc_compare.png")
    else:
        out_path = args.out

    fig.savefig(out_path, dpi=200)
    print(f"Saved SWC compare figure to: {out_path}")


if __name__ == "__main__":
    main()

