"""
读取 Graph-Mamba 超参实验的 CSV，生成可视化图表，方便你查看哪些超参效果最好。

使用方法（在项目根目录）:
  conda activate medsam   # 或含 matplotlib 的环境
  python scripts/plot_graph_mamba_experiments.py

会生成:
  - /data2/kfchen/tracing_ws/morphology_seg/graph_mamba_experiments.png
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_results(csv_path: str):
    rows = []
    if not os.path.isfile(csv_path):
        print(f"CSV 不存在: {csv_path}")
        return rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转成 float，处理缺失值
            for k in ["best_val_accuracy", "best_test_accuracy", "best_val_auc", "best_test_auc"]:
                v = row.get(k, "")
                try:
                    row[k] = float(v) if v not in ("", "None", None) else None
                except ValueError:
                    row[k] = None
            rows.append(row)
    return rows


def plot_by_name_tag(rows, out_png: str):
    """
    画一个简单的柱状图：
    x 轴：name_tag
    y 轴：best_val_accuracy（若无则用 best_test_accuracy）
    """
    if not rows:
        print("没有实验结果可视化。")
        return

    # 以 name_tag 为顺序
    x_labels = [r["name_tag"] for r in rows]
    y_vals = []
    for r in rows:
        y = r.get("best_val_accuracy") or r.get("best_test_accuracy")
        y_vals.append(y if y is not None else 0.0)

    plt.figure(figsize=(max(8, len(x_labels) * 0.5), 5))
    bars = plt.bar(range(len(x_labels)), y_vals, color="steelblue")
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Graph-Mamba 超参实验 (val/test accuracy)")
    plt.tight_layout()

    # 在柱子上标数值
    for b, v in zip(bars, y_vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"图表已保存到: {out_png}")


def main():
    import config  # noqa: F401

    csv_path = os.path.join(config.DATA_ROOT, "graph_mamba_experiments.csv")
    out_png = os.path.join(config.DATA_ROOT, "graph_mamba_experiments.png")

    print(f"读取实验结果: {csv_path}")
    rows = load_results(csv_path)

    # 按时间排序（可选），确保顺序与运行顺序一致
    rows.sort(key=lambda r: r.get("timestamp", ""))

    plot_by_name_tag(rows, out_png)


if __name__ == "__main__":
    main()

