"""
Graph-Mamba 性能测试脚本：
- 自动从 graph_mamba_experiments.csv 中找到验证精度最高的实验（当前是 layers_6）
- 读取对应结果目录下的 agg/val/test/best.json，打印最优指标

用法（在项目根目录）:
  conda activate hb_seg  # 或含 torch / json 的环境
  python scripts/test_graph_mamba.py
"""

import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_csv_rows(csv_path: str):
    rows = []
    if not os.path.isfile(csv_path):
        print(f"CSV 不存在: {csv_path}")
        return rows
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            if not line.strip():
                continue
            cols = line.strip().split(",")
            row = dict(zip(header, cols))
            rows.append(row)
    return rows


def find_best_by_val_accuracy(rows):
    best = None
    best_val = -1.0
    for r in rows:
        try:
            val = float(r.get("best_val_accuracy", "") or 0.0)
        except ValueError:
            val = 0.0
        if val > best_val:
            best_val = val
            best = r
    return best


def main():
    import config  # noqa: F401

    csv_path = os.path.join(config.DATA_ROOT, "graph_mamba_experiments.csv")
    rows = load_csv_rows(csv_path)
    if not rows:
        print("没有找到 Graph-Mamba 实验结果 CSV，先运行 run_graph_mamba_experiments.py。")
        return

    best = find_best_by_val_accuracy(rows)
    if not best:
        print("无法根据 best_val_accuracy 找到最优实验。")
        return

    name_tag = best["name_tag"]
    print(f"验证集精度最高的实验: name_tag={name_tag}, best_val_accuracy={best['best_val_accuracy']}")
    print(f"overrides={best['overrides']}")

    # 结果目录推断：GRAPH_MAMBA_OUT_DIR / (config_basename + '-' + name_tag)
    config_basename = "morphology-node-GatedGCN-only"
    run_name = f"{config_basename}-{name_tag}"
    base_out = config.GRAPH_MAMBA_OUT_DIR
    run_dir = os.path.join(base_out, run_name)

    val_best_path = os.path.join(run_dir, "agg", "val", "best.json")
    test_best_path = os.path.join(run_dir, "agg", "test", "best.json")

    def _safe_load(path):
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
        return None

    val_best = _safe_load(val_best_path)
    test_best = _safe_load(test_best_path)

    print("\n=== 最优 val 指标 (agg/val/best.json) ===")
    if val_best:
        for k, v in val_best.items():
            print(f"{k}: {v}")
    else:
        print("val/best.json 不存在，可能未成功聚合。")

    print("\n=== 最优 test 指标 (agg/test/best.json) ===")
    if test_best:
        for k, v in test_best.items():
            print(f"{k}: {v}")
    else:
        print("test/best.json 不存在（当前 agg_runs 只聚合了 train/val）。")

    print(f"\n完整结果目录: {run_dir}")


if __name__ == "__main__":
    main()

