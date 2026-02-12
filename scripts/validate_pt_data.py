"""
抽样检查合成 .pt 数据：NaN/Inf、格式、标签分布、边与特征范围。
用于判断训练中出现的 NaN 是否与原始数据有关。

用法:
  python scripts/validate_pt_data.py [--data_dir PATH] [--max_files N]
"""
import os
import sys
import argparse
import glob
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_one(data, path: str) -> list[str]:
    """检查单个 Data，返回问题列表。"""
    issues = []
    # 必需字段
    for key in ("x", "y", "edge_index", "pos"):
        if not hasattr(data, key):
            issues.append(f"缺少字段: {key}")
            continue
        t = getattr(data, key)
        if t is None:
            issues.append(f"{key} 为 None")
        elif torch.is_tensor(t):
            if torch.isnan(t).any() or torch.isinf(t).any():
                issues.append(f"{key} 含 NaN/Inf")
            if key == "x" and t.dim() != 2:
                issues.append(f"x 应为 2 维 [N,4]，当前 shape={t.shape}")
            if key == "x" and t.size(1) != 4:
                issues.append(f"x 应为 4 列 [r, type, dist, angle]，当前 cols={t.size(1)}")
            if key == "y":
                if t.dim() > 1:
                    t = t.squeeze()
                unique = t.unique().tolist()
                if set(unique) - {0.0, 1.0}:
                    issues.append(f"y 应仅为 0/1，当前 unique={unique}")
            if key == "edge_index" and t.size(0) != 2:
                issues.append(f"edge_index 应为 [2,E]，当前 shape={t.shape}")
    if not issues and hasattr(data, "edge_index") and data.edge_index is not None:
        ei = data.edge_index
        n = data.x.size(0) if hasattr(data, "x") and data.x is not None else 0
        if n > 0 and (ei.max() >= n or ei.min() < 0):
            issues.append(f"edge_index 越界: 节点数={n}, index 范围=[{ei.min().item()}, {ei.max().item()}]")
    # 特征范围（仅提示）
    if hasattr(data, "x") and data.x is not None and not torch.isnan(data.x).any():
        x = data.x
        for i, name in enumerate(["r", "type", "dist", "angle"]):
            col = x[:, i]
            if col.numel() > 0:
                if torch.isinf(col).any():
                    issues.append(f"x 第{i}列({name})含 Inf")
                # 全为同一值会导致标准化后 std=0，可能引发数值问题
                if col.numel() > 1 and col.std().item() < 1e-8:
                    issues.append(f"x 第{i}列({name})几乎为常数 std={col.std().item():.2e}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate .pt graph files")
    parser.add_argument("--data_dir", type=str, default=None, help="目录 (默认 config.TRAIN_DATA_DIR)")
    parser.add_argument("--max_files", type=int, default=200, help="最多检查多少个 .pt")
    args = parser.parse_args()

    import config
    data_dir = os.path.abspath(args.data_dir or config.TRAIN_DATA_DIR)
    if not os.path.isdir(data_dir):
        print(f"错误: 目录不存在 {data_dir}")
        return 1

    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))[: args.max_files]
    if not pt_files:
        print(f"未找到 .pt 文件: {data_dir}")
        return 1

    print(f"检查目录: {data_dir}")
    print(f"抽样数量: {len(pt_files)}")
    print()

    bad = []
    stats = {"nan_inf": 0, "shape": 0, "label": 0, "edge": 0, "constant": 0}
    label_counts = {0: 0, 1: 0}
    total_nodes = 0

    # .pt 存的是 PyG Data，需 weights_only=False 才能反序列化
    for path in pt_files:
        try:
            data = torch.load(path, weights_only=False)
        except Exception as e:
            bad.append((os.path.basename(path), [f"加载失败: {e}"]))
            continue
        issues = check_one(data, path)
        for i in issues:
            if "NaN/Inf" in i or "含 NaN" in i or "含 Inf" in i:
                stats["nan_inf"] += 1
            elif "shape" in i or "列" in i:
                stats["shape"] += 1
            elif "y " in i or "unique" in i:
                stats["label"] += 1
            elif "edge" in i or "越界" in i:
                stats["edge"] += 1
            elif "常数" in i:
                stats["constant"] += 1
        if issues:
            bad.append((os.path.basename(path), issues))
        # 统计标签
        if hasattr(data, "y") and data.y is not None:
            y = data.y.squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)
            for v in data.y.squeeze().tolist():
                v = int(v)
                label_counts[v] = label_counts.get(v, 0) + 1
        if hasattr(data, "x") and data.x is not None:
            total_nodes += data.x.size(0)

    # 汇总
    print("===== 检查结果 =====")
    if bad:
        print(f"存在问题文件数: {len(bad)} / {len(pt_files)}")
        for name, issues in bad[:15]:
            print(f"  {name}: {'; '.join(issues)}")
        if len(bad) > 15:
            print(f"  ... 共 {len(bad)} 个文件")
        print()
        print("问题类型统计:", stats)
    else:
        print("未发现明显数据错误（NaN/Inf、格式、标签、边索引）。")
    print()
    print("标签分布 (节点级):", dict(label_counts))
    print(f"总节点数: {total_nodes}, 平均每图: {total_nodes / len(pt_files):.1f}")

    # 额外：抽查原始 x 的 dist/angle 是否有异常值
    print()
    print("===== 原始特征范围抽查 (前 5 个文件) =====")
    for path in pt_files[:5]:
        data = torch.load(path, weights_only=False)
        if not hasattr(data, "x") or data.x is None:
            continue
        x = data.x
        if x.size(1) >= 4:
            r, typ, dist, angle = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            print(f"  {os.path.basename(path)}: r=[{r.min():.2f},{r.max():.2f}] "
                  f"dist=[{dist.min():.2f},{dist.max():.2f}] angle=[{angle.min():.2f},{angle.max():.2f}] "
                  f"nan_r={torch.isnan(r).any().item()} nan_dist={torch.isnan(dist).any().item()}")

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
