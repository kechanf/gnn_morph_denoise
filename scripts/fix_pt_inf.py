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


# 兼容旧版本保存的 DataEdgeAttr / DataTensorAttr（曾在 torch_geometric.data.data 中定义），
# 否则在 PyTorch>=2.6 + 新版 PyG 下，torch.load 旧 .pt 时会因为类缺失而报错，
# 出现类似 "Can't get attribute 'DataEdgeAttr' on torch_geometric.data.data"。
def _register_legacy_pyg_attr_stubs():
    try:
        from torch_geometric.data import Data
        import torch_geometric.data.data as _pyg_data_mod

        if not hasattr(_pyg_data_mod, "DataEdgeAttr"):
            class DataEdgeAttr(Data):  # type: ignore[misc]
                """Backward-compatible stub for legacy pickles."""
                pass

            _pyg_data_mod.DataEdgeAttr = DataEdgeAttr  # type: ignore[attr-defined]

        if not hasattr(_pyg_data_mod, "DataTensorAttr"):
            class DataTensorAttr(Data):  # type: ignore[misc]
                """Backward-compatible stub for legacy pickles."""
                pass

            _pyg_data_mod.DataTensorAttr = DataTensorAttr  # type: ignore[attr-defined]
    except Exception:
        # 若导入失败，不影响正常路径；仅在加载旧 pickle 时才需要。
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="只统计不写回")
    args = parser.parse_args()

    # 在开始加载任何 .pt 之前先注册兼容 stub，避免 torch.load 旧文件时报错。
    _register_legacy_pyg_attr_stubs()

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
