"""
在 morphology-node 数据集上，用最优 Graph-Mamba (layers_6) 模型做推理，
把若干 test graph 的节点预测体素化到 256 x 256 x 256 网格，并保存为 .npy，
供上层脚本做三视角 MIP 可视化。

用法（在 gnn_project 根目录）:
  conda activate medsam   # 或你跑 Graph-Mamba 的环境
  cd external/Graph-Mamba
  python inference_morph_voxels.py --num_graphs 5
"""

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data


# 保证能 import graphgps 等本仓库模块
GRAPH_MAMBA_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GRAPH_MAMBA_ROOT)
import graphgps  # noqa: F401  # 注册 GraphGPS 的自定义 config / 模块

from torch_geometric.graphgym.config import (cfg, set_cfg, load_cfg)  # noqa: E402
from torch_geometric.graphgym.loader import create_loader  # noqa: E402
from torch_geometric.graphgym.model_builder import create_model  # noqa: E402
from torch_geometric.graphgym.checkpoint import load_ckpt  # noqa: E402
from torch_geometric.graphgym.utils.device import auto_select_device  # noqa: E402


def voxelize_points(
    pos: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 256,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> np.ndarray:
    """
    把带掩码的点云体素化到 (D,H,W) 网格。
    约定：
      - pos: [N,3] (x,y,z)
      - mask: [N]  0/1
      - 输出 vol: [D,H,W]，其中 D 对应 z，H 对应 y，W 对应 x
    """
    if pos.shape[0] == 0:
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    pos = pos.astype(np.float32)
    m = mask.astype(bool)
    if not m.any():
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    pts = pos[m]  # 只看被选中的点
    # 归一化范围固定为整张图的 bounding box，避免 target / pred 使用不同坐标系
    if mins is None or maxs is None:
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
    extent = maxs - mins
    extent[extent == 0] = 1.0  # 避免除零

    norm = (pts - mins) / extent  # 映射到 [0,1]
    idx = np.clip((norm * (grid_size - 1)).astype(int), 0, grid_size - 1)

    # x,y,z -> W,H,D
    x = idx[:, 0]
    y = idx[:, 1]
    z = idx[:, 2]
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    vol[z, y, x] = 1
    return vol


def prepare_cfg(cfg_yaml: str, opts):
    """
    载入基础 YAML（morphology-node-GatedGCN-only.yaml），
    并通过 opts 覆盖超参（例如 gnn.layers_mp=6），用于重建模型和 loader。
    """
    # 模拟 GraphGym main.py 的 args 结构
    class Args:
        pass

    args = Args()
    args.cfg_file = cfg_yaml
    args.opts = opts

    set_cfg(cfg)
    load_cfg(cfg, args)


def get_loaders() -> Tuple[torch.utils.data.DataLoader, ...]:
    loaders = create_loader()
    # GraphGym 通常返回 [train_loader, val_loader, test_loader]
    if isinstance(loaders, (list, tuple)) and len(loaders) >= 3:
        return loaders[0], loaders[1], loaders[2]
    # 兜底：如果返回单个 loader，就都用它
    return loaders, loaders, loaders


def main():
    parser = argparse.ArgumentParser(description="Run inference on morphology-node with best Graph-Mamba model and export 256^3 voxels.")
    parser.add_argument("--num_graphs", type=int, default=5, help="从 test 集中导出多少个图（默认 5）")
    parser.add_argument("--grid_size", type=int, default=256, help="体素网格尺寸（默认 256）")
    args = parser.parse_args()

    # 导入上层 config，拿 DATA_ROOT/GRAPH_MAMBA_OUT_DIR
    PROJECT_ROOT = os.path.dirname(os.path.dirname(GRAPH_MAMBA_ROOT))
    sys.path.insert(0, PROJECT_ROOT)
    import config as gnn_config  # noqa: E402
    from utils.graph_build import build_mst_graph  # noqa: E402
    from data.swc_io import save_graph_to_swc  # noqa: E402

    # 最优实验：layers_6；基于原始 YAML + 覆盖 gnn.layers_mp=6
    run_root = os.path.join(
        gnn_config.GRAPH_MAMBA_OUT_DIR,
        "morphology-node-GatedGCN-only-layers_6",
    )
    cfg_yaml = os.path.join(
        GRAPH_MAMBA_ROOT,
        "configs",
        "Mamba",
        "morphology-node-GatedGCN-only.yaml",
    )
    ckpt_dir = os.path.join(run_root, "42", "ckpt")
    ckpt_path = os.path.join(ckpt_dir, "199.ckpt")

    if not os.path.isfile(cfg_yaml):
        raise SystemExit(f"基础 YAML 未找到: {cfg_yaml}")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"ckpt 未找到: {ckpt_path}")

    vox_root = os.path.join(
        gnn_config.DATA_ROOT,
        "graph_mamba_voxels",
        "layers_6",
    )
    target_dir = os.path.join(vox_root, "target")
    pred_dir = os.path.join(vox_root, "pred")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # SWC 输出根目录：graph_mamba_swc/layers_6
    swc_root = os.path.join(
        gnn_config.DATA_ROOT,
        "graph_mamba_swc",
        "layers_6",
    )
    os.makedirs(swc_root, exist_ok=True)

    # 准备 cfg / 设备 / 数据 / 模型
    opts = [
        "out_dir",
        run_root,
        "dataset.dir",
        gnn_config.TRAIN_DATA_DIR,
        "name_tag",
        "layers_6",
        "gnn.layers_mp",
        "6",
    ]
    prepare_cfg(cfg_yaml, opts)
    auto_select_device()
    # 新版 GraphGym 可能没有 cfg.device，这里直接根据 CUDA 可用性与 accelerator 推断
    use_cuda = torch.cuda.is_available() and getattr(cfg, "accelerator", "cpu") != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader = get_loaders()

    model = create_model().to(device)
    # 设置 run_dir 以便 GraphGym 在正确目录下查找 ckpt
    cfg.run_dir = os.path.join(run_root, "42")
    load_ckpt(model, optimizer=None, scheduler=None, epoch=199)
    model.eval()

    saved = 0
    for batch in test_loader:
        if saved >= args.num_graphs:
            break
        batch = batch.to(device)
        with torch.no_grad():
            pred, true = model(batch)  # inductive_node head: 返回 (pred, label)

        # pred: [N, 1] logits, true: [N]
        probs = torch.sigmoid(pred.view(-1))
        pred_mask = (probs > 0.5).cpu().numpy().astype(np.uint8)
        gt_mask = true.view(-1).cpu().numpy().astype(np.uint8)
        pos = batch.pos.cpu().numpy()
        g_idx = batch.batch.cpu().numpy()  # 每个节点属于哪个 graph

        # 拆成单 graph 级别的数据列表，便于构建 SWC
        data_list = batch.to_data_list()

        num_graph_in_batch = int(g_idx.max()) + 1
        for gid in range(num_graph_in_batch):
            if saved >= args.num_graphs:
                break
            sel = (g_idx == gid)
            if not sel.any():
                continue
            pos_g = pos[sel]
            pred_g = pred_mask[sel]
            gt_g = gt_mask[sel]
            # 统一使用该 graph 的整体 bounding box，避免 target / pred 在体素坐标系上产生整体偏移
            mins = pos_g.min(axis=0)
            maxs = pos_g.max(axis=0)

            vol_pred = voxelize_points(pos_g, pred_g, grid_size=args.grid_size, mins=mins, maxs=maxs)
            vol_gt = voxelize_points(pos_g, gt_g, grid_size=args.grid_size, mins=mins, maxs=maxs)

            # ==== SWC: 基于 Graph-Mamba 的 Data 构造 MST 并导出 ====
            pyg_data = data_list[gid].cpu()
            x_np = pyg_data.x.numpy()
            pos_np = pyg_data.pos.numpy()
            if x_np.size == 0 or pos_np.size == 0:
                # 空图，跳过 SWC 构建，但仍可保存体素
                mst_data = None
            else:
                # 半径用第一列（标准化后的半径），类型从 one-hot (cols 1..3) 还原
                radius = x_np[:, 0]
                type_one_hot = x_np[:, 1:4] if x_np.shape[1] >= 4 else np.zeros((x_np.shape[0], 3), dtype=np.float32)
                if type_one_hot.size == 0:
                    node_type = np.zeros(x_np.shape[0], dtype=np.float32)
                else:
                    node_type = type_one_hot.argmax(axis=1).astype(np.float32)  # 0,1,2
                x_mst = np.stack([radius, node_type], axis=1)
                mst_data = Data(
                    x=torch.from_numpy(x_mst).float(),
                    pos=torch.from_numpy(pos_np).float(),
                )

            base = f"graph_{saved:03d}"
            # input / pred / gt 三种掩码，对应 SWC
            if mst_data is not None:
                input_mask = np.ones(mst_data.num_nodes, dtype=np.uint8)
                for suffix, mask_arr, desc in [
                    ("input", input_mask, "Graph-Mamba Input (All Points)"),
                    ("pred", pred_g, "Graph-Mamba Prediction"),
                    ("gt", gt_g, "Ground Truth Label"),
                ]:
                    # mask_arr 是该 graph 的节点级掩码，长度与 mst_data.num_nodes 一致
                    full_mask = mask_arr.astype(np.uint8)
                    tree = build_mst_graph(mst_data, full_mask, k=gnn_config.MST_K_NEIGHBORS)
                    if tree is None:
                        continue
                    swc_path = os.path.join(swc_root, f"{base}_{suffix}.swc")
                    save_graph_to_swc(tree, swc_path, description=desc)

            # ==== 保存体素 ====
            np.save(os.path.join(pred_dir, f"{base}.npy"), vol_pred)
            np.save(os.path.join(target_dir, f"{base}.npy"), vol_gt)
            print(f"[{saved+1}/{args.num_graphs}] saved voxel + SWC: {base}")
            saved += 1

    print(f"Done. Saved {saved} graphs to {vox_root} and {swc_root}")


if __name__ == "__main__":
    main()

