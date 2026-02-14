"""
从节点坐标与保留掩码构建 MST 有向树（用于导出 SWC）。
"""
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data


def build_mst_graph(raw_data: Data, mask_array: np.ndarray, k: int = 15) -> nx.DiGraph:
    """
    根据 mask_array 保留节点（1 保留，0 丢弃），在保留点的坐标上建 KNN 图再取 MST，
    取含 Root（type=1）的连通分量，再按 BFS 定向为有向树。节点属性含 x, y, z, r。

    Args:
        raw_data: 含 .pos 与 .x（含 radius、type 等）
        mask_array: 与节点数同长的 0/1 数组
        k: KNN 的 k

    Returns:
        NetworkX DiGraph，或节点不足时 None
    """
    coords = raw_data.pos.numpy()
    features = raw_data.x.numpy()
    radii = features[:, 0]
    types = features[:, 1]
    num_nodes = coords.shape[0]
    indices = np.arange(num_nodes)

    is_root = types == 1.0
    keep_mask = (mask_array == 1) | is_root
    survivor_indices = indices[keep_mask]

    if len(survivor_indices) < 2:
        return None

    survivor_coords = coords[survivor_indices]
    survivor_radii = radii[survivor_indices]

    raw_root_idx = np.where(is_root)[0]
    if len(raw_root_idx) > 0 and raw_root_idx[0] in survivor_indices:
        new_root_local_idx = np.where(survivor_indices == raw_root_idx[0])[0][0]
    else:
        new_root_local_idx = 0

    curr_k = min(k, len(survivor_indices) - 1)
    A = kneighbors_graph(
        survivor_coords, n_neighbors=curr_k, mode="distance", include_self=False
    )
    G_knn = nx.from_scipy_sparse_array(A)
    mst = nx.minimum_spanning_tree(G_knn, weight="weight")

    if not nx.is_connected(mst):
        components = list(nx.connected_components(mst))
        for comp in components:
            if new_root_local_idx in comp:
                mst = mst.subgraph(comp).copy()
                break
        final_local_ids = sorted(list(mst.nodes))
        mapping = {old_id: i for i, old_id in enumerate(final_local_ids)}
        survivor_indices = survivor_indices[final_local_ids]
        survivor_coords = survivor_coords[final_local_ids]
        survivor_radii = survivor_radii[final_local_ids]
        new_root_local_idx = mapping[new_root_local_idx]
        mst = nx.relabel_nodes(mst, mapping)

    dfs_tree = nx.bfs_tree(mst, source=new_root_local_idx)
    for node_idx in dfs_tree.nodes:
        dfs_tree.nodes[node_idx]["x"] = survivor_coords[node_idx][0]
        dfs_tree.nodes[node_idx]["y"] = survivor_coords[node_idx][1]
        dfs_tree.nodes[node_idx]["z"] = survivor_coords[node_idx][2]
        dfs_tree.nodes[node_idx]["r"] = survivor_radii[node_idx]

    return dfs_tree
