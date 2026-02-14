from neuroutils.swc.io import load_swc, swc_to_graph, save_graph_as_swc, load_swc_as_graph
from neuroutils.swc.transform import translate_tree, apply_random_rotation
from neuroutils.swc.geodesic import compute_geodesic_distance, compute_branch_angles
import networkx as nx
import os
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from collections import deque
import torch
from torch_geometric.data import Data

def get_random_rotation_matrix():
    """
    生成一个随机的 3D 旋转矩阵 (3x3)。
    使用 QR 分解法保证在 SO(3) 空间上的均匀分布。
    """
    H = np.random.randn(3, 3)
    Q, R = np.linalg.qr(H)

    # 确保行列式为 +1 (纯旋转，不包含镜像翻转)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def attach_interfer(target_G, target_root, inter_G, inter_root):
    # ---- 1. 随机选目标节点 b ----
    b = random.choice(list(target_G.nodes))
    bx, by, bz = (target_G.nodes[b]['x'],
                  target_G.nodes[b]['y'],
                  target_G.nodes[b]['z'])

    # ---- 2. 随机选干扰节点 a ----
    a = random.choice(list(inter_G.nodes))

    # ==== [新增逻辑] 半径融合 ====
    # 在进行任何几何变换或删除操作前，先更新 b 的半径
    # 规则：r_b = max(r_b, r_a)
    radius_a = inter_G.nodes[a]['r']
    radius_b = target_G.nodes[b]['r']
    target_G.nodes[b]['r'] = max(radius_a, radius_b)
    # ===========================

    # ---- 3. 随机旋转 (绕点 a) ----
    apply_random_rotation(inter_G, center_node_id=a, rot_matrix=get_random_rotation_matrix())

    # 重新获取 a 的坐标 (旋转后理论不变，但为了严谨)
    ax, ay, az = (inter_G.nodes[a]['x'],
                  inter_G.nodes[a]['y'],
                  inter_G.nodes[a]['z'])

    # ---- 4. 平移干扰树，使得 a 对齐到 b ----
    shift = np.array([bx - ax, by - ay, bz - az])
    translate_tree(inter_G, shift)

    # ---- 5. 将 a 的邻居接到 b ----
    neighbors = list(inter_G.neighbors(a))

    # a 被删除，但结构继续
    inter_G.remove_node(a)

    # ---- 6. 将干扰树的节点全部加进 target ----
    # 为防止 id 冲突，将干扰树节点重新编号
    max_id = max(target_G.nodes)
    id_map = {}
    for old in inter_G.nodes:
        max_id += 1
        id_map[old] = max_id
        # 寻找新的inter_root

    # 加节点
    for old, new in id_map.items():
        target_G.add_node(new, **inter_G.nodes[old])

    # 加边（除了已删掉 a）
    for (u, v) in inter_G.edges:
        target_G.add_edge(id_map[u], id_map[v])

    # ---- 7. 补上 a 的邻居连接到 b ----
    for old_nb in neighbors:
        new_nb = id_map[old_nb]
        target_G.add_edge(b, new_nb)

    return target_G, id_map[inter_root]


def generate_interfer_tree(target_path, interfer_paths):
    # 读 target SWC
    df = load_swc(target_path)
    target_G = swc_to_graph(df)
    origin_target_point_n = list(target_G.nodes)

    # target root
    root = int(df.loc[df['parent'] == -1, 'n'].values[0])
    inter_roots = []

    # 合并所有干扰树
    for ipath in interfer_paths:
        inter_df = load_swc(ipath)
        inter_G = swc_to_graph(inter_df)
        inter_root = int(inter_df.loc[inter_df['parent'] == -1, 'n'].values[0])

        target_G, new_inter_root = attach_interfer(target_G, root, inter_G, inter_root)
        inter_roots.append(new_inter_root)

    # 最后从 root 出发重建合法 swc
    return target_G, root, inter_roots, origin_target_point_n


def preprocess_merge_nodes(target_G, origin_node_ids, target_root_id, inter_root_ids, dist_threshold=1.0):
    """
    预处理：合并距离小于阈值的节点。

    优先级策略:
    1. Origin Node
    2. Interfer Root Node
    3. Smaller ID

    返回:
    - new_G
    - new_origin_ids
    - new_target_root_id
    - new_inter_root_ids
    """

    # --- 1. 准备数据 ---
    nodes = list(target_G.nodes)
    # 提取坐标矩阵 [N, 3]
    coords = np.array([[target_G.nodes[n]['x'], target_G.nodes[n]['y'], target_G.nodes[n]['z']] for n in nodes])
    id_to_idx = {n: i for i, n in enumerate(nodes)}

    origin_set = set(origin_node_ids)
    inter_root_set = set(inter_root_ids)

    # --- 2. 寻找聚类 (KDTree) ---
    tree = KDTree(coords)
    # query_pairs 找到所有距离 < threshold 的点对
    pairs = tree.query_pairs(r=dist_threshold)

    # 构建一个临时的图来找连通分量（即哪些点连在一起要合并）
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(nodes)
    cluster_graph.add_edges_from([(nodes[i], nodes[j]) for i, j in pairs])

    # 获取所有聚类 (Connected Components)
    components = list(nx.connected_components(cluster_graph))

    # 如果没有需要合并的点，直接返回
    # (判断标准: 组件数量 == 节点数量，说明每个点都是独立的)
    if len(components) == len(nodes):
        return target_G, origin_node_ids, target_root_id, inter_root_ids

    # --- 3. 执行合并 ---
    # 我们不直接修改 target_G，而是通过 mapping 构建新图或重连边
    # 但由于 networkx 的 contracted_nodes 比较慢且操作繁琐，
    # 我们采用 "映射表 + 重建" 的方式

    merge_map = {} # old_id -> survivor_id

    for comp in components:
        comp = list(comp)
        if len(comp) == 1:
            merge_map[comp[0]] = comp[0]
            continue

        # === 核心逻辑：选出 Survivor ===
        def sort_key(node_id):
            # Python 的 sort 是升序，我们想要优先级高的排在最后(或者reverse=True)
            # 这里的逻辑是：返回一个tuple，依次比较

            # 1. 是否在 origin_set (True=1, False=0)
            is_origin = 1 if node_id in origin_set else 0

            # 2. 是否在 inter_root_set
            is_inter = 1 if node_id in inter_root_set else 0

            # 3. ID 大小 (我们希望 ID 小的优先，所以在 tuple 里取负数，
            #    这样 sorted(reverse=True) 时，ID 小的 (-ID 大) 会排在前面)
            #    Wait, let's simplify. Let's use reverse=True (Desc).
            #    Origin(1) > Non-Origin(0).
            #    Inter(1) > Non-Inter(0).
            #    Smaller ID should win. In descending sort, larger value wins.
            #    So we need to transform ID such that smaller ID => larger value.
            #    Use -node_id.
            return (is_origin, is_inter, -node_id)

        # 排序，取第一个作为幸存者
        comp.sort(key=sort_key, reverse=True)
        survivor = comp[0]
        victims = comp[1:]

        # 记录映射
        for v in comp:
            merge_map[v] = survivor

        # === 核心逻辑：合并半径 ===
        # 半径取 component 中最大的
        max_r = 0.0
        for v in comp:
            r = target_G.nodes[v].get('r', 0.0)
            if r > max_r:
                max_r = r
        target_G.nodes[survivor]['r'] = max_r

        # 坐标保持 survivor 的不动 (符合题目隐含需求，不求平均)

    # --- 4. 重构图结构 ---
    new_G = nx.Graph()

    # 添加所有幸存下来的节点
    unique_survivors = set(merge_map.values())
    for n in unique_survivors:
        new_G.add_node(n, **target_G.nodes[n])

    # 添加边 (重映射)
    for u, v in target_G.edges:
        new_u = merge_map[u]
        new_v = merge_map[v]

        # 忽略自环 (合并导致的)
        if new_u != new_v:
            # 如果这两个点之间已经有边了(多重边)，nx.Graph 会自动忽略或覆盖
            new_G.add_edge(new_u, new_v)

    # --- 5. 更新 ID 列表 ---

    # 更新 Target Root
    new_target_root_id = merge_map[target_root_id]

    # 更新 Origin Node IDs
    new_origin_ids = set()
    for oid in origin_node_ids:
        new_origin_ids.add(merge_map[oid])
    new_origin_ids = list(new_origin_ids)

    # 更新 Inter Root IDs
    # 注意：如果 InterRoot 被合并进了 Origin，这里 ID 会变成 Origin 的 ID
    new_inter_root_ids = set()
    for iid in inter_root_ids:
        survivor = merge_map[iid]
        # 【决策点】:
        # 如果 Survivor 变成了 Origin 节点，我们要不要把它保留在 InterRoot 列表中？
        # 根据你之前的逻辑（Feature 2: 0,1,2），如果一个点同时是 Origin 和 InterRoot，
        # 你的优先级是 Origin (Type 1) > InterRoot (Type 2)。
        # 所以即使这里保留了，在 convert_to_gnn_data 里也会被标记为 Type 1。
        # 此时保留它是安全的。
        new_inter_root_ids.add(survivor)
    new_inter_root_ids = list(new_inter_root_ids)

    return new_G, new_origin_ids, new_target_root_id, new_inter_root_ids


def convert_to_gnn_data(target_G, origin_node_ids, target_root_id, inter_root_ids, path_distances, angle_features):
    """
    Features:
    0: Radius (r)
    1: Node Type (0=Other, 1=TargetRoot, 2=InterRoot)
    2: Path Distance (Geodesic)
    3: Branch Angle (Radians)  <-- NEW
    """

    nodes_list = list(target_G.nodes)
    node_mapping = {node_id: idx for idx, node_id in enumerate(nodes_list)}

    origin_set = set(origin_node_ids)
    inter_root_set = set(inter_root_ids)

    x_features = []
    y_labels = []

    for node_id in nodes_list:
        node_data = target_G.nodes[node_id]

        # --- Feature 0: Radius ---
        r = node_data.get('r', 1.0)

        # --- Feature 1: Type ---
        if node_id == target_root_id:
            node_type = 1.0
        elif node_id in inter_root_set:
            node_type = 2.0
        else:
            node_type = 0.0

        # --- Feature 2: Distance ---
        dist = path_distances.get(node_id, -1.0)

        # --- Feature 3: Angle ---
        angle = angle_features.get(node_id, 0.0)

        # 组装
        x_features.append([r, node_type, dist, angle])

        # --- Label ---
        label = 1 if node_id in origin_set else 0
        y_labels.append(label)

    # Convert to Tensors
    x = torch.tensor(x_features, dtype=torch.float)
    y = torch.tensor(y_labels, dtype=torch.float)

    # Edges
    src_list, dst_list = [], []
    for u, v in target_G.edges:
        u_idx, v_idx = node_mapping[u], node_mapping[v]
        src_list.extend([u_idx, v_idx])
        dst_list.extend([v_idx, u_idx])
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Pos
    pos = torch.tensor([[target_G.nodes[n]['x'], target_G.nodes[n]['y'], target_G.nodes[n]['z']] for n in nodes_list], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    return data


def generate_dataset(target_path, interfer_paths, out_path, dist_threshold=1.0):
    # 1. 生成 (Generation)
    target_G, root, inter_roots, origin_ids = generate_interfer_tree(target_path, interfer_paths)

    # 2. 预处理：合并节点 (Preprocessing)
    # 注意：返回值会更新 root 和 inter_roots 的 ID
    target_G, origin_ids, root, inter_roots = preprocess_merge_nodes(
        target_G, origin_ids, root, inter_roots, dist_threshold=dist_threshold
    )

    # 3. 特征工程 A: 测地距离
    path_distances = compute_geodesic_distance(target_G, root_id=root)

    # 4. 特征工程 B: 分枝夹角 [新增]
    angle_features = compute_branch_angles(target_G, root_id=root)

    # 5. 转换为 GNN Data
    data = convert_to_gnn_data(
        target_G,
        origin_ids,
        root,
        inter_roots,
        path_distances,
        angle_features
    )

    print(f"Feature vector size: {data.x.shape}")
    # Output: [N, 4] -> (Radius, Type, Dist, Angle)

    # save
    torch.save(data, out_path)


if __name__ == "__main__":
    swc_pool = f"/data2/kfchen/tracing_ws/morphology_seg/auto8k_resampled_10um"
    output_dir = f"/data2/kfchen/tracing_ws/morphology_seg/synthesis_data"
    os.makedirs(output_dir, exist_ok=True)

    swc_files = [f for f in os.listdir(swc_pool) if f.endswith('.swc')]
    random_num_range = (5, 10)
    synthesis_data_num = 2000

    for i in range(synthesis_data_num):
        target_file = random.choice(swc_files)
        target_path = os.path.join(swc_pool, target_file)

        num_interfer = random.randint(*random_num_range)
        interfer_files = random.choices(swc_files, k=num_interfer)
        interfer_paths = [os.path.join(swc_pool, f) for f in interfer_files]

        out_path = os.path.join(output_dir, target_file.split('_')[0] + f"_synth_{i}.pt")

        try:
            generate_dataset(target_path, interfer_paths, out_path)
            print(f"Generated synthesis data saved to: {out_path}")
        except Exception as e:
            print(f"Failed to generate synthesis data for {target_file} with error: {e}")