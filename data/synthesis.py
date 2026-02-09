"""
合成数据：将目标树与多棵干扰树合并为一张图，并转换为 GNN Data。
"""
import random
import networkx as nx
import numpy as np
import torch
from scipy.spatial import KDTree
from torch_geometric.data import Data

from neuroutils.swc.io import load_swc, swc_to_graph
from neuroutils.swc.transform import translate_tree, apply_random_rotation
from neuroutils.swc.geodesic import compute_geodesic_distance, compute_branch_angles
import config


def get_random_rotation_matrix() -> np.ndarray:
    """
    生成 SO(3) 上均匀分布的随机 3x3 旋转矩阵（QR 分解法）。
    """
    H = np.random.randn(3, 3)
    Q, R = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _unit(v: np.ndarray) -> np.ndarray | None:
    """归一化向量，长度过小则返回 None。"""
    n = np.linalg.norm(v)
    if n < config.ANGLE_CONSTRAINT_EPS:
        return None
    return v / n


def _sample_dir_in_cone(u: np.ndarray, max_deg: float) -> np.ndarray:
    """
    在以 u 为轴、最大张角 max_deg 的圆锥内，采样一个随机方向。
    - u: 单位向量
    """
    u = _unit(u)
    if u is None:
        raise ValueError("reference direction is zero")

    max_rad = np.deg2rad(max_deg)
    # 在 [0, max_rad] 内均匀采样夹角
    phi = np.random.rand() * max_rad
    # 在 [0, 2π) 内随机方位角
    psi = np.random.rand() * 2.0 * np.pi

    # 构造与 u 正交的基底 e1, e2
    # 任选一个不与 u 平行的向量做叉乘
    if abs(u[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(u, tmp))
    if e1 is None:
        e1 = np.array([1.0, 0.0, 0.0])
    e2 = _unit(np.cross(u, e1))
    if e2 is None:
        e2 = np.array([0.0, 1.0, 0.0])

    # 球坐标到直角坐标：在 u 为轴的坐标系中
    return np.cos(phi) * u + np.sin(phi) * (np.cos(psi) * e1 + np.sin(psi) * e2)


def _rotate_fragment_to_match_angle(
    G: nx.Graph,
    base_id: int,
    fragment_node_ids: list[int],
    max_deg: float,
) -> None:
    """
    以 base_id 为中心，对 fragment_node_ids 中的所有点做刚性旋转，
    使 base -> fragment_root 的方向与“已有参考方向”之间的夹角
    在 [0, max_deg] 内的某个随机角度。
    """
    if not fragment_node_ids:
        return

    base = np.array(
        [
            G.nodes[base_id]["x"],
            G.nodes[base_id]["y"],
            G.nodes[base_id]["z"],
        ]
    )

    # 1. 参考方向 u：优先使用 base 的某个已有邻居，否则指向 root 或某个 origin 节点
    neighbors = [n for n in G.neighbors(base_id) if n in G.nodes]
    u = None
    for nb in neighbors:
        v_nb = np.array(
            [G.nodes[nb]["x"], G.nodes[nb]["y"], G.nodes[nb]["z"]]
        ) - base
        u = _unit(v_nb)
        if u is not None:
            break
    if u is None:
        # 没有合适邻居，放弃约束
        return

    # 2. 当前 fragment 根方向 v
    frag_root = fragment_node_ids[0]
    v = np.array(
        [G.nodes[frag_root]["x"], G.nodes[frag_root]["y"], G.nodes[frag_root]["z"]]
    ) - base
    v_norm = _unit(v)
    if v_norm is None:
        return

    # 3. 在圆锥内采样目标方向 w
    try:
        w = _sample_dir_in_cone(u, max_deg)
    except ValueError:
        return
    w = _unit(w)
    if w is None:
        return

    # 4. Rodrigues 公式构造旋转矩阵 R，使 v_norm -> w
    cross = np.cross(v_norm, w)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < config.ANGLE_CONSTRAINT_EPS:
        # 已经几乎对齐，或者反向；直接跳过或做 180 度旋转
        return
    k = cross / cross_norm
    dot = np.clip(np.dot(v_norm, w), -1.0, 1.0)
    angle = np.arccos(dot)

    K = np.array(
        [
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ]
    )
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # 5. 对 fragment_node_ids 做旋转
    for nid in fragment_node_ids:
        p = np.array(
            [G.nodes[nid]["x"], G.nodes[nid]["y"], G.nodes[nid]["z"]]
        )
        rel = p - base
        new_p = R @ rel + base
        G.nodes[nid]["x"] = float(new_p[0])
        G.nodes[nid]["y"] = float(new_p[1])
        G.nodes[nid]["z"] = float(new_p[2])


def attach_full_tree(target_G: nx.Graph, target_root: int, inter_G: nx.Graph, inter_root: int):
    """
    策略1：将整棵干扰树 inter_G 随机接到目标树 target_G 的某个节点上。

    - 在 target_G 上随机选节点 b，在 inter_G 上随机选节点 a
    - 用 max(r_a, r_b) 更新 b 的半径
    - 绕 a 随机旋转干扰树后平移，使 a 与 b 重合，删除 a，将 a 的邻居接到 b
    - 将干扰树其余节点并入 target_G（重新编号），并返回新的 inter_root 对应 id

    Returns:
        (target_G, new_inter_root)
    """
    b = random.choice(list(target_G.nodes))
    bx, by, bz = (
        target_G.nodes[b]["x"],
        target_G.nodes[b]["y"],
        target_G.nodes[b]["z"],
    )
    a = random.choice(list(inter_G.nodes))

    radius_a = inter_G.nodes[a]["r"]
    radius_b = target_G.nodes[b]["r"]
    target_G.nodes[b]["r"] = max(radius_a, radius_b)

    apply_random_rotation(inter_G, center_node_id=a, rot_matrix=get_random_rotation_matrix())
    ax, ay, az = (
        inter_G.nodes[a]["x"],
        inter_G.nodes[a]["y"],
        inter_G.nodes[a]["z"],
    )
    shift = np.array([bx - ax, by - ay, bz - az])
    translate_tree(inter_G, shift)

    neighbors = list(inter_G.neighbors(a))
    inter_G.remove_node(a)

    max_id = max(target_G.nodes)
    id_map = {}
    for old in inter_G.nodes:
        max_id += 1
        id_map[old] = max_id

    fragment_ids: list[int] = []
    for old, new in id_map.items():
        target_G.add_node(new, **inter_G.nodes[old])
        fragment_ids.append(new)
    for (u, v) in inter_G.edges:
        target_G.add_edge(id_map[u], id_map[v])
    for old_nb in neighbors:
        new_nb = id_map[old_nb]
        target_G.add_edge(b, new_nb)

    # 对整棵干扰树做一次角度约束：以 b 为基点，fragment_ids 为片段
    try:
        _rotate_fragment_to_match_angle(
            target_G,
            base_id=b,
            fragment_node_ids=fragment_ids,
            max_deg=config.ANGLE_CONSTRAINT_DEG,
        )
    except Exception:
        # 角度约束是辅助性的，失败时不影响主流程
        pass

    # 若 inter_root 是已删除的 a，则用附着侧子树的“根”（a 的某个邻居）作为新 inter_root
    new_inter_root = id_map.get(inter_root)
    if new_inter_root is None and neighbors:
        new_inter_root = id_map[neighbors[0]]
    return target_G, new_inter_root


def attach_local_spur(target_G: nx.Graph, target_root: int, length: int = 5, max_radius: float = 2.0):
    """
    策略2：在目标树节点上生成局部短刺状假分支（模拟追踪错误）。
    
    - 随机选目标树节点 b
    - 从 b 开始，沿随机方向生成 length 个节点的线性路径
    - 每个节点半径递减，模拟假分支特征
    
    Returns:
        (target_G, new_inter_root) - new_inter_root 是假分支的第一个节点
    """
    b = random.choice(list(target_G.nodes))
    bx, by, bz = target_G.nodes[b]["x"], target_G.nodes[b]["y"], target_G.nodes[b]["z"]
    br = target_G.nodes[b]["r"]
    
    # 随机方向向量（归一化）
    direction = np.random.randn(3)
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    # 步长（微米）
    step_min, step_max = config.LOCAL_SPUR_CFG["step_um_range"]
    step_size = random.uniform(step_min, step_max)
    
    max_id = max(target_G.nodes)
    new_inter_root = max_id + 1
    prev_node = b
    
    for i in range(length):
        new_id = max_id + 1 + i
        offset = direction * step_size * (i + 1)
        new_x, new_y, new_z = bx + offset[0], by + offset[1], bz + offset[2]
        new_r = max_radius * (1.0 - i * 0.15)  # 半径递减
        
        target_G.add_node(new_id, x=new_x, y=new_y, z=new_z, r=max(new_r, 0.5))
        if i == 0:
            target_G.add_edge(b, new_id)
        else:
            target_G.add_edge(prev_node, new_id)
        prev_node = new_id
    
    return target_G, new_inter_root


def attach_branch_segment(target_G: nx.Graph, target_root: int, inter_G: nx.Graph, segment_length: int = 5):
    """
    策略3：从干扰树中提取一段路径（3-10个节点）贴到目标树上。
    
    - 从 inter_G 中随机选起始节点，BFS 提取 segment_length 个节点
    - 将这段路径接到目标树随机节点上
    
    Returns:
        (target_G, new_inter_root) - new_inter_root 是路径的第一个节点
    """
    b = random.choice(list(target_G.nodes))
    bx, by, bz = target_G.nodes[b]["x"], target_G.nodes[b]["y"], target_G.nodes[b]["z"]
    
    # 从干扰树中提取路径
    start_node = random.choice(list(inter_G.nodes))
    visited = {start_node}
    segment_nodes = [start_node]
    queue = [start_node]
    
    while len(segment_nodes) < segment_length and queue:
        current = queue.pop(0)
        neighbors = [n for n in inter_G.neighbors(current) if n not in visited]
        if neighbors:
            next_node = random.choice(neighbors)
            visited.add(next_node)
            segment_nodes.append(next_node)
            queue.append(next_node)
    
    if len(segment_nodes) < 2:
        return target_G, None
    
    # 随机旋转路径
    rot_matrix = get_random_rotation_matrix()
    segment_coords = np.array([[inter_G.nodes[n]["x"], inter_G.nodes[n]["y"], inter_G.nodes[n]["z"]] 
                               for n in segment_nodes])
    center = segment_coords[0]
    segment_coords_centered = segment_coords - center
    segment_coords_rotated = (rot_matrix @ segment_coords_centered.T).T + center
    
    # 平移使第一个节点与 b 重合
    shift = np.array([bx, by, bz]) - segment_coords_rotated[0]
    segment_coords_final = segment_coords_rotated + shift
    
    # 添加到目标图
    max_id = max(target_G.nodes)
    id_map = {}
    new_inter_root = max_id + 1
    fragment_ids: list[int] = []
    
    for i, old_id in enumerate(segment_nodes):
        new_id = max_id + 1 + i
        id_map[old_id] = new_id
        target_G.add_node(
            new_id,
            x=segment_coords_final[i][0],
            y=segment_coords_final[i][1],
            z=segment_coords_final[i][2],
            r=inter_G.nodes[old_id]["r"]
        )
        fragment_ids.append(new_id)
    
    # 添加边（保持路径结构）
    for i in range(len(segment_nodes) - 1):
        target_G.add_edge(id_map[segment_nodes[i]], id_map[segment_nodes[i + 1]])
    
    # 将路径起点接到 b
    target_G.add_edge(b, new_inter_root)

    # 角度约束：让新分支与 b 的已有方向形成小于阈值的随机角度
    try:
        _rotate_fragment_to_match_angle(
            target_G,
            base_id=b,
            fragment_node_ids=fragment_ids,
            max_deg=config.ANGLE_CONSTRAINT_DEG,
        )
    except Exception:
        pass
    
    return target_G, new_inter_root


def attach_small_cluster(target_G: nx.Graph, target_root: int, num_points: int = 3, radius_um: float = 5.0):
    """
    策略4：在目标树节点附近生成小的随机点簇（模拟成像伪影/噪声点）。
    
    - 随机选目标树节点 b
    - 在 b 周围半径 radius_um 内生成 num_points 个随机点
    - 这些点形成一个小簇（用最小生成树连接）
    
    Returns:
        (target_G, new_inter_root) - new_inter_root 是簇中第一个节点
    """
    b = random.choice(list(target_G.nodes))
    bx, by, bz = target_G.nodes[b]["x"], target_G.nodes[b]["y"], target_G.nodes[b]["z"]
    br = target_G.nodes[b]["r"]
    
    max_id = max(target_G.nodes)
    new_inter_root = max_id + 1
    
    # 生成随机点（球面均匀分布）
    points = []
    for _ in range(num_points):
        # 球面均匀采样
        u, v = np.random.rand(2)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        r_dist = radius_um * np.random.rand() ** (1 / 3)  # 球内均匀
        
        x = bx + r_dist * np.sin(phi) * np.cos(theta)
        y = by + r_dist * np.sin(phi) * np.sin(theta)
        z = bz + r_dist * np.cos(phi)
        points.append((x, y, z))
    
    # 添加到图
    cluster_ids = []
    for i, (x, y, z) in enumerate(points):
        node_id = max_id + 1 + i
        cluster_ids.append(node_id)
        target_G.add_node(node_id, x=x, y=y, z=z, r=br * random.uniform(0.3, 0.8))
    
    # 用最小生成树连接簇内点
    if len(cluster_ids) > 1:
        from scipy.spatial.distance import pdist, squareform
        coords = np.array(points)
        dists = squareform(pdist(coords))
        # 构建完全图并取 MST
        cluster_G = nx.Graph()
        for i, nid1 in enumerate(cluster_ids):
            for j, nid2 in enumerate(cluster_ids):
                if i < j:
                    cluster_G.add_edge(nid1, nid2, weight=dists[i, j])
        mst = nx.minimum_spanning_tree(cluster_G)
        for u, v in mst.edges():
            target_G.add_edge(u, v)
    
    # 将簇连接到 b
    target_G.add_edge(b, new_inter_root)

    # 角度约束：以 b 为基点，对整个簇做一次旋转
    try:
        _rotate_fragment_to_match_angle(
            target_G,
            base_id=b,
            fragment_node_ids=cluster_ids,
            max_deg=config.ANGLE_CONSTRAINT_DEG,
        )
    except Exception:
        pass
    
    return target_G, new_inter_root


# 高级策略：断裂 + 碎片噪声（碎片来自真实 SWC 片段）
def attach_break_fragment(
    target_G: nx.Graph,
    target_root: int,
    origin_node_ids: list,
    interfer_paths: list[str],
    num_break_edges: int = 4,
    num_fragments: int = 8,
) -> tuple[nx.Graph, list[int]]:
    """
    高级策略：在目标树中制造“断裂 + 碎片噪声”，碎片来自 seed 库中的真实树片段。

    - 随机断开若干条边（造成树的多处断裂）
    - 从 interfer SWC 中截取多个小分支片段，平移/旋转后放到断裂点附近
    - 优先将碎片与「空间上接近」的目标节点/断裂端点建立虚假连接（而非完全随机）

    Returns:
        (target_G, new_inter_roots)
    """
    new_inter_roots: list[int] = []

    # 若没有可用的干扰 SWC，则暂时不添加碎片，只做断裂
    if not interfer_paths:
        interfer_graphs: list[nx.Graph] = []
    else:
        interfer_graphs = []
        for p in interfer_paths:
            try:
                inter_df = load_swc(p)
                interfer_graphs.append(swc_to_graph(inter_df))
            except Exception:
                continue

    # 1. 选择要断开的边（避免把 root 的入边当成断裂目标）
    edges = list(target_G.edges)
    if not edges:
        return target_G, new_inter_roots

    candidate_edges = [(u, v) for u, v in edges if v != target_root]
    if not candidate_edges:
        candidate_edges = edges

    k = min(num_break_edges, len(candidate_edges))
    broken_edges = random.sample(candidate_edges, k=k)
    for u, v in broken_edges:
        if target_G.has_edge(u, v):
            target_G.remove_edge(u, v)

    break_nodes = list({n for e in broken_edges for n in e})

    # 2. 为“近距虚假连接”建立 KDTree（仅对 origin 节点）
    origin_set = set(origin_node_ids)
    origin_nodes = list(origin_set)
    origin_coords = np.array(
        [
            [target_G.nodes[n]["x"], target_G.nodes[n]["y"], target_G.nodes[n]["z"]]
            for n in origin_nodes
        ]
    )
    origin_kdtree = KDTree(origin_coords) if len(origin_nodes) > 0 else None

    max_id = max(target_G.nodes)

    def sample_segment_from_interfer() -> list[int] | None:
        """从随机干扰树中截取 3-10 个节点的路径。"""
        if not interfer_graphs:
            return None
        for _ in range(5):  # 尝试多次
            inter_G = random.choice(interfer_graphs)
            if inter_G.number_of_nodes() < 2:
                continue
            start_node = random.choice(list(inter_G.nodes))
            visited = {start_node}
            segment_nodes = [start_node]
            queue = [start_node]
            seg_len = random.randint(3, 10)
            while len(segment_nodes) < seg_len and queue:
                cur = queue.pop(0)
                neigh = [n for n in inter_G.neighbors(cur) if n not in visited]
                if neigh:
                    nxt = random.choice(neigh)
                    visited.add(nxt)
                    segment_nodes.append(nxt)
                    queue.append(nxt)
            if len(segment_nodes) >= 2:
                return segment_nodes
        return None

    # 3. 在断裂点附近加入多个真实碎片
    for _ in range(num_fragments):
        if not broken_edges:
            break
        base_edge = random.choice(broken_edges)
        base_node = random.choice(list(base_edge))
        bx = target_G.nodes[base_node]["x"]
        by = target_G.nodes[base_node]["y"]
        bz = target_G.nodes[base_node]["z"]
        br = target_G.nodes[base_node]["r"]

        seg_nodes = sample_segment_from_interfer()
        if not seg_nodes:
            continue

        # 取该段在原树中的坐标
        # 这里假设所有干扰树都使用相同的节点属性布局
        inter_G = None
        for g in interfer_graphs:
            if seg_nodes[0] in g.nodes:
                inter_G = g
                break
        if inter_G is None:
            continue

        seg_coords = np.array(
            [[inter_G.nodes[n]["x"], inter_G.nodes[n]["y"], inter_G.nodes[n]["z"]] for n in seg_nodes]
        )
        # 随机旋转
        rot = get_random_rotation_matrix()
        center = seg_coords[0]
        seg_centered = seg_coords - center
        seg_rot = (rot @ seg_centered.T).T + center

        # 平移到断裂点附近，加一点随机扰动，增大距离
        jitter_radius = random.uniform(5.0, 15.0)
        u1, v1 = np.random.rand(2)
        theta = 2 * np.pi * u1
        phi = np.arccos(2 * v1 - 1)
        jx = jitter_radius * np.sin(phi) * np.cos(theta)
        jy = jitter_radius * np.sin(phi) * np.sin(theta)
        jz = jitter_radius * np.cos(phi)
        shift = np.array([bx + jx, by + jy, bz + jz]) - seg_rot[0]
        seg_final = seg_rot + shift

        # 添加到目标图
        fragment_ids: list[int] = []
        for i, old_id in enumerate(seg_nodes):
            max_id += 1
            nid = max_id
            fragment_ids.append(nid)
            target_G.add_node(
                nid,
                x=seg_final[i][0],
                y=seg_final[i][1],
                z=seg_final[i][2],
                r=br * random.uniform(0.4, 0.9),
            )

        # 保持片段内部拓扑（按原顺序链式连接）
        for i in range(len(fragment_ids) - 1):
            target_G.add_edge(fragment_ids[i], fragment_ids[i + 1])

        # 片段第一个节点作为干扰根
        new_root = fragment_ids[0]
        new_inter_roots.append(new_root)

        # 与“附近的目标节点”建立错误连接（而不是完全随机），并施加角度约束
        if origin_kdtree is not None and origin_nodes:
            root_coord = np.array([[seg_final[0][0], seg_final[0][1], seg_final[0][2]]])
            dist, idx = origin_kdtree.query(root_coord, k=min(3, len(origin_nodes)))
            # KDTree.query 返回形状可能是 (1, k) 或标量，这里统一展平成 1D
            dist = np.array(dist).reshape(-1)
            idx = np.array(idx).reshape(-1)
            # 选一个在一定半径内的最近节点（空间上靠近且不是 root 本身）
            candidates = [
                origin_nodes[int(i)]
                for d, i in zip(dist, idx)
                if float(d) < config.BREAK_FRAGMENT_CFG["near_origin_radius_um"]
                and origin_nodes[int(i)] != target_root
            ]
            if candidates:
                wrong_anchor = random.choice(candidates)
                # 角度约束：以 wrong_anchor 为基点旋转整个碎片
                try:
                    _rotate_fragment_to_match_angle(
                        target_G,
                        base_id=wrong_anchor,
                        fragment_node_ids=fragment_ids,
                        max_deg=config.ANGLE_CONSTRAINT_DEG,
                    )
                except Exception:
                    pass
                target_G.add_edge(new_root, wrong_anchor)

        # 再与“虚假断裂点”建立若干连接，进一步扰乱拓扑
        if break_nodes:
            # 只连接到空间上比较近的断裂端点
            bx_list = np.array(
                [[target_G.nodes[n]["x"], target_G.nodes[n]["y"], target_G.nodes[n]["z"]] for n in break_nodes]
            )
            root_coord = np.array([seg_final[0]])
            dists = np.linalg.norm(bx_list - root_coord, axis=1)
            close_idxs = [
                i
                for i, d in enumerate(dists)
                if d < config.BREAK_FRAGMENT_CFG["near_break_radius_um"]
            ]
            if close_idxs:
                for j in random.sample(close_idxs, k=min(2, len(close_idxs))):
                    target_G.add_edge(new_root, break_nodes[j])

    return target_G, new_inter_roots


# 保持向后兼容
attach_interfer = attach_full_tree


def generate_interfer_tree(target_path: str, interfer_paths: list, strategy: str = "full_tree"):
    """
    读取目标 SWC 与多棵干扰 SWC，按指定策略合并为一棵图。
    
    Args:
        target_path: 目标 SWC 路径
        interfer_paths: 干扰 SWC 路径列表
        strategy: 合成策略
            - "full_tree": 整棵干扰树（默认）
            - "local_spur": 局部短刺（忽略 interfer_paths，生成假分支）
            - "branch_segment": 干扰树的一段路径
            - "small_cluster": 小点簇（忽略 interfer_paths）
            - "break_fragment": 断裂 + 碎片噪声（仅作用于 target）
            - "mixed": 按权重随机混合上述策略

    Returns:
        (target_G, root, inter_roots, origin_target_point_n)
    """
    df = load_swc(target_path)
    target_G = swc_to_graph(df)
    origin_target_point_n = list(target_G.nodes)
    root = int(df.loc[df["parent"] == -1, "n"].values[0])
    inter_roots: list[int] = []

    if strategy == "full_tree":
        for ipath in interfer_paths:
            inter_df = load_swc(ipath)
            inter_G = swc_to_graph(inter_df)
            inter_root = int(inter_df.loc[inter_df["parent"] == -1, "n"].values[0])
            target_G, new_inter_root = attach_full_tree(target_G, root, inter_G, inter_root)
            if new_inter_root is not None:
                inter_roots.append(new_inter_root)
    
    elif strategy == "local_spur":
        length_min, length_max = config.LOCAL_SPUR_CFG["length_range"]
        num_spurs = len(interfer_paths) if interfer_paths else random.randint(3, 8)
        for _ in range(num_spurs):
            target_G, new_inter_root = attach_local_spur(
                target_G, root, length=random.randint(length_min, length_max),
                max_radius=config.LOCAL_SPUR_CFG["max_radius"],
            )
            if new_inter_root is not None:
                inter_roots.append(new_inter_root)
    
    elif strategy == "branch_segment":
        for ipath in interfer_paths:
            inter_df = load_swc(ipath)
            inter_G = swc_to_graph(inter_df)
            length_min, length_max = config.BRANCH_SEGMENT_CFG["length_range"]
            target_G, new_inter_root = attach_branch_segment(
                target_G, root, inter_G, segment_length=random.randint(length_min, length_max)
            )
            if new_inter_root is not None:
                inter_roots.append(new_inter_root)
    
    elif strategy == "small_cluster":
        num_min, num_max = config.SMALL_CLUSTER_CFG["num_points_range"]
        r_min, r_max = config.SMALL_CLUSTER_CFG["radius_um_range"]
        num_clusters = len(interfer_paths) if interfer_paths else random.randint(2, 5)
        for _ in range(num_clusters):
            target_G, new_inter_root = attach_small_cluster(
                target_G,
                root,
                num_points=random.randint(num_min, num_max),
                radius_um=random.uniform(r_min, r_max),
            )
            if new_inter_root is not None:
                inter_roots.append(new_inter_root)

    elif strategy == "break_fragment":
        cfg = config.BREAK_FRAGMENT_CFG
        be_min, be_max = cfg["num_break_edges_range"]
        fr_min, fr_max = cfg["num_fragments_range"]
        target_G, new_roots = attach_break_fragment(
            target_G,
            root,
            origin_target_point_n,
            interfer_paths,
            num_break_edges=random.randint(be_min, be_max),
            num_fragments=random.randint(fr_min, fr_max),
        )
        inter_roots.extend(new_roots)
    
    elif strategy == "mixed":
        # 使用配置中的权重控制各策略的使用比值
        simple_w = config.SIMPLE_NEG_STRATEGY_WEIGHTS
        adv_w = config.ADV_STRATEGY_WEIGHTS

        strategies_list: list[str] = []
        weights: list[float] = []

        for name, w in simple_w.items():
            if w > 0:
                strategies_list.append(name)
                weights.append(float(w))
        for name, w in adv_w.items():
            if w > 0:
                strategies_list.append(name)
                weights.append(float(w))

        if not strategies_list:
            raise ValueError("No strategy enabled in SIMPLE_NEG_STRATEGY_WEIGHTS / ADV_STRATEGY_WEIGHTS")

        # 每个 interfer_path 触发一次策略应用；不依赖 interfer_paths 的策略会忽略 ipath
        min_inj = config.MIXED_CFG["min_injections"]
        n_iter = max(len(interfer_paths), min_inj)
        for idx in range(n_iter):
            s = random.choices(strategies_list, weights=weights, k=1)[0]
            ipath = interfer_paths[idx % len(interfer_paths)] if interfer_paths else None

            if s == "full_tree" and ipath is not None:
                inter_df = load_swc(ipath)
                inter_G = swc_to_graph(inter_df)
                inter_root = int(inter_df.loc[inter_df["parent"] == -1, "n"].values[0])
                target_G, new_inter_root = attach_full_tree(target_G, root, inter_G, inter_root)
                if new_inter_root is not None:
                    inter_roots.append(new_inter_root)

            elif s == "local_spur":
                target_G, new_inter_root = attach_local_spur(
                    target_G, root, length=random.randint(3, 8)
                )
                if new_inter_root is not None:
                    inter_roots.append(new_inter_root)

            elif s == "branch_segment" and ipath is not None:
                inter_df = load_swc(ipath)
                inter_G = swc_to_graph(inter_df)
                length_min, length_max = config.BRANCH_SEGMENT_CFG["length_range"]
                target_G, new_inter_root = attach_branch_segment(
                    target_G, root, inter_G, segment_length=random.randint(length_min, length_max)
                )
                if new_inter_root is not None:
                    inter_roots.append(new_inter_root)

            elif s == "small_cluster":
                num_min, num_max = config.SMALL_CLUSTER_CFG["num_points_range"]
                r_min, r_max = config.SMALL_CLUSTER_CFG["radius_um_range"]
                target_G, new_inter_root = attach_small_cluster(
                    target_G,
                    root,
                    num_points=random.randint(num_min, num_max),
                    radius_um=random.uniform(r_min, r_max),
                )
                if new_inter_root is not None:
                    inter_roots.append(new_inter_root)

            elif s == "break_fragment":
                cfg = config.BREAK_FRAGMENT_CFG
                be_min, be_max = cfg["num_break_edges_range"]
                fr_min, fr_max = cfg["num_fragments_range"]
                target_G, new_roots = attach_break_fragment(
                    target_G,
                    root,
                    origin_target_point_n,
                    interfer_paths,
                    num_break_edges=random.randint(be_min, be_max),
                    num_fragments=random.randint(fr_min, fr_max),
                )
                inter_roots.extend(new_roots)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return target_G, root, inter_roots, origin_target_point_n


def preprocess_merge_nodes(
    target_G: nx.Graph,
    origin_node_ids: list,
    target_root_id: int,
    inter_root_ids: list,
    dist_threshold: float = 1.0,
):
    """
    合并距离小于 dist_threshold 的节点。合并时半径取分量内最大值。
    幸存者优先级：Origin > InterRoot > 较小 ID。

    Returns:
        (new_G, new_origin_ids, new_target_root_id, new_inter_root_ids)
    """
    nodes = list(target_G.nodes)
    coords = np.array([
        [target_G.nodes[n]["x"], target_G.nodes[n]["y"], target_G.nodes[n]["z"]]
        for n in nodes
    ])
    origin_set = set(origin_node_ids)
    inter_root_set = set(inter_root_ids)

    tree = KDTree(coords)
    pairs = tree.query_pairs(r=dist_threshold)
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(nodes)
    cluster_graph.add_edges_from([(nodes[i], nodes[j]) for i, j in pairs])
    components = list(nx.connected_components(cluster_graph))

    if len(components) == len(nodes):
        return target_G, origin_node_ids, target_root_id, inter_root_ids

    merge_map = {}

    for comp in components:
        comp = list(comp)
        if len(comp) == 1:
            merge_map[comp[0]] = comp[0]
            continue

        def sort_key(node_id):
            is_origin = 1 if node_id in origin_set else 0
            is_inter = 1 if node_id in inter_root_set else 0
            return (is_origin, is_inter, -node_id)

        comp.sort(key=sort_key, reverse=True)
        survivor = comp[0]
        max_r = max(target_G.nodes[v].get("r", 0.0) for v in comp)
        target_G.nodes[survivor]["r"] = max_r
        for v in comp:
            merge_map[v] = survivor

    new_G = nx.Graph()
    unique_survivors = set(merge_map.values())
    for n in unique_survivors:
        new_G.add_node(n, **target_G.nodes[n])
    for u, v in target_G.edges:
        new_u, new_v = merge_map[u], merge_map[v]
        if new_u != new_v:
            new_G.add_edge(new_u, new_v)

    new_target_root_id = merge_map[target_root_id]
    new_origin_ids = list({merge_map[oid] for oid in origin_node_ids})
    new_inter_root_ids = list({merge_map[iid] for iid in inter_root_ids})

    return new_G, new_origin_ids, new_target_root_id, new_inter_root_ids


def convert_to_gnn_data(
    target_G: nx.Graph,
    origin_node_ids: list,
    target_root_id: int,
    inter_root_ids: list,
    path_distances: dict,
    angle_features: dict,
) -> Data:
    """
    将合并后的图转为 PyG Data。
    节点特征: [r, node_type, path_dist, branch_angle]，标签: 1=目标树, 0=干扰树。
    """
    nodes_list = list(target_G.nodes)
    node_mapping = {node_id: idx for idx, node_id in enumerate(nodes_list)}
    origin_set = set(origin_node_ids)
    inter_root_set = set(inter_root_ids)

    x_features = []
    y_labels = []
    for node_id in nodes_list:
        node_data = target_G.nodes[node_id]
        r = node_data.get("r", 1.0)
        if node_id == target_root_id:
            node_type = 1.0
        elif node_id in inter_root_set:
            node_type = 2.0
        else:
            node_type = 0.0
        dist = path_distances.get(node_id, -1.0)
        angle = angle_features.get(node_id, 0.0)
        x_features.append([r, node_type, dist, angle])
        y_labels.append(1 if node_id in origin_set else 0)

    x = torch.tensor(x_features, dtype=torch.float)
    y = torch.tensor(y_labels, dtype=torch.float)
    src_list, dst_list = [], []
    for u, v in target_G.edges:
        u_idx, v_idx = node_mapping[u], node_mapping[v]
        src_list.extend([u_idx, v_idx])
        dst_list.extend([v_idx, u_idx])
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    pos = torch.tensor(
        [
            [target_G.nodes[n]["x"], target_G.nodes[n]["y"], target_G.nodes[n]["z"]]
            for n in nodes_list
        ],
        dtype=torch.float,
    )
    return Data(x=x, edge_index=edge_index, y=y, pos=pos)


def generate_dataset(
    target_path: str,
    interfer_paths: list,
    out_path: str,
    dist_threshold: float = 1.0,
    strategy: str = "full_tree",
) -> None:
    """
    完整流程：生成合并图 → 合并近距节点 → 测地距离与分枝角 → 转为 GNN Data 并保存。
    
    Args:
        target_path: 目标 SWC 路径
        interfer_paths: 干扰 SWC 路径列表
        out_path: 输出 .pt 路径
        dist_threshold: 节点合并距离阈值
        strategy: 合成策略（见 generate_interfer_tree）
    """
    target_G, root, inter_roots, origin_ids = generate_interfer_tree(
        target_path, interfer_paths, strategy=strategy
    )
    target_G, origin_ids, root, inter_roots = preprocess_merge_nodes(
        target_G, origin_ids, root, inter_roots, dist_threshold=dist_threshold
    )
    path_distances = compute_geodesic_distance(target_G, root_id=root)
    angle_features = compute_branch_angles(target_G, root_id=root)
    data = convert_to_gnn_data(
        target_G, origin_ids, root, inter_roots, path_distances, angle_features
    )
    torch.save(data, out_path)
