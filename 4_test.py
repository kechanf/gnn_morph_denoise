import os
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph

# ==========================================
# 1. 基础组件 (Model & Preprocess)
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)

def preprocess_features(data):
    x = data.x
    # [Radius, Type, Dist, Angle]
    radius, node_type, dist, angle = x[:, 0:1], x[:, 1].long(), x[:, 2:3], x[:, 3:4]

    def standardize(t):
        mask = t >= 0
        if mask.sum() > 1:
            return (t - t[mask].mean()) / (t[mask].std() + 1e-6)
        return t

    new_x = torch.cat([standardize(radius), F.one_hot(node_type, 3).float(), standardize(dist), standardize(angle)], dim=1)
    data.x = new_x
    return data

# ==========================================
# 2. 解耦的核心逻辑 (Graph Building & IO)
# ==========================================

def build_mst_graph(raw_data, mask_array, k=15):
    """
    输入: raw_data (含坐标), mask_array (0/1)
    输出: NetworkX DiGraph (已重新连通并定向)
    """
    coords = raw_data.pos.numpy()
    features = raw_data.x.numpy()
    radii = features[:, 0]
    types = features[:, 1]

    num_nodes = coords.shape[0]
    indices = np.arange(num_nodes)

    # 策略: Mask=1 或 原始Root 必须保留
    is_root = (types == 1.0)
    keep_mask = (mask_array == 1) | is_root
    survivor_indices = indices[keep_mask]

    if len(survivor_indices) < 2:
        return None

    survivor_coords = coords[survivor_indices]
    survivor_radii = radii[survivor_indices]

    # 寻找新的 Root
    raw_root_idx = np.where(is_root)[0]
    if len(raw_root_idx) > 0 and raw_root_idx[0] in survivor_indices:
        new_root_local_idx = np.where(survivor_indices == raw_root_idx[0])[0][0]
    else:
        new_root_local_idx = 0

    # 构建 KNN + MST
    curr_k = min(k, len(survivor_indices) - 1)
    A = kneighbors_graph(survivor_coords, n_neighbors=curr_k, mode='distance', include_self=False)
    G_knn = nx.from_scipy_sparse_array(A)
    mst = nx.minimum_spanning_tree(G_knn, weight='weight')

    # 只取含 Root 的最大连通分量 (去噪)
    if not nx.is_connected(mst):
        components = list(nx.connected_components(mst))
        for comp in components:
            if new_root_local_idx in comp:
                mst = mst.subgraph(comp).copy()
                break

        # 重整 ID
        final_local_ids = sorted(list(mst.nodes))
        mapping = {old_id: i for i, old_id in enumerate(final_local_ids)}

        survivor_indices = survivor_indices[final_local_ids]
        survivor_coords = survivor_coords[final_local_ids]
        survivor_radii = survivor_radii[final_local_ids]
        new_root_local_idx = mapping[new_root_local_idx]
        mst = nx.relabel_nodes(mst, mapping)

    # 确定父子方向
    dfs_tree = nx.bfs_tree(mst, source=new_root_local_idx)

    # 写入节点属性
    for node_idx in dfs_tree.nodes:
        dfs_tree.nodes[node_idx]['x'] = survivor_coords[node_idx][0]
        dfs_tree.nodes[node_idx]['y'] = survivor_coords[node_idx][1]
        dfs_tree.nodes[node_idx]['z'] = survivor_coords[node_idx][2]
        dfs_tree.nodes[node_idx]['r'] = survivor_radii[node_idx]

    return dfs_tree

def save_graph_to_swc(graph, save_path, description=""):
    """
    输入: NetworkX DiGraph
    输出: SWC 文件
    """
    if graph is None:
        print(f"Skipping {save_path} (Empty Graph)")
        return

    # 找入度为0的点作为Root
    root = [n for n, d in graph.in_degree() if d == 0][0]
    traversal_order = list(nx.bfs_tree(graph, root))

    node_to_swc_id = {}
    swc_lines = []

    for i, node_idx in enumerate(traversal_order):
        swc_id = i + 1
        node_to_swc_id[node_idx] = swc_id
        data = graph.nodes[node_idx]

        preds = list(graph.predecessors(node_idx))
        if not preds:
            parent_id = -1
            n_type = 1 # Soma
        else:
            parent_id = node_to_swc_id[preds[0]]
            n_type = 3 # Dendrite

        line = f"{swc_id} {n_type} {data['x']:.4f} {data['y']:.4f} {data['z']:.4f} {data['r']:.4f} {parent_id}"
        swc_lines.append(line)

    with open(save_path, 'w') as f:
        f.write(f"# {description}\n")
        f.write("# id type x y z r parent\n")
        for line in swc_lines:
            f.write(line + "\n")

# ==========================================
# 3. 主流程 (Main)
# ==========================================
if __name__ == "__main__":
    ws_root = "/data2/kfchen/tracing_ws/morphology_seg"
    MODEL_PATH = ws_root + "/best_model.pth"# 假设你保存了模型
    TEST_DATA_DIR = ws_root + "/synthesis_data"
    OUTPUT_DIR = ws_root + "/results_swc"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    model = GCN(6, 64, 2).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print("Warning: Model not found, using random weights!")
    model.eval()

    # 2. Process Files
    pt_files = sorted([os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.endswith('.pt')])[:10]

    print(f"Processing {len(pt_files)} files...")

    for pt_file in pt_files:
        base_name = os.path.basename(pt_file).replace(".pt", "")

        # A. Load Data
        raw_data = torch.load(pt_file)

        # B. Model Inference
        input_data = raw_data.clone()
        input_data = preprocess_features(input_data).to(device)
        with torch.no_grad():
            pred_mask = model(input_data).argmax(dim=1).cpu().numpy()

        gt_mask = raw_data.y.long().cpu().numpy()

        # === 新增逻辑：全选掩码 (Input) ===
        # 创建一个全为 1 的数组，长度等于节点数
        input_mask = np.ones(raw_data.num_nodes, dtype=np.int64)

        # C. 统一导出队列
        export_tasks = [
            ("input", input_mask, "Raw Input (With Noise)"), # 1. 原始带噪数据
            ("pred",  pred_mask,  "Model Prediction"),       # 2. 模型去噪结果
            ("gt",    gt_mask,    "Ground Truth Label")      # 3. 真值标签
        ]

        for suffix, mask, desc in export_tasks:
            # 构建图
            graph = build_mst_graph(raw_data, mask, k=15)

            # 保存文件
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{suffix}.swc")
            save_graph_to_swc(graph, save_path, description=desc)

        print(f"Processed {base_name} -> Saved input/pred/gt")
