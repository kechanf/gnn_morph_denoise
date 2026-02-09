"""
SWC 图与文件互写：NetworkX 有向树 ↔ SWC 文件。
"""
import networkx as nx


def save_graph_to_swc(graph: nx.DiGraph, save_path: str, description: str = "") -> None:
    """
    将 NetworkX 有向树（节点含 x, y, z, r）写入 SWC 文件。
    根为入度为 0 的节点，按 BFS 顺序编号。
    """
    if graph is None or graph.number_of_nodes() == 0:
        print(f"Skipping {save_path} (empty graph)")
        return

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
            n_type = 1  # Soma
        else:
            parent_id = node_to_swc_id[preds[0]]
            n_type = 3  # Dendrite
        line = f"{swc_id} {n_type} {data['x']:.4f} {data['y']:.4f} {data['z']:.4f} {data['r']:.4f} {parent_id}"
        swc_lines.append(line)

    with open(save_path, "w") as f:
        f.write(f"# {description}\n")
        f.write("# id type x y z r parent\n")
        for line in swc_lines:
            f.write(line + "\n")
