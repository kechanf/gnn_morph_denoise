"""
生成断裂 + 碎片噪声策略的样本，供人工检查。
用法: 在项目根目录执行  python scripts/generate_break_fragment_samples.py
"""
import os
import random
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from data.synthesis import generate_interfer_tree  # noqa: E402
from data.swc_io import save_graph_to_swc  # noqa: E402


def export_graph_with_labels(target_G, origin_ids, root, output_dir, prefix: str) -> None:
    """
    导出合成图，并分别导出目标节点和干扰节点的 SWC。
    """
    os.makedirs(output_dir, exist_ok=True)
    origin_set = set(origin_ids)

    # 导出完整图
    full_bfs = nx.bfs_tree(target_G, root)
    for node in full_bfs.nodes():
        if node in target_G.nodes:
            full_bfs.nodes[node].update(target_G.nodes[node])
    save_graph_to_swc(
        full_bfs,
        os.path.join(output_dir, f"{prefix}_full.swc"),
        description=f"Full graph (break_fragment)",
    )

    # 导出仅目标节点
    target_ids = list(origin_set)
    target_only = target_G.subgraph(target_ids).copy()
    if target_only.number_of_nodes() > 0:
        if root in target_only.nodes:
            target_root = root
        else:
            target_root = list(target_only.nodes)[0]
        # 保留原有边
        target_edges = [
            (u, v) for u, v in target_G.edges() if u in origin_set and v in origin_set
        ]
        target_only.clear_edges()
        target_only.add_edges_from(target_edges)
        if not nx.is_connected(target_only):
            components = list(nx.connected_components(target_only))
            largest = max(components, key=len)
            target_only = target_only.subgraph(largest).copy()
            if root in largest:
                target_root = root
            else:
                target_root = list(largest)[0]
        target_bfs = nx.bfs_tree(target_only, target_root)
        for node in target_bfs.nodes():
            if node in target_only.nodes:
                target_bfs.nodes[node].update(target_only.nodes[node])
        save_graph_to_swc(
            target_bfs,
            os.path.join(output_dir, f"{prefix}_target_only.swc"),
            description="Target nodes only",
        )

    # 导出仅干扰节点（碎片等）
    inter_ids = [n for n in target_G.nodes if n not in origin_set]
    if inter_ids:
        inter_only = target_G.subgraph(inter_ids).copy()
        if inter_only.number_of_nodes() > 1:
            inter_edges = [
                (u, v) for u, v in target_G.edges() if u in inter_ids and v in inter_ids
            ]
            inter_only.clear_edges()
            inter_only.add_edges_from(inter_edges)
            if not nx.is_connected(inter_only):
                components = list(nx.connected_components(inter_only))
                largest = max(components, key=len)
                inter_only = inter_only.subgraph(largest).copy()
        inter_root = list(inter_only.nodes)[0]
        inter_bfs = nx.bfs_tree(inter_only, inter_root)
        for node in inter_bfs.nodes():
            if node in inter_only.nodes:
                inter_bfs.nodes[node].update(inter_only.nodes[node])
        save_graph_to_swc(
            inter_bfs,
            os.path.join(output_dir, f"{prefix}_interfer_only.swc"),
            description="Interfer nodes only (break_fragment)",
        )


def main() -> None:
    random.seed(123)
    import numpy as np

    np.random.seed(123)

    swc_pool = config.SWC_POOL_DIR
    output_dir = os.path.join(config.DATA_ROOT, "break_fragment_samples")
    os.makedirs(output_dir, exist_ok=True)

    swc_files = [f for f in os.listdir(swc_pool) if f.endswith(".swc")]
    if not swc_files:
        print(f"No SWC files found in {swc_pool}")
        return

    target_file = random.choice(swc_files)
    target_path = os.path.join(swc_pool, target_file)
    print(f"Target tree for break_fragment: {target_file}")

    # 作为碎片来源的 seed 库（从 pool 中选若干棵作为干扰树）
    interfer_files = [f for f in swc_files if f != target_file]
    if not interfer_files:
        interfer_files = swc_files
    # 固定一批干扰 SWC，所有样本共享，便于你对比观察
    num_seed = min(config.BREAK_FRAGMENT_SAMPLE_NUM_SEED_SWCS, len(interfer_files))
    seed_files = random.sample(interfer_files, k=num_seed)
    interfer_paths = [os.path.join(swc_pool, f) for f in seed_files]
    print(f"Using {len(interfer_paths)} interfer SWCs as fragment seeds.")

    for i in range(config.BREAK_FRAGMENT_SAMPLE_NUM):
        print(f"\n=== Sample {i+1}/10 ===")
        try:
            target_G, root, inter_roots, origin_ids = generate_interfer_tree(
                target_path, interfer_paths=interfer_paths, strategy="break_fragment"
            )
            prefix = f"breakfrag_{i:02d}"
            export_graph_with_labels(target_G, origin_ids, root, output_dir, prefix)
            print(
                f"  ✓ Saved {prefix}_*.swc to {output_dir} "
                f"(nodes: {target_G.number_of_nodes()}, inter_roots: {len(inter_roots)})"
            )
        except Exception as e:
            import traceback
            print(f"  ✗ Failed to generate sample {i}: {e}")
            traceback.print_exc()

    print(f"\nDone. Check samples in: {output_dir}")


if __name__ == "__main__":
    main()

