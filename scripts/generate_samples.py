"""
生成不同策略的合成样本并导出 SWC，供观察和对比。
用法: python scripts/generate_samples.py
"""
import sys
import os
import random
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.synthesis import generate_interfer_tree, preprocess_merge_nodes
from data.swc_io import save_graph_to_swc
from neuroutils.swc.io import load_swc, swc_to_graph


def export_graph_with_labels(target_G, origin_ids, root, inter_roots, output_dir, prefix):
    """
    导出合成图，并分别导出目标节点和干扰节点的 SWC。
    """
    os.makedirs(output_dir, exist_ok=True)
    origin_set = set(origin_ids)
    
    # 导出完整图（所有节点）
    # 需要确保 bfs_tree 保留节点属性
    full_bfs = nx.bfs_tree(target_G, root)
    # 复制节点属性
    for node in full_bfs.nodes():
        if node in target_G.nodes:
            full_bfs.nodes[node].update(target_G.nodes[node])
    save_graph_to_swc(
        full_bfs,
        os.path.join(output_dir, f"{prefix}_full.swc"),
        description=f"Full merged graph (strategy: {prefix})"
    )
    
    # 导出仅目标节点（需要重建连通图）
    target_only = target_G.subgraph(origin_ids).copy()
    if target_only.number_of_nodes() > 0:
        # 找目标图的根（应该是原始 root）
        if root in target_only.nodes:
            target_root = root
        else:
            target_root = list(target_only.nodes)[0]
        if target_only.number_of_nodes() > 1:
            # 重建边（只保留原图中存在的边）
            target_edges = [(u, v) for u, v in target_G.edges() 
                           if u in origin_set and v in origin_set]
            target_only.clear_edges()
            target_only.add_edges_from(target_edges)
            # 取最大连通分量
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
                description=f"Target nodes only"
            )
    
    # 导出仅干扰节点
    inter_ids = [n for n in target_G.nodes if n not in origin_set]
    if inter_ids:
        inter_only = target_G.subgraph(inter_ids).copy()
        if inter_only.number_of_nodes() > 1:
            inter_edges = [(u, v) for u, v in target_G.edges() 
                          if u in inter_ids and v in inter_ids]
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
                description=f"Interfer nodes only"
            )


def main():
    # 设置随机种子以便复现
    random.seed(42)
    import numpy as np
    np.random.seed(42)
    
    swc_pool = config.SWC_POOL_DIR
    output_dir = os.path.join(config.DATA_ROOT, "sample_visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    swc_files = [f for f in os.listdir(swc_pool) if f.endswith('.swc')]
    if len(swc_files) < 2:
        print(f"Error: Need at least 2 SWC files in {swc_pool}")
        return
    
    # 选择一个目标树
    target_file = random.choice(swc_files)
    target_path = os.path.join(swc_pool, target_file)
    print(f"Target tree: {target_file}")
    
    # 为每种策略生成一个样本
    strategies = ["full_tree", "local_spur", "branch_segment", "small_cluster", "mixed"]
    
    for strategy in strategies:
        print(f"\n=== Generating sample with strategy: {strategy} ===")
        
        # 准备干扰路径（某些策略可能不需要）
        if strategy in ["full_tree", "branch_segment", "mixed"]:
            num_interfer = random.randint(2, 4)
            interfer_files = random.choices(swc_files, k=num_interfer)
            interfer_paths = [os.path.join(swc_pool, f) for f in interfer_files]
        else:
            interfer_paths = []  # local_spur 和 small_cluster 不需要干扰树
        
        try:
            # 生成合成图
            target_G, root, inter_roots, origin_ids = generate_interfer_tree(
                target_path, interfer_paths, strategy=strategy
            )
            
            # 合并近距节点（可选，这里先不合并以便观察原始合成效果）
            # target_G, origin_ids, root, inter_roots = preprocess_merge_nodes(
            #     target_G, origin_ids, root, inter_roots, dist_threshold=config.MERGE_DIST_THRESHOLD
            # )
            
            # 导出
            prefix = f"sample_{strategy}"
            export_graph_with_labels(target_G, origin_ids, root, inter_roots, output_dir, prefix)
            
            print(f"  ✓ Generated: {prefix}")
            print(f"    - Nodes: {target_G.number_of_nodes()} (target: {len(origin_ids)}, interfer: {target_G.number_of_nodes() - len(origin_ids)})")
            print(f"    - Edges: {target_G.number_of_edges()}")
            print(f"    - Files saved to: {output_dir}/{prefix}_*.swc")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Done! Check SWC files in: {output_dir} ===")
    print("You can visualize them with Vaa3D or other SWC viewers.")


if __name__ == "__main__":
    main()
