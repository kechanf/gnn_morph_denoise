"""
生成若干「完整合成树」SWC 样本（只导出 full，不导出 target_only / interfer_only）。

用法:
    在项目根目录执行:
        python scripts/generate_mixed_swc_samples.py
"""
import os
import random
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from data.synthesis import generate_interfer_tree  # noqa: E402
from data.swc_io import save_graph_to_swc  # noqa: E402
from neuroutils.swc.io import load_swc, swc_to_graph  # noqa: E402


def main() -> None:
    random.seed(2026)
    import numpy as np

    np.random.seed(2026)

    swc_pool = config.SWC_POOL_DIR
    output_dir = os.path.join(config.DATA_ROOT, "mixed_swc_samples")
    os.makedirs(output_dir, exist_ok=True)

    swc_files = [f for f in os.listdir(swc_pool) if f.endswith(".swc")]
    if len(swc_files) < 2:
        print(f"SWC_POOL_DIR 中 SWC 数量不足: {swc_pool}")
        return

    num_samples = 10
    low, high = config.SYNTHESIS_NUM_INTERFER_RANGE

    print(f"SWC pool: {swc_pool}, N = {len(swc_files)}")
    print(f"Generating {num_samples} mixed-strategy SWC samples into: {output_dir}")

    for i in range(num_samples):
        target_file = random.choice(swc_files)
        target_path = os.path.join(swc_pool, target_file)

        num_interfer = random.randint(low, high)
        interfer_files = random.choices(swc_files, k=num_interfer)
        interfer_paths = [os.path.join(swc_pool, f) for f in interfer_files]

        prefix = target_file.split("_")[0]
        base_name = f"{prefix}_mixed_swc_{i:02d}"

        try:
            # 只做合成 + 噪声，不做节点合并/转换，直接输出 SWC
            target_G, root, inter_roots, origin_ids = generate_interfer_tree(
                target_path, interfer_paths, strategy="mixed"
            )

            # 确保图是定向树形式（从 root 出发 BFS 导向）
            bfs_tree = nx.bfs_tree(target_G, root)
            for node in bfs_tree.nodes():
                if node in target_G.nodes:
                    bfs_tree.nodes[node].update(target_G.nodes[node])

            swc_path = os.path.join(output_dir, f"{base_name}.swc")
            save_graph_to_swc(
                bfs_tree,
                swc_path,
                description=f"Mixed-strategy synthetic tree {i}",
            )
            print(f"  [OK] {swc_path}  (nodes={target_G.number_of_nodes()}, edges={target_G.number_of_edges()})")

        except Exception as e:
            import traceback

            print(f"  [FAIL] target={target_file}, i={i}: {e}")
            traceback.print_exc()

    print("\nDone. Please inspect SWCs in:", output_dir)


if __name__ == "__main__":
    main()

