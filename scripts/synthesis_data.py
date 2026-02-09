"""
入口脚本：根据配置生成合成图数据 (.pt)。
用法: 在项目根目录执行  python scripts/synthesis_data.py
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.synthesis import generate_dataset


def main():
    os.makedirs(config.SYNTHESIS_OUTPUT_DIR, exist_ok=True)
    swc_files = [f for f in os.listdir(config.SWC_POOL_DIR) if f.endswith(".swc")]
    low, high = config.SYNTHESIS_NUM_INTERFER_RANGE
    num_samples = config.SYNTHESIS_NUM_SAMPLES

    for i in range(num_samples):
        target_file = random.choice(swc_files)
        target_path = os.path.join(config.SWC_POOL_DIR, target_file)
        num_interfer = random.randint(low, high)
        interfer_files = random.choices(swc_files, k=num_interfer)
        interfer_paths = [os.path.join(config.SWC_POOL_DIR, f) for f in interfer_files]
        prefix = target_file.split("_")[0]
        out_path = os.path.join(config.SYNTHESIS_OUTPUT_DIR, f"{prefix}_synth_{i}.pt")

        try:
            generate_dataset(
                target_path,
                interfer_paths,
                out_path,
                dist_threshold=config.MERGE_DIST_THRESHOLD,
                strategy="mixed",  # 使用混合策略
            )
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{num_samples} samples generated")
        except Exception as e:
            print(f"Failed {target_file} (i={i}): {e}")

    print(f"Done. Output dir: {config.SYNTHESIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
