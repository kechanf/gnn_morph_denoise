"""
入口脚本：根据配置生成合成图数据 (.pt)。
用法: 在项目根目录执行  python scripts/synthesis_data.py

支持单样本超时：若 config.SYNTHESIS_SAMPLE_TIMEOUT_SEC 为正数，
单次生成超过该秒数会跳过该样本并继续下一个，避免卡死。
"""
import sys
import os
import random
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.synthesis import generate_dataset


def _generate_one(args):
    """在子进程中执行单次 generate_dataset，供超时控制。"""
    target_path, interfer_paths, out_path, dist_threshold, strategy = args
    generate_dataset(
        target_path,
        interfer_paths,
        out_path,
        dist_threshold=dist_threshold,
        strategy=strategy,
    )


def main():
    os.makedirs(config.SYNTHESIS_OUTPUT_DIR, exist_ok=True)
    swc_files = [f for f in os.listdir(config.SWC_POOL_DIR) if f.endswith(".swc")]
    low, high = config.SYNTHESIS_NUM_INTERFER_RANGE
    num_samples = config.SYNTHESIS_NUM_SAMPLES
    timeout_sec = getattr(config, "SYNTHESIS_SAMPLE_TIMEOUT_SEC", None) or 0

    for i in range(num_samples):
        target_file = random.choice(swc_files)
        target_path = os.path.join(config.SWC_POOL_DIR, target_file)
        num_interfer = random.randint(low, high)
        interfer_files = random.choices(swc_files, k=num_interfer)
        interfer_paths = [os.path.join(config.SWC_POOL_DIR, f) for f in interfer_files]
        prefix = target_file.split("_")[0]
        out_path = os.path.join(config.SYNTHESIS_OUTPUT_DIR, f"{prefix}_synth_{i}.pt")

        args = (
            target_path,
            interfer_paths,
            out_path,
            config.MERGE_DIST_THRESHOLD,
            "mixed",
        )

        try:
            if timeout_sec > 0:
                proc = multiprocessing.Process(target=_generate_one, args=(args,))
                proc.start()
                proc.join(timeout=timeout_sec)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                    try:
                        os.remove(out_path)
                    except FileNotFoundError:
                        pass
                    print(f"Timeout ({timeout_sec}s) {target_file} (i={i}), skipped")
                elif proc.exitcode != 0:
                    try:
                        os.remove(out_path)
                    except FileNotFoundError:
                        pass
                    print(f"Failed {target_file} (i={i}) exitcode={proc.exitcode}")
            else:
                generate_dataset(
                    target_path,
                    interfer_paths,
                    out_path,
                    dist_threshold=config.MERGE_DIST_THRESHOLD,
                    strategy="mixed",
                )
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{num_samples} samples generated")
        except Exception as e:
            print(f"Failed {target_file} (i={i}): {e}")

    print(f"Done. Output dir: {config.SYNTHESIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
