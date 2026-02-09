"""
入口脚本：对配置中的输入目录执行 SWC 重采样，输出到配置中的输出目录。
用法: 在项目根目录执行  python scripts/resample_swc.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.resample import resample_swc_dir


def main():
    n = resample_swc_dir(
        config.RESAMPLE_INPUT_DIR,
        config.RESAMPLE_OUTPUT_DIR,
        step_um=config.RESAMPLE_STEP_UM,
    )
    print(f"Resampled {n} SWC files -> {config.RESAMPLE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
