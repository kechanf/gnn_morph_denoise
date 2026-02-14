"""
SWC 按步长重采样：遍历目录下所有 .swc，重采样后写入输出目录。
"""
import os
from tqdm import tqdm

from neuroutils.swc.parser import safe_resample_swc_file


def resample_swc_dir(input_dir: str, output_dir: str, step_um: float = 10.0) -> int:
    """
    对 input_dir 下所有 .swc 文件按 step_um 重采样，保存到 output_dir。

    Args:
        input_dir: 输入 SWC 目录
        output_dir: 输出 SWC 目录（会自动创建）
        step_um: 重采样步长（微米）

    Returns:
        成功处理的文件数量
    """
    os.makedirs(output_dir, exist_ok=True)
    swc_files = [f for f in os.listdir(input_dir) if f.endswith(".swc")]
    for swc_file in tqdm(swc_files, desc="Resample SWC"):
        input_path = os.path.join(input_dir, swc_file)
        output_path = os.path.join(output_dir, swc_file)
        safe_resample_swc_file(input_path, output_path, step=step_um)
    return len(swc_files)
