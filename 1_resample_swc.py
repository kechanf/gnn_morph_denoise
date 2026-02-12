from neuroutils.swc.parser import safe_resample_swc_file
import os
from tqdm import tqdm

step = 10  # 10um
input_dir = "/data2/kfchen/tracing_ws/morphology_seg/auto8.4k_0510_resample1um_mergedBranches0712"
output_dir = f"/data2/kfchen/tracing_ws/morphology_seg/auto8k_resampled_{step}um"
os.makedirs(output_dir, exist_ok=True)

swc_files = [f for f in os.listdir(input_dir) if f.endswith('.swc')]
for swc_file in tqdm(swc_files):
    input_path = os.path.join(input_dir, swc_file)
    output_path = os.path.join(output_dir, swc_file)
    safe_resample_swc_file(input_path, output_path, step=step)