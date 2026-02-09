"""
入口脚本：加载最佳模型，对合成数据做节点预测，并导出 input/pred/gt 三种 SWC。
用法: 在项目根目录执行  python scripts/test.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
from data.swc_io import save_graph_to_swc
from models.gcn import GCN
from utils.features import preprocess_features
from utils.graph_build import build_mst_graph


def main():
    os.makedirs(config.TEST_OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCN(
        config.IN_CHANNELS,
        config.HIDDEN_CHANNELS,
        config.OUT_CHANNELS,
        dropout=config.DROPOUT,
    ).to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded: {config.MODEL_SAVE_PATH}")
    else:
        print("Warning: model not found, using random weights.")
    model.eval()

    pt_files = sorted(
        [
            os.path.join(config.TEST_DATA_DIR, f)
            for f in os.listdir(config.TEST_DATA_DIR)
            if f.endswith(".pt")
        ]
    )
    if config.TEST_NUM_FILES is not None:
        pt_files = pt_files[: config.TEST_NUM_FILES]
    print(f"Processing {len(pt_files)} files -> {config.TEST_OUTPUT_DIR}")

    for pt_file in pt_files:
        base_name = os.path.basename(pt_file).replace(".pt", "")
        raw_data = torch.load(pt_file)

        input_data = raw_data.clone()
        input_data = preprocess_features(input_data).to(device)
        with torch.no_grad():
            pred_mask = model(input_data).argmax(dim=1).cpu().numpy()
        gt_mask = raw_data.y.long().cpu().numpy()
        input_mask = np.ones(raw_data.num_nodes, dtype=np.int64)

        export_tasks = [
            ("input", input_mask, "Raw Input (With Noise)"),
            ("pred", pred_mask, "Model Prediction"),
            ("gt", gt_mask, "Ground Truth Label"),
        ]
        for suffix, mask, desc in export_tasks:
            graph = build_mst_graph(raw_data, mask, k=config.MST_K_NEIGHBORS)
            save_path = os.path.join(config.TEST_OUTPUT_DIR, f"{base_name}_{suffix}.swc")
            save_graph_to_swc(graph, save_path, description=desc)
        print(f"  {base_name} -> input / pred / gt")


if __name__ == "__main__":
    main()
