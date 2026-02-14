"""
PyG 图数据集：从目录加载 .pt 图并做实时特征预处理。
"""
import os
import glob

import torch
from torch_geometric.data import Dataset

from utils.features import preprocess_features


class TreeDataset(Dataset):
    """从目录中读取所有 .pt 图，get 时做 One-Hot + 归一化预处理。"""

    def __init__(self, root_dir: str):
        super().__init__(root_dir, transform=None, pre_transform=None)
        self.root_dir = root_dir
        self.pt_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        print(f"Found {len(self.pt_files)} graph files.")

    def len(self) -> int:
        return len(self.pt_files)

    def get(self, idx: int):
        data_path = self.pt_files[idx]
        data = torch.load(data_path)
        data = preprocess_features(data)
        return data
