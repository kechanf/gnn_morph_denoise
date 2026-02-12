"""
Morphology node classification dataset: loads .pt graph files from a directory.
Each .pt is a PyG Data with x (Radius, Type, Dist, Angle), y (node labels 0/1), edge_index, pos.
Preprocessing: [Radius, Type, Dist, Angle] -> 6-dim (radius_norm, type_one_hot(3), dist_norm, angle_norm).
Used by gnn_project with Graph-Mamba as baseline.
"""
import glob
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def _standardize(tensor):
    mask = tensor >= 0
    if mask.sum() > 1:
        mean = tensor[mask].mean()
        std = tensor[mask].std() + 1e-6
        return (tensor - mean) / std
    return tensor


def _preprocess_features(data):
    """[Radius, Type, Dist, Angle] -> 6-dim; y to long; 为 GatedGCN 构造 edge_attr。"""
    x = data.x
    radius = x[:, 0:1]
    node_type = x[:, 1].long()
    dist = x[:, 2:3]
    angle = x[:, 3:4]
    type_one_hot = F.one_hot(node_type, num_classes=3).float()
    data.x = torch.cat([
        _standardize(radius),
        type_one_hot,
        _standardize(dist),
        _standardize(angle),
    ], dim=1)
    data.y = data.y.long().squeeze(-1) if data.y.dim() > 1 else data.y.long()
    # GatedGCN 需要 edge_attr，用两端点节点特征的均值
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        src, dst = data.edge_index[0], data.edge_index[1]
        data.edge_attr = (data.x[src] + data.x[dst]) / 2.0
    return data


class MorphologyNodeDataset(InMemoryDataset):
    """Inductive node classification: many graphs, each node has label 0/1."""

    def __init__(self, root, transform=None, pre_transform=None):
        self.pt_dir = root  # directory containing *.pt files
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._data = self.data  # GraphGym set_dataset_info expects _data
        self.name = 'morphology-node'

    @property
    def processed_file_names(self):
        return 'morphology_processed.pt'

    def process(self):
        pt_files = sorted(glob.glob(osp.join(self.pt_dir, '*.pt')))
        if not pt_files:
            raise FileNotFoundError(f"No *.pt files in {self.pt_dir}")
        data_list = []
        for path in tqdm(pt_files, desc='Morphology'):
            data = torch.load(path)
            data = _preprocess_features(data)
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = data.x.size(0)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
