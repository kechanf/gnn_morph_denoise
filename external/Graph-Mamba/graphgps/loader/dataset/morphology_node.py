"""
Morphology node classification dataset: loads .pt graph files from a directory.
Each .pt is a PyG Data with x (Radius, Type, Dist, Angle), y (node labels 0/1), edge_index, pos.
Preprocessing: [Radius, Type, Dist, Angle] -> 6-dim (radius_norm, type_one_hot(3), dist_norm, angle_norm).
额外字段 dist_from_root, is_target_root 供 Mamba_GNNPriorityBFS 显式使用。
若新增此字段后首次运行，需删除 processed 缓存（morphology_processed.pt）以触发重新处理。
"""
import glob
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# 兼容旧版本保存的 DataEdgeAttr（曾在 torch_geometric.data.data 中定义），
# 新版 PyG 中该类可能不存在，导致 pickle 反序列化失败。这里将其别名为 Data。
try:
    import torch_geometric.data.data as _pyg_data_mod

    # 旧版本中额外定义过 DataEdgeAttr / DataTensorAttr 等辅助类，
    # 新版移除后，旧 pickle 会在反序列化时报找不到类，这里统一别名到 Data。
    if not hasattr(_pyg_data_mod, "DataEdgeAttr"):
        class DataEdgeAttr(Data):
            """Backward-compatible stub for legacy pickles."""
            pass

        _pyg_data_mod.DataEdgeAttr = DataEdgeAttr

    if not hasattr(_pyg_data_mod, "DataTensorAttr"):
        class DataTensorAttr(Data):
            """Backward-compatible stub for legacy pickles."""
            pass

        _pyg_data_mod.DataTensorAttr = DataTensorAttr
except Exception:
    # 若导入失败，不影响后续正常路径；仅在加载旧 pickle 时才需要。
    pass


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
    # 显式保留 dist/root，供 Mamba_GNNPriorityBFS 的 LexSort 与 MLP 使用
    dist_raw = dist.squeeze(-1).clone().float()
    dist_raw[dist_raw < 0] = 1e9  # unreachable -> sort last
    data.dist_from_root = dist_raw
    data.is_target_root = (node_type == 1).float()
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
        # PyTorch>=2.6 默认 weights_only=True，会阻止包含自定义类（如
        # torch_geometric.data.Data）的对象被反序列化。这里文件由本项目生成且
        # 来源可信，显式关闭 weights_only 以兼容旧版本 processed 缓存。
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
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
            # 同理，原始 *.pt 也是由本项目生成的 torch_geometric.data.Data 对象，
            # 显式设置 weights_only=False 以避免 PyTorch>=2.6 的安全限制。
            data = torch.load(path, weights_only=False)
            data = _preprocess_features(data)
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = data.x.size(0)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
