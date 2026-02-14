"""
图节点特征预处理：One-Hot(node_type) + 图内标准化(radius, dist, angle)。
与训练/测试阶段保持一致。
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    """对有效值 (>=0) 做图内标准化，无效位置保持不变。"""
    mask = tensor >= 0
    if mask.sum() > 1:
        mean = tensor[mask].mean()
        std = tensor[mask].std() + 1e-6
        return (tensor - mean) / std
    return tensor


def preprocess_features(data: Data) -> Data:
    """
    原始特征 [Radius, Type, Dist, Angle] →
    [radius_norm, type_one_hot(3), dist_norm, angle_norm]，共 6 维；
    并将 data.y 转为 long。
    """
    x = data.x
    radius = x[:, 0:1]
    node_type = x[:, 1].long()
    dist = x[:, 2:3]
    angle = x[:, 3:4]

    type_one_hot = F.one_hot(node_type, num_classes=3).float()
    radius_norm = standardize(radius)
    dist_norm = standardize(dist)
    angle_norm = standardize(angle)

    data.x = torch.cat([radius_norm, type_one_hot, dist_norm, angle_norm], dim=1)
    data.y = data.y.long()
    return data
