import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class NoiseGraphEncoder(nn.Module):
    """
    将噪声森林图 (x_noise, edge_index_noise) 编码为图级向量 cond_y，用作 DiGress 的条件信息。

    目前实现为一个极简的两层 GCN + 全局 mean pooling：
    - in_dim:   输入节点特征维度（本项目中为 1，对应单一节点类型占位）；
    - hidden_dim: 中间隐层维度，默认可与 Transformer 的 dx 对齐；
    - out_dim: 输出维度，应与 diffusion_model_discrete 中的 self.ydim 一致，
               这样可以直接与原有 y_t / extra_features.y 做逐元素相加。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          节点特征，shape = (N_total, in_dim)
            edge_index: 边索引，shape = (2, E_total)
            batch:      图索引，shape = (N_total,)

        Returns:
            cond_y: 图级条件向量，shape = (batch_size, out_dim)
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        hg = global_mean_pool(h, batch)  # (B, hidden_dim)
        return self.proj(hg)             # (B, out_dim)


class NoiseNodeClassifier(nn.Module):
    """
    基于噪声森林图 (x_noise, edge_index_noise) 的简单节点二分类器。
    目标：用最小改动，在现有生成式骨干之外，引入一个显式的
          「orig_y（节点是否属于目标树）」监督信号。

    - 输入：噪声森林的抽象节点特征（通常为全 1）和边；
    - 输出：每个节点一个 logit，表示属于目标树 (orig_y=1) 的置信度。
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          节点特征，shape = (N, in_dim)
            edge_index: 边索引，shape = (2, E)

        Returns:
            logits: 每个节点的二分类 logit，shape = (N,)
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        logits = self.head(h).squeeze(-1)
        return logits


