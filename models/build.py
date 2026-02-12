"""
根据 model_type / num_layers 等参数构建节点分类 GNN 模型。
支持：gcn, gat, sage, gin。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


def build_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    dropout: float = 0.5,
):
    """
    构建 GNN 节点分类模型。

    model_type: "gcn" | "gat" | "sage" | "gin"
    num_layers: 卷积层数 (>= 1)
    """
    model_type = model_type.lower()
    if model_type == "gcn":
        return _StackedGCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if model_type == "gat":
        return _StackedGAT(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if model_type == "sage":
        return _StackedSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if model_type == "gin":
        return _StackedGIN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    raise ValueError(f"Unknown model_type: {model_type}")


class _StackedGCN(nn.Module):
    """可变层数 GCN：num_layers 个 GCNConv + 分类头。"""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hidden_ch))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_ch, hidden_ch))
        self.classifier = Linear(hidden_ch, out_ch)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class _StackedGAT(nn.Module):
    """可变层数 GAT：单头 GATConv，保持维度一致。"""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_ch, hidden_ch, heads=1, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_ch, hidden_ch, heads=1, concat=True))
        self.classifier = Linear(hidden_ch, out_ch)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class _StackedSAGE(nn.Module):
    """可变层数 GraphSAGE (mean 聚合)。"""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_ch, hidden_ch, aggr="mean"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_ch, hidden_ch, aggr="mean"))
        self.classifier = Linear(hidden_ch, out_ch)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class _StackedGIN(nn.Module):
    """可变层数 GIN：每层一个 GINConv(MLP)。"""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            inp = in_ch if i == 0 else hidden_ch
            mlp = Sequential(
                Linear(inp, hidden_ch),
                nn.ReLU(),
                Linear(hidden_ch, hidden_ch),
            )
            self.convs.append(GINConv(mlp))
        self.classifier = Linear(hidden_ch, out_ch)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
