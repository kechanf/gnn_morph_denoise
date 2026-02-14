"""
可选模型子模块：新模型在此文件中实现，通过 build_model(model_type=...) 调用。
不修改 models/build.py 中的现有 _Stacked* 类，仅在本文件新增类并在 build 中增加分支即可。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# 在此处添加你的新模型类，接口需与 build_model 参数一致：
#   __init__(self, in_ch, hidden_ch, out_ch, num_layers, dropout)
#   forward(self, data) -> [N, out_ch] 节点 logits
# ---------------------------------------------------------------------------


class StackedGCNResidual(nn.Module):
    """
    示例：带残差连接的 GCN 栈（与现有 GCN 并存，通过 model_type 切换）。
    """
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers, dropout=0.5):
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
            x_new = conv(x, edge_index)
            if i > 0 and x_new.size(-1) == x.size(-1):
                x = x + x_new  # residual
            else:
                x = x_new
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# 供 build_model 按字符串选用的映射（在 build.py 中会按 model_type 查此映射或直接 import 本模块类）
def get_custom_model(model_type: str, in_ch: int, hidden_ch: int, out_ch: int, num_layers: int, dropout: float):
    """在子模块内根据 model_type 返回新模型实例，便于 build.py 只加一行调用。"""
    model_type = model_type.lower()
    if model_type == "gcn_residual":
        return StackedGCNResidual(in_ch, hidden_ch, out_ch, num_layers, dropout)
    raise ValueError(f"Unknown custom model_type: {model_type}")
