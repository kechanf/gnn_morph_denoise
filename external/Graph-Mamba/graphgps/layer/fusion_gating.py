"""
Conflict-Aware Feature Fusion (逐维门控, Channel-wise Gating).

在 Logit 层面做偏置：α = Sigmoid(MLP(z) - β·Δ_diff)，保证凸组合
H_Fused = α ⊙ H_Mamba + (1-α) ⊙ H_GNN。
Δ_diff = (H_Mamba - H_GNN)^2 为逐维冲突度量；冲突大时减 Mamba logit，偏 GNN。
"""
import torch
import torch.nn as nn


class ConflictAwareFusion(nn.Module):
    """
    Channel-wise (逐维) uncertainty-aware gating for fusing Mamba and GNN outputs.
    Logit bias: alpha = Sigmoid(gate_logit - beta * diff), then convex combination.
    """

    def __init__(self, dim: int, beta: float = 1.0, learnable_beta: bool = True, gate_init_zero: bool = False):
        super().__init__()
        self.gate_net = nn.Linear(dim * 2, dim)
        if gate_init_zero:
            nn.init.zeros_(self.gate_net.weight)
            nn.init.zeros_(self.gate_net.bias)
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(self, h_mamba: torch.Tensor, h_gnn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_mamba: [N, dim] Mamba branch output (long-range).
            h_gnn:   [N, dim] GNN branch output (local geometry).
        Returns:
            h_fused: [N, dim] α ⊙ h_mamba + (1-α) ⊙ h_gnn.
        """
        # 1. Concat
        z = torch.cat([h_mamba, h_gnn], dim=-1)

        # 2. Base logit (learnable gate)
        gate_logit = self.gate_net(z)

        # 3. Conflict penalty (per-dim squared difference)
        diff = (h_mamba - h_gnn).pow(2)

        # 4. Logit bias: when diff large -> gate_logit reduced -> alpha small -> trust GNN
        alpha = torch.sigmoid(gate_logit - self.beta * diff)

        # 5. Convex combination
        h_fused = alpha * h_mamba + (1.0 - alpha) * h_gnn
        return h_fused
