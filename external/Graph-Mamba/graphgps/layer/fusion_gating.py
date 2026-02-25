"""
Conflict-Aware Feature Fusion (逐维门控, Channel-wise Gating).

在 Logit 层面做偏置：α = Sigmoid(MLP(z) - β·Δ_diff)，保证凸组合
H_Fused = α ⊙ H_Mamba + (1-α) ⊙ H_GNN。
Δ_diff = (H_Mamba - H_GNN)^2 为逐维冲突度量；冲突大时减 Mamba logit，偏 GNN。

支持：深度感知 beta 初始化、温度锐化 (tau)、特征正交约束 (orthogonal constraint)。
"""
from typing import Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _depth_aware_beta(layer_idx: int, num_layers: int, beta_max: float = 3.0, beta_min: float = 0.1) -> float:
    """浅层 beta 大（信 GNN），深层 beta 小（包容 Mamba）。"""
    ratio = layer_idx / max(1, num_layers - 1)  # 0.0 (浅) -> 1.0 (深)
    return beta_max * (1.0 - ratio) + beta_min * ratio


class ConflictAwareFusion(nn.Module):
    """
    Channel-wise (逐维) uncertainty-aware gating for fusing Mamba and GNN outputs.
    Logit bias: alpha = Sigmoid((gate_logit - beta * diff) / tau), then convex combination.
    """

    def __init__(
        self,
        dim: int,
        beta: float = 1.0,
        learnable_beta: bool = True,
        gate_init_zero: bool = False,
        log_alpha: bool = False,
        log_alpha_per_layer: bool = False,
        layer_id: Optional[int] = None,
        *,
        layer_idx: Optional[int] = None,
        num_layers: Optional[int] = None,
        use_depth_aware_beta: bool = False,
        tau: float = 1.0,
    ):
        super().__init__()
        self.log_alpha = log_alpha
        self.log_alpha_per_layer = log_alpha_per_layer
        self.layer_id = layer_id
        self.tau = tau
        # 记录配置，便于在日志中还原当前实验设置
        self.use_depth_aware_beta = use_depth_aware_beta
        self.learnable_beta = learnable_beta
        # 最近一次 forward 的特征正交 loss（用于在 train loop 中汇总）
        self.last_ortho_loss: Optional[torch.Tensor] = None

        # gate_net: 单层 Linear(2*dim -> dim) 逐维门控
        self.gate_net = nn.Linear(dim * 2, dim)

        init_beta_value = float(beta)
        if use_depth_aware_beta and layer_idx is not None and num_layers is not None:
            # 深度感知：gate 零初始化（单层 Linear），beta 按层从大到小
            nn.init.zeros_(self.gate_net.weight)
            nn.init.zeros_(self.gate_net.bias)
            init_beta = _depth_aware_beta(layer_idx, num_layers)
            init_beta_value = float(init_beta)
            if learnable_beta:
                self.beta = nn.Parameter(torch.tensor(float(init_beta)))
            else:
                self.register_buffer("beta", torch.tensor(float(init_beta)))
        else:
            if gate_init_zero:
                nn.init.zeros_(self.gate_net.weight)
                nn.init.zeros_(self.gate_net.bias)
            if learnable_beta:
                self.beta = nn.Parameter(torch.tensor(float(beta)))
            else:
                self.register_buffer("beta", torch.tensor(float(beta)))

        # 在模型构建时打一行配置说明，便于日后排查/对比实验
        logger.info(
            "fusion_gating cfg: layer=%s depth_aware_beta=%s learnable_beta=%s "
            "tau=%.3f init_beta=%.3f",
            self.layer_id if self.layer_id is not None else "NA",
            str(self.use_depth_aware_beta),
            str(self.learnable_beta),
            self.tau,
            init_beta_value,
        )

    def forward(self, h_mamba: torch.Tensor, h_gnn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_mamba: [N, dim] Mamba branch output (long-range).
            h_gnn:   [N, dim] GNN branch output (local geometry).
        Returns:
            h_fused: [N, dim] α ⊙ h_mamba + (1-α) ⊙ h_gnn.
        """
        # 0. 特征正交约束：鼓励 H_mamba 与 H_gnn 在每个节点上余弦相似度趋近 0
        #    注意使用 abs(cos) 避免 180° 反相关被当成“好事”
        cos_sim = F.cosine_similarity(h_mamba, h_gnn, dim=-1, eps=1e-8)
        self.last_ortho_loss = cos_sim.abs().mean()

        # 1. 门控主分支（直接基于当前特征做门控）
        z = torch.cat([h_mamba, h_gnn], dim=-1)
        gate_logit = self.gate_net(z)
        beta_safe = self.beta.abs()

        if self.tau != 1.0:
            # 温度锐化：diff 用 L1，logit 除以 tau 推向 0/1
            diff = (h_mamba - h_gnn).abs()
            raw = gate_logit - beta_safe * diff
            alpha = torch.sigmoid(raw / self.tau)
        else:
            diff = (h_mamba - h_gnn).pow(2)
            alpha = torch.sigmoid(gate_logit - beta_safe * diff)

        if self.log_alpha:
            with torch.no_grad():
                mean = alpha.float().mean().item()
                var = alpha.float().var(unbiased=True).item()
            if self.log_alpha_per_layer and self.layer_id is not None:
                logger.info("fusion alpha: layer=%d mean=%.4f var=%.4f", self.layer_id, mean, var)
            else:
                logger.info("fusion alpha: mean=%.4f var=%.4f", mean, var)

        h_fused = alpha * h_mamba + (1.0 - alpha) * h_gnn
        return h_fused
