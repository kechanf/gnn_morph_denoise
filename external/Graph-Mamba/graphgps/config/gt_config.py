from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gt')
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'SANLayer'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Size of the edge embedding
    cfg.gt.dim_edge = None

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # Type of extra edges used for transformer
    cfg.gt.secondary_edges = 'full_graph'

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    # Mamba+GNN 融合方式: sum (默认) | conflict_aware (逐维门控)
    cfg.gt.fusion = 'sum'
    cfg.gt.fusion_beta = 1.0
    # 为 True 时 gate 零初始化，使初始 alpha ≈ 0.5（两端等权）
    cfg.gt.fusion_gate_init_zero = False

    # Mamba_GNNPriorityBFS: 辅助 BCE loss 权重；dist 正弦编码维度
    cfg.gt.gnn_priority_aux_weight = 0.5
    cfg.gt.gnn_priority_pe_dim = 16

    cfg.gt.residual = True

    cfg.gt.activation = 'relu'

    # BigBird model/GPS-BigBird layer.
    cfg.gt.bigbird = CN()

    cfg.gt.bigbird.attention_type = "block_sparse"

    cfg.gt.bigbird.chunk_size_feed_forward = 0

    cfg.gt.bigbird.is_decoder = False

    cfg.gt.bigbird.add_cross_attention = False

    cfg.gt.bigbird.hidden_act = "relu"

    cfg.gt.bigbird.max_position_embeddings = 128

    cfg.gt.bigbird.use_bias = False

    cfg.gt.bigbird.num_random_blocks = 3

    cfg.gt.bigbird.block_size = 3

    cfg.gt.bigbird.layer_norm_eps = 1e-6
