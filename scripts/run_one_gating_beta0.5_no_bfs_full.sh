#!/usr/bin/env bash
# 单次实验：门控 beta=0.5，不加 BFS（baseline 10_10），全量数据；
# 开启深度感知 beta + 温度锐化 tau=0.5，并开启分层 alpha 监测。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
ENV_NAME="${ENV_NAME:-graph-mamba}"
DATA_DIR="${DATA_DIR:-/home/kfchen/gnn_project_local/synthesis_data}"

echo "=== 单实验: 门控 beta=0.5, 无 BFS, 全量 + 深度感知beta + tau=0.5 + 分层监测 ==="
echo "  DATA_DIR: ${DATA_DIR}"
echo "  name_tag: full_gating_beta0.5_no_bfs_depth_tau"
echo ""

conda run -n "${ENV_NAME}" python scripts/run_graph_mamba.py \
  --data_dir "${DATA_DIR}" \
  --wandb False \
  --baseline 10_10 \
  --repeat 1 \
  --name_tag "full_gating_beta0.5_no_bfs_depth_tau" \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 0.5 \
  --override gt.fusion_depth_aware_beta True \
  --override gt.fusion_learnable_beta False \
  --override gt.fusion_tau 0.5 \
  --override gt.fusion_log_alpha_per_layer True

echo "=== 完成 ==="
