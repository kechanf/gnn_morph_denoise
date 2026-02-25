#!/usr/bin/env bash
# 扫描 BFS + 门控 在不同 tau 下的表现：
#   tau ∈ {0.3, 0.4, 0.6, 0.7}
#   - baseline: bfs  (10+10 + Mamba_GNNPriorityBFS)
#   - fusion: conflict_aware
#   - depth-aware beta: True
#   - fusion_learnable_beta: False  （固定深度先验）
#   - fusion_ortho_lambda: 0.0      （不加特征正交）
#   - fusion_log_alpha_per_layer: True  （记录分层 alpha）
#
# 用法：
#   在项目根目录执行：
#     bash scripts/run_bfs_gating_tau_sweep.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-graph-mamba}"
DATA_DIR="${DATA_DIR:-/home/kfchen/gnn_project_local/synthesis_data}"

echo "=== BFS + 新门控 tau 扫描实验 ==="
echo "  DATA_DIR: ${DATA_DIR}"
echo "  ENV_NAME: ${ENV_NAME}"
echo ""

# 已经单独跑过 tau=0.3，此处只串行跑后面三组
TAUS=(0.4 0.6 0.7)

for tau in "${TAUS[@]}"; do
  name_tag="bfs_gating_depth_tau${tau}"
  echo "------------------------------------------------------------"
  echo "运行实验: tau=${tau}  (name_tag=${name_tag})"
  echo "------------------------------------------------------------"

  conda run -n "${ENV_NAME}" python scripts/run_graph_mamba.py \
    --data_dir "${DATA_DIR}" \
    --wandb False \
    --baseline bfs \
    --repeat 1 \
    --name_tag "${name_tag}" \
    --override gt.fusion conflict_aware \
    --override gt.fusion_log_alpha_per_layer True \
    --override gt.fusion_depth_aware_beta True \
    --override gt.fusion_learnable_beta False \
    --override gt.fusion_tau "${tau}"

  echo ""
done

echo "=== BFS + 门控 tau 扫描全部完成（如中途无报错） ==="

