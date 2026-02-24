#!/usr/bin/env bash
# 仅跑剩余 2 组：beta=10.0（equal + random），并行执行。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
ENV_NAME="${ENV_NAME:-graph-mamba}"
DATA_DIR="${DATA_DIR:-/home/kfchen/gnn_project_local/synthesis_data}"

run_one() {
  local name_tag="$1"
  local beta="$2"
  local init="$3"
  echo "[start] ${name_tag}"
  conda run -n "${ENV_NAME}" python scripts/run_graph_mamba.py \
    --data_dir "${DATA_DIR}" \
    --wandb False \
    --baseline bfs \
    --repeat 1 \
    --name_tag "${name_tag}" \
    --override gt.fusion conflict_aware \
    --override gt.fusion_beta "${beta}" \
    --override gt.fusion_gate_init_zero "${init}"
  echo "[done ] ${name_tag}"
}

echo "=== 并行启动: beta=10.0 init_equal_p2, beta=10.0 init_random_p2 ==="
run_one "bfs_gating_beta10.0_init_equal_p2" "10.0" "True" &
pid1=$!
run_one "bfs_gating_beta10.0_init_random_p2" "10.0" "False" &
pid2=$!
wait $pid1 || exit 1
wait $pid2 || exit 1
echo "=== 两组均已完成 ==="
