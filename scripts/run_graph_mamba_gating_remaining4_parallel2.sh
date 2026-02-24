#!/usr/bin/env bash
# 跑剩余 4 组门控实验：每次并行 2 组。
#
# 默认实验组合（两两并行）：
#   1) beta=1.0,  gate_init_zero=True
#   2) beta=1.0,  gate_init_zero=False
#   3) beta=10.0, gate_init_zero=True
#   4) beta=10.0, gate_init_zero=False
#
# 说明：
# - 采用独立 name_tag 后缀 _p2，避免和之前中断/已跑目录冲突。
# - 其余配置与 baseline bfs 对齐（--baseline bfs + 同数据目录）。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-graph-mamba}"
DATA_DIR="${DATA_DIR:-/home/kfchen/gnn_project_local/synthesis_data}"
RUN_SUFFIX="${RUN_SUFFIX:-p2}"

BETAS=("1.0" "1.0" "10.0" "10.0")
INITS=("True" "False" "True" "False")
INIT_TAGS=("equal" "random" "equal" "random")

run_one() {
  local beta="$1"
  local init="$2"
  local init_tag="$3"
  local name_tag="bfs_gating_beta${beta}_init_${init_tag}_${RUN_SUFFIX}"

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

echo "=== 剩余4组门控实验（并行2组）==="
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "ENV_NAME:     ${ENV_NAME}"
echo "DATA_DIR:     ${DATA_DIR}"
echo "RUN_SUFFIX:   ${RUN_SUFFIX}"
echo ""

for i in 0 2; do
  echo "------------------------------------------------------------"
  echo "并行批次 $((i/2+1)):"
  echo "  A: beta=${BETAS[$i]}, init=${INITS[$i]}"
  echo "  B: beta=${BETAS[$((i+1))]}, init=${INITS[$((i+1))]}"
  echo "------------------------------------------------------------"

  run_one "${BETAS[$i]}" "${INITS[$i]}" "${INIT_TAGS[$i]}" &
  pid_a=$!
  run_one "${BETAS[$((i+1))]}" "${INITS[$((i+1))]}" "${INIT_TAGS[$((i+1))]}" &
  pid_b=$!

  status_a=0
  status_b=0
  wait "${pid_a}" || status_a=$?
  wait "${pid_b}" || status_b=$?

  if [[ ${status_a} -ne 0 || ${status_b} -ne 0 ]]; then
    echo "[error] 批次 $((i/2+1)) 失败: status_a=${status_a}, status_b=${status_b}"
    exit 1
  fi
done

echo "=== 剩余4组并行实验已全部完成 ==="

