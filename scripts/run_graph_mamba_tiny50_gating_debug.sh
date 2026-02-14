#!/usr/bin/env bash
# 在 tiny50 上跑「带门控融合」的 EX 配置，用于 debug（少量 epoch）。
# 需要已安装 mamba-ssm 的 graph-mamba 环境。
# 用法: 在项目根目录执行  bash scripts/run_graph_mamba_tiny50_gating_debug.sh
# 可选:  export DATADIR_TINY50=/path/to/synthesis_data_tiny_50
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 默认使用本机 gnn_project_local 下的 tiny50；可覆盖 DATADIR_TINY50 或由 config 提供
DATADIR_TINY50="${DATADIR_TINY50:-/home/kfchen/gnn_project_local/synthesis_data_tiny_50}"
if [[ ! -d "$DATADIR_TINY50" ]]; then
  DATADIR_TINY50=$(python -c "import config; print(config.TRAIN_DATA_DIR_TINY_50)" 2>/dev/null || true)
fi
if [[ ! -d "$DATADIR_TINY50" ]]; then
  echo "Error: tiny50 数据目录不存在: ${DATADIR_TINY50}"
  echo "  请设置 GNN_DATA_ROOT 或 DATADIR_TINY50 指向含 synthesis_data_tiny_50 的目录。"
  exit 1
fi

echo "=== tiny50 门控融合 debug（3 epoch）==="
echo "  数据: ${DATADIR_TINY50}"
echo "  配置: EX + gt.fusion=conflict_aware, gt.layers=10, optim.max_epoch=3"
echo ""

conda run -n graph-mamba python scripts/run_graph_mamba.py \
  --data_dir "${DATADIR_TINY50}" \
  --wandb False \
  --name_tag tiny50_gating_debug \
  --repeat 1 \
  --override gt.layers 10 \
  --override gt.fusion conflict_aware \
  --override gt.fusion_beta 1.0 \
  --override optim.max_epoch 3

echo "=== 完成 ==="
