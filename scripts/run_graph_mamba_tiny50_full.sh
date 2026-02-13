#!/usr/bin/env bash
# 在 tiny50 上运行最完整的 Graph-Mamba（EX 配置：Mamba + GatedGCN + edge/node encoder + LapPE）
# 需要已安装 mamba-ssm 的 graph-mamba 环境；若出现 torch_sparse/PyTorch ABI 报错，需重装
#   torch_scatter/torch_sparse 以匹配当前 PyTorch 版本。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 与 run_two_stage_tiny50.sh 一致的数据路径
DATADIR_TINY50="${DATADIR_TINY50:-/data2/kfchen/tracing_ws/morphology_seg/synthesis_data_tiny_50}"
if [[ ! -d "$DATADIR_TINY50" ]]; then
  # 若未设置且默认不存在，尝试从 config 读取
  DATADIR_TINY50=$(python -c "import config; print(config.TRAIN_DATA_DIR_TINY_50)" 2>/dev/null || true)
fi
if [[ ! -d "$DATADIR_TINY50" ]]; then
  echo "Error: tiny50 数据目录不存在: ${DATADIR_TINY50}"
  exit 1
fi

echo "=== 在 tiny50 上运行最完整 Graph-Mamba（EX，含 Mamba-SSM）==="
echo "  数据目录: ${DATADIR_TINY50}"
echo "  配置: morphology-node-EX.yaml（GatedGCN + Mamba + edge/node encoder + LapPE）"
echo ""

conda run -n graph-mamba python scripts/run_graph_mamba.py \
  --data_dir "${DATADIR_TINY50}" \
  --wandb False \
  --name_tag tiny50_full \
  --repeat 1

echo "=== 完成 ==="
