#!/bin/bash
# tiny50 上快速 debug 训练（用于验证 GNN Priority BFS 等改动）
# 用法: cd /home/kfchen/gnn_project && bash scripts/run_tiny50_debug.sh
#
# 数据路径: 优先 DATADIR_TINY50 环境变量，否则用 config.TRAIN_DATA_DIR_TINY_50
# 若 tiny50 不存在，可先运行: python scripts/create_tiny_dataset.py

set -e
cd "$(dirname "$0")/.."

# 优先 DATADIR_TINY50，否则 config，最后尝试本地路径
DATADIR="${DATADIR_TINY50:-$(python -c "import config; print(config.TRAIN_DATA_DIR_TINY_50)" 2>/dev/null)}"
if [ -z "$DATADIR" ] || [ ! -d "$DATADIR" ]; then
  DATADIR="/home/kfchen/gnn_project_local/synthesis_data_tiny_50"
fi
if [ ! -d "$DATADIR" ]; then
  echo "Error: tiny50 数据目录不存在: $DATADIR"
  echo "  1) 设置 GNN_DATA_ROOT 指向数据根目录"
  echo "  2) 运行 python scripts/create_tiny_dataset.py 从 synthesis_data 生成 tiny50"
  echo "  或设置 DATADIR_TINY50=/path/to/synthesis_data_tiny_50"
  exit 1
fi

# 删除旧 processed 缓存，确保 dist_from_root / is_target_root 生效（LexSort 路径）
CACHE="${DATADIR}/processed/morphology_processed.pt"
if [ -f "$CACHE" ]; then
  echo "Removing old cache: $CACHE"
  rm -f "$CACHE"
fi

# 3 epoch 快速 debug，batch 8，Mamba_GNNPriorityBFS，本地 out_dir
conda run -n graph-mamba python scripts/run_graph_mamba.py \
  --data_dir "$DATADIR" \
  --wandb False \
  --name_tag tiny50_debug_gnn_priority \
  --repeat 1 \
  --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
  --override optim.max_epoch 3 \
  --override train.batch_size 8 \
  --override out_dir "${GRAPH_MAMBA_OUT_DIR:-/home/kfchen/gnn_project_local/graph_mamba_results}"

echo "Done. Check results in out_dir"
