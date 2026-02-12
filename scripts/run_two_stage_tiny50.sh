#!/usr/bin/env bash

set -euo pipefail

# 两阶段 tiny50 实验脚本：
# 1) 在 tiny50 上训练 DiGress 生成/去噪模型并保存 checkpoint
# 2) 以该 checkpoint 的 GraphTransformer 作为编码器，在同一 tiny50 上训练节点分类器（orig_y）

PROJECT_ROOT="/home/kfchen/gnn_project"
DIGRESS_DIR="${PROJECT_ROOT}/external/DiGress"
DATADIR_TINY50="/data2/kfchen/tracing_ws/morphology_seg/synthesis_data_tiny_50"
CUDA_DEVICES="1"   # 使用 GPU2

cd "${DIGRESS_DIR}"

echo "=== 阶段 1：tiny50 上训练 DiGress (dataset=morphology) 生成模型 ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate digress

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python src/main.py \
  +experiment=debug \
  dataset=morphology \
  dataset.datadir="${DATADIR_TINY50}" \
  general.name="morph_tiny50_two_stage" \
  general.gpus=1 \
  general.wandb=disabled \
  hydra.job.chdir=false \
  train.n_epochs=3 \
  train.batch_size=2 \
  train.save_model=True

CKPT_DIR="${DIGRESS_DIR}/checkpoints/morph_tiny50_two_stage"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "未找到生成模型 checkpoint 目录: ${CKPT_DIR}"
  exit 1
fi

CKPT_PATH=$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT_PATH}" ]]; then
  echo "生成模型 checkpoint 目录中没有 .ckpt 文件: ${CKPT_DIR}"
  exit 1
fi

echo "=== 阶段 2：tiny50 上训练基于 DiGress(morphology) 编码器的节点分类器 ==="

# 阶段 2 必须与阶段 1 使用相同 model 结构（阶段 1 用了 +experiment=debug）
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python src/train_node_classifier.py \
  +experiment=debug \
  dataset.datadir="${DATADIR_TINY50}" \
  general.name="node_cls_morphology_tiny50_two_stage" \
  general.gpus=1 \
  train.batch_size=2 \
  train.n_epochs=10 \
  +model.encoder_ckpt="${CKPT_PATH}"

echo "=== 两阶段 tiny50 实验完成 ==="

