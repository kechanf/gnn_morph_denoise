#!/usr/bin/env bash

set -euo pipefail

# 两阶段完整数据集实验脚本（与 tiny50 流程一致，仅数据与配置不同）：
# 1) 在完整 morphology 数据上训练 DiGress 生成模型并保存 checkpoint
# 2) 以该 checkpoint 的 GraphTransformer 作为编码器，在同一数据上训练节点分类器（orig_y）
# 使用 debug 配置（小模型，避免 OOM），两阶段共用同一 model 结构。

PROJECT_ROOT="/home/kfchen/gnn_project"
DIGRESS_DIR="${PROJECT_ROOT}/external/DiGress"
DATADIR_FULL="/data2/kfchen/tracing_ws/morphology_seg/synthesis_data"
CUDA_DEVICES="1"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${DIGRESS_DIR}"

echo "=== 阶段 1：完整数据上训练 DiGress (dataset=morphology) 生成模型 ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate digress

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python src/main.py \
  +experiment=debug \
  dataset=morphology \
  dataset.datadir="${DATADIR_FULL}" \
  general.name="morph_full_two_stage" \
  general.gpus=1 \
  general.wandb=disabled \
  hydra.job.chdir=false \
  train.n_epochs=50 \
  train.batch_size=1 \
  train.save_model=True

CKPT_DIR="${DIGRESS_DIR}/checkpoints/morph_full_two_stage"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "未找到生成模型 checkpoint 目录: ${CKPT_DIR}"
  exit 1
fi

CKPT_PATH=$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT_PATH}" ]]; then
  echo "生成模型 checkpoint 目录中没有 .ckpt 文件: ${CKPT_DIR}"
  exit 1
fi

echo "=== 阶段 2：完整数据上训练基于 DiGress(morphology) 编码器的节点分类器 ==="

# 阶段 2 必须与阶段 1 使用相同 experiment（model 结构一致）
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python src/train_node_classifier.py \
  +experiment=debug \
  dataset.datadir="${DATADIR_FULL}" \
  general.name="node_cls_morphology_full_two_stage" \
  general.gpus=1 \
  train.batch_size=1 \
  train.n_epochs=20 \
  +model.encoder_ckpt="${CKPT_PATH}"

echo "=== 两阶段完整数据实验完成 ==="
