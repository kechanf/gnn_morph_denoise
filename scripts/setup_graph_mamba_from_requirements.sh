#!/usr/bin/env bash
# 按 external/Graph-Mamba/requirements_conda.txt 尽量一致地配置 Graph-Mamba 环境。
# 用法: bash scripts/setup_graph_mamba_from_requirements.sh [环境名]
# 默认环境名: graph-mamba
# 若环境已存在会先删除再创建，以保持与 requirements 一致。

set -e
ENV_NAME="${1:-graph-mamba}"
REQ_FILE="$(dirname "$0")/../external/Graph-Mamba/requirements_conda.txt"

echo "=== 按 requirements_conda.txt 配置环境: $ENV_NAME ==="

# 1. 创建 Python 3.9 环境（与 requirements 一致）
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "移除已有环境: $ENV_NAME"
  conda env remove -n "$ENV_NAME" -y
fi
echo "创建环境: $ENV_NAME (python=3.9)"
conda create -n "$ENV_NAME" python=3.9 -y

# 2. MKL < 2024.1（避免 import torch 报 iJIT_NotifyEvent）
conda install -n "$ENV_NAME" "mkl<2024.1" -y

# 3. 用 pip 安装 PyTorch 2.0.0 + CUDA 11.7（与 requirements 一致；conda 常缺 cuda 11.7 依赖）
echo "pip 安装 PyTorch 2.0.0 cu117..."
conda run -n "$ENV_NAME" pip install --no-cache-dir torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
CUDA_VER=117

# 4. nvcc 11.7（供 mamba-ssm 编译）
echo "安装 cuda-nvcc 11.7..."
conda install -n "$ENV_NAME" -c nvidia cuda-nvcc=11.7 -y

# 5. 基础 conda 依赖
conda install -n "$ENV_NAME" -y \
  numpy \
  pyyaml \
  scipy \
  scikit-learn \
  networkx \
  tqdm

# 6. pip 安装 Graph-Mamba / PyG 相关（版本尽量对齐 requirements_conda.txt）
echo "pip 安装 PyG、ogb、yacs、tensorboardx、performer-pytorch、torchmetrics..."
conda run -n "$ENV_NAME" pip install --no-cache-dir \
  torch-geometric==2.0.4 \
  ogb \
  yacs \
  tensorboardx \
  performer-pytorch \
  torchmetrics

# PyG 扩展需与 PyTorch 2.0 + cu117 匹配
echo "安装 PyG 扩展 (pt20 cu117)..."
conda run -n "$ENV_NAME" pip install --no-cache-dir \
  torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# 7. mamba-ssm：先装构建依赖，再无构建隔离安装（使用环境内 PyTorch/cuda）
echo "安装 ninja、packaging，然后 mamba-ssm（--no-build-isolation）..."
conda run -n "$ENV_NAME" pip install --no-cache-dir ninja packaging
# 优先尝试与 requirements 一致的 1.0.1；若失败可改试 2.3.0
if conda run -n "$ENV_NAME" pip install --no-cache-dir mamba-ssm==1.0.1 --no-build-isolation 2>/dev/null; then
  echo "mamba-ssm 1.0.1 安装完成"
else
  echo "mamba-ssm 1.0.1 安装失败，尝试 2.3.0..."
  conda run -n "$ENV_NAME" pip install --no-cache-dir mamba-ssm --no-build-isolation || true
fi

echo "=== 环境 $ENV_NAME 配置完成 ==="
echo "验证: conda activate $ENV_NAME && python -c \"import torch; import torch_geometric; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)\""
echo "若 mamba-ssm 仍报 GLIBC：见 docs/MAMBA_SSM_INSTALL.md（可用 --no-mamba 跑 GatedGCN-only）"
