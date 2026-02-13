#!/usr/bin/env bash
# 在 graph-mamba 环境中安装 mamba-ssm（解决 PyTorch cu128 与 nvcc 11.7 不匹配）
# 用法: bash scripts/install_mamba_ssm_graph_mamba.sh [环境名] [可选: 本地 wheel 路径]
set -e

ENV_NAME="${1:-graph-mamba}"
LOCAL_WHEEL="${2:-}"

echo "[install_mamba_ssm] 环境: $ENV_NAME"

if [[ -n "$LOCAL_WHEEL" ]]; then
  echo "[install_mamba_ssm] 从本地 wheel 安装: $LOCAL_WHEEL"
  conda run -n "$ENV_NAME" pip install --no-deps "$LOCAL_WHEEL"
  conda run -n "$ENV_NAME" python -c "import mamba_ssm; print('mamba_ssm 导入成功')"
  echo "[install_mamba_ssm] 完成"
  exit 0
fi

# 检查当前 nvcc 与 torch 的 CUDA 是否一致
TORCH_CUDA=$(conda run -n "$ENV_NAME" python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)
NVIDIA_NVCC_VER=$(conda run -n "$ENV_NAME" bash -c "nvcc -V 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1" 2>/dev/null || true)

echo "[install_mamba_ssm] PyTorch CUDA: ${TORCH_CUDA:-未检测到}, nvcc: ${NVIDIA_NVCC_VER:-未检测到}"

# 若 PyTorch 为 cu12.x 而 nvcc 为 11.x，则安装 cuda-nvcc 12.x
if [[ "$TORCH_CUDA" == 12.* && "$NVIDIA_NVCC_VER" == 11.* ]]; then
  echo "[install_mamba_ssm] 检测到 PyTorch cu12 与 nvcc 11 不匹配，尝试安装 cuda-nvcc 12.8..."
  conda install -n "$ENV_NAME" -y nvidia/label/cuda-12.8.0::cuda-nvcc || true
  NVIDIA_NVCC_VER=$(conda run -n "$ENV_NAME" bash -c "nvcc -V 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1" 2>/dev/null || true)
  echo "[install_mamba_ssm] 当前 nvcc: $NVIDIA_NVCC_VER"
fi

conda run -n "$ENV_NAME" pip install ninja packaging -q
echo "[install_mamba_ssm] 使用 MAMBA_FORCE_BUILD=TRUE 从源码构建 mamba-ssm 2.3.0..."
conda run -n "$ENV_NAME" bash -c "export MAMBA_FORCE_BUILD=TRUE && pip install mamba-ssm==2.3.0 --no-build-isolation"

# 若缺少 libopenblas.so.0（scipy/transformers 链会用到），安装并确保 env lib 在加载路径中
if ! conda run -n "$ENV_NAME" python -c "import mamba_ssm" 2>/dev/null; then
  echo "[install_mamba_ssm] 首次导入失败，尝试安装 libopenblas 并重试..."
  conda install -n "$ENV_NAME" -y libopenblas -c conda-forge 2>/dev/null || true
  # 确保环境 lib 在 LD_LIBRARY_PATH 中（conda run 可能未包含）
  conda run -n "$ENV_NAME" bash -c "export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}; python -c \"import mamba_ssm; print('mamba_ssm 导入成功')\"" || true
fi
conda run -n "$ENV_NAME" bash -c "export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}; python -c \"import mamba_ssm; print('mamba_ssm 导入成功')\""
echo "[install_mamba_ssm] 完成"
