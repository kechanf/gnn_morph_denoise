#!/usr/bin/env bash
# DiGress 基线部署脚本
# 用法: 在 gnn_project 根目录执行 bash scripts/install_digress.sh
# 若 clone 超时，可先手动: git clone https://github.com/cvignac/DiGress.git external/DiGress

set -e
GNN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIGRESS_DIR="$GNN_ROOT/external/DiGress"
ENV_NAME="${DIGRESS_CONDA_ENV:-digress}"

echo "[DiGress] 项目根目录: $GNN_ROOT"

# 1. 克隆 DiGress（若不存在）
if [ ! -d "$DIGRESS_DIR/src" ]; then
  echo "[DiGress] 正在克隆仓库到 $DIGRESS_DIR ..."
  mkdir -p "$(dirname "$DIGRESS_DIR")"
  git clone --depth 1 https://github.com/cvignac/DiGress.git "$DIGRESS_DIR" || {
    echo "[DiGress] 克隆失败（可能网络超时）。请手动执行："
    echo "  git clone https://github.com/cvignac/DiGress.git $DIGRESS_DIR"
    echo " 然后重新运行本脚本。"
    exit 1
  }
else
  echo "[DiGress] 已存在 $DIGRESS_DIR，跳过克隆。"
fi

# 2. 创建 conda 环境（若不存在，使用 classic 求解器避免 libmamba 报错）
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[DiGress] 创建 conda 环境: $ENV_NAME (python=3.9, rdkit) [--solver classic]"
  conda create -c conda-forge -n "$ENV_NAME" rdkit=2023.03.2 python=3.9 -y --solver classic
else
  echo "[DiGress] 环境 $ENV_NAME 已存在，跳过创建。"
fi

# 3. 在 digress 环境中安装依赖（使用 conda run 避免 source 问题）
echo "[DiGress] 安装 graph-tool、PyTorch、requirements 与 DiGress 包..."

conda run -n "$ENV_NAME" conda install -c conda-forge graph-tool=2.45 -y --solver classic || true

# PyTorch（CUDA 11.8；若无 GPU 可改用 cpu 版本）
conda run -n "$ENV_NAME" pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 || \
  conda run -n "$ENV_NAME" pip install torch==2.0.1 torchvision torchaudio || true

conda run -n "$ENV_NAME" pip install -r "$DIGRESS_DIR/requirements.txt"
conda run -n "$ENV_NAME" pip install -e "$DIGRESS_DIR"

# 4. 编译 orca（用于图统计评估）
ORCA_DIR="$DIGRESS_DIR/src/analysis/orca"
if [ -f "$ORCA_DIR/orca.cpp" ]; then
  echo "[DiGress] 编译 orca..."
  (cd "$ORCA_DIR" && g++ -O2 -std=c++11 -o orca orca.cpp) || echo "[DiGress] orca 编译失败，评估可能受限。"
else
  echo "[DiGress] 未找到 orca 源码，跳过编译。"
fi

# 5. 快速自检
echo "[DiGress] 自检..."
conda run -n "$ENV_NAME" python3 -c "from rdkit import Chem; import torch; from torch_geometric.data import Data; print('rdkit, PyTorch, PyG OK')" || true
conda run -n "$ENV_NAME" python3 -c "import graph_tool as gt" 2>/dev/null || echo "[DiGress] graph_tool 未安装或不可用（Mac 上可能与 PyG 冲突），可忽略。"

echo "[DiGress] 部署完成。激活环境: conda activate $ENV_NAME"
echo "[DiGress] 运行调试: cd $DIGRESS_DIR && python3 main.py +experiment=debug.yaml"
echo "[DiGress] 详见 docs/DIGRESS_BASELINE.md"
