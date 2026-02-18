#!/usr/bin/env bash
# 扫描门控参数 (fusion_beta, fusion_gate_init_zero) 的 6 组实验。
# beta ∈ {0.1, 1.0, 10.0} × gate_init_zero ∈ {True, False}
# 其余配置均与 Graph-Mamba BFS baseline 对齐：
#   - 使用 --baseline bfs（10+10 + Mamba_GNNPriorityBFS）
#   - 数据目录 / 超参 等来自 config.py / morphology-node-EX.yaml
#
# 用法：
#   cd /path/to/gnn_project
#   bash scripts/run_graph_mamba_gating_sweep.sh
#
# 依赖：
#   - 已存在 Conda 环境 graph-mamba（或按需修改 ENV_NAME）
#   - external/Graph-Mamba 及其依赖已安装

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-graph-mamba}"

BETAS=(0.1 1.0 10.0)
# init_flag = True 表示 gate_init_zero=True（等权起步），False 表示随机起步
INIT_FLAGS=(True False)

echo "=== 门控参数扫描：beta ∈ {0.1, 1.0, 10.0}, gate_init_zero ∈ {True, False} ==="
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  Conda 环境: ${ENV_NAME}"
echo ""

for beta in "${BETAS[@]}"; do
  for init in "${INIT_FLAGS[@]}"; do
    if [[ "${init}" == "True" ]]; then
      init_tag="equal"
    else
      init_tag="random"
    fi

    name_tag="bfs_gating_beta${beta}_init_${init_tag}"

    echo "----------------------------------------------------------------"
    echo "运行实验: beta=${beta}, fusion_gate_init_zero=${init} (${name_tag})"
    echo "----------------------------------------------------------------"

    conda run -n "${ENV_NAME}" python scripts/run_graph_mamba.py \
      --wandb False \
      --baseline bfs \
      --repeat 1 \
      --name_tag "${name_tag}" \
      --override gt.fusion conflict_aware \
      --override gt.fusion_beta "${beta}" \
      --override gt.fusion_gate_init_zero "${init}"

    echo ""
  done
done

echo "=== 所有 6 组门控实验已运行完毕（如无中途报错）==="

