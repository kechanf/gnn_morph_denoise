#!/usr/bin/env bash
# 监控第2组实验（beta=0.1, init_random）完成状态。
# 检测到 "Task done" 后：
# 1) 打印提示
# 2) 尝试桌面通知（notify-send 可用时）
# 3) 不再响铃（仅终端+桌面通知）

set -euo pipefail

LOG_FILE="${1:-/PBshare/SEU-ALLEN/Users/KaifengChen/gnn_data/morphology_seg/graph_mamba_results/morphology-node-EX-bfs_gating_beta0.1_init_random/42/logging.log}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"

echo "[watch] 监控日志: ${LOG_FILE}"
echo "[watch] 轮询间隔: ${CHECK_INTERVAL_SEC}s"

while [[ ! -f "${LOG_FILE}" ]]; do
  sleep "${CHECK_INTERVAL_SEC}"
done

while true; do
  if python3 - "${LOG_FILE}" <<'PY'
import pathlib
import sys

log_file = pathlib.Path(sys.argv[1])
text = log_file.read_text(errors="ignore")
raise SystemExit(0 if "Task done" in text else 1)
PY
  then
    msg="第2组实验已完成: bfs_gating_beta0.1_init_random"
    echo "[watch] ${msg}"
    date
    if command -v notify-send >/dev/null 2>&1; then
      notify-send "Graph-Mamba Sweep" "${msg}"
    fi
    break
  fi
  sleep "${CHECK_INTERVAL_SEC}"
done

