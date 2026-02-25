#!/usr/bin/env bash
#
# 依次训练五种 GNN 形态学节点分类模型（与 docs/GNN_DEVELOPMENT_SUMMARY.md 一致）。
# 需在 gnn_project 根目录执行，或通过 --project_root 指定根目录。
#
# 用法:
#   ./scripts/train_all_five_models.sh
#   ./scripts/train_all_five_models.sh --data_dir /path/to/synthesis_data
#   ./scripts/train_all_five_models.sh --max_epoch 5   # 快速试跑
#   ./scripts/train_all_five_models.sh --log run.log --continue
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认数据目录：优先环境变量 GNN_DATA_ROOT，否则用 config 中的 DATA_ROOT
DATA_ROOT="${GNN_DATA_ROOT:-}"
if [[ -z "$DATA_ROOT" ]]; then
    DATA_ROOT="$(python -c "import config; print(config.DATA_ROOT)" 2>/dev/null || echo "/path/to/data_root")"
fi
DATA_DIR="${DATA_DIR:-$DATA_ROOT/synthesis_data}"
LOG_FILE=""
CONTINUE_ON_ERROR=false
MAX_EPOCH=""
DRY_RUN=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2"; shift 2 ;;
        --log)        LOG_FILE="$2"; shift 2 ;;
        --continue)   CONTINUE_ON_ERROR=true; shift ;;
        --max_epoch)  MAX_EPOCH="$2"; shift 2 ;;
        --dry_run)    DRY_RUN=true; shift ;;
        --project_root) PROJECT_ROOT="$2"; cd "$PROJECT_ROOT"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--data_dir DIR] [--log LOG] [--continue] [--max_epoch N] [--dry_run]"; exit 1 ;;
    esac
done

export GNN_DATA_ROOT="$DATA_ROOT"

run_one() {
    local name="$1"
    shift
    echo ""
    echo "=============================================="
    echo "  [$name]"
    echo "=============================================="
    if [[ -n "$LOG_FILE" ]]; then
        echo "[$(date -Iseconds)] $name START" >> "$LOG_FILE"
    fi
    local cmd=("python" "scripts/run_graph_mamba.py" "--data_dir" "$DATA_DIR" "--wandb" "False" "$@")
    if [[ -n "$MAX_EPOCH" ]]; then
        cmd+=(--override "optim.max_epoch" "$MAX_EPOCH")
    fi
    if $DRY_RUN; then
        echo "DRY RUN: ${cmd[*]}"
        return 0
    fi
    if [[ -n "$LOG_FILE" ]]; then
        set +e
        "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
        rc=${PIPESTATUS[0]}
        set -e
        if [[ $rc -eq 0 ]]; then
            echo "[$(date -Iseconds)] $name OK" >> "$LOG_FILE"
        else
            echo "[$(date -Iseconds)] $name FAIL (exit $rc)" >> "$LOG_FILE"
        fi
        return $rc
    else
        "${cmd[@]}" || return 1
    fi
}

fail() {
    echo "ERROR: $1 失败。退出码: $2"
    if ! $CONTINUE_ON_ERROR; then
        exit "$2"
    fi
}

echo "数据目录: $DATA_DIR"
echo "项目根目录: $PROJECT_ROOT"
if [[ -n "$MAX_EPOCH" ]]; then echo "覆盖 max_epoch: $MAX_EPOCH"; fi
if [[ -n "$LOG_FILE" ]]; then echo "日志文件: $LOG_FILE"; fi

# 1. 纯 GNN Baseline（20 层 GatedGCN）
run_one "1/5 纯 GNN Baseline (20_aligned)" --baseline 20_aligned || fail "纯 GNN Baseline" $?

# 2. Mamba Baseline（10+10）
run_one "2/5 Mamba Baseline (10_10)" --baseline 10_10 || fail "Mamba Baseline" $?

# 3. BFS 优化（Mamba_GNNPriorityBFS）
run_one "3/5 BFS 优化 (bfs)" --baseline bfs || fail "BFS 优化" $?

# 4. 门控优化（Conflict-Aware Gating，Baseline A + conflict_aware）
run_one "4/5 门控优化 (conflict_aware)" \
    --name_tag full_gating_run \
    --repeat 1 \
    --override gt.layers 10 \
    --override gt.fusion conflict_aware \
    --override gt.fusion_beta 1.0 \
    || fail "门控优化" $?

# 5. BFS + 门控（组合）
run_one "5/5 BFS+门控 (combo)" \
    --name_tag baseline_combo_bfs_gating \
    --override gt.layer_type "CustomGatedGCN+Mamba_GNNPriorityBFS" \
    --override gt.layers 10 \
    --override gt.fusion conflict_aware \
    --override gt.fusion_beta 1.0 \
    || fail "BFS+门控" $?

echo ""
echo "=============================================="
echo "  全部 5 个模型训练完成"
echo "=============================================="
