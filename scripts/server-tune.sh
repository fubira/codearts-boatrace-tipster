#!/usr/bin/env bash
# server-tune.sh — Run Optuna hyperparameter tuning on a remote server via SSH
#
# Usage:
#   ./scripts/server-tune.sh --model boat1      # 二値分類モデルのOptuna（デフォルト: nohup）
#   ./scripts/server-tune.sh --model ranking    # ランキングモデルのOptuna（デフォルト）
#   ./scripts/server-tune.sh --watch            # ログ監視（tail -f、完了で自動終了）
#   ./scripts/server-tune.sh --status           # 簡易進捗確認
#   ./scripts/server-tune.sh --fetch            # 結果取得（ログをダウンロード）
#   ./scripts/server-tune.sh --foreground       # SSH接続維持モード
#   ./scripts/server-tune.sh --setup            # 初回セットアップ（uv + workspace + deps）
#
# Server directory layout:
#   $REMOTE_DIR/
#     ├── ml/               rsync from local (src, scripts, pyproject.toml)
#     │   └── .venv/        (uv sync)
#     └── data/
#         └── boatrace-tipster.db
#
# Prerequisites:
#   - SSH config for the server (e.g. Host one in ~/.ssh/config)
#   - Copy scripts/server-tune.conf.example to scripts/server-tune.conf

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Configuration ---
SERVER="one"
REMOTE_DIR="\$HOME/boatrace-tipster-tune"

CONF="$(dirname "$0")/server-tune.conf"
if [ -f "$CONF" ]; then
  # shellcheck source=/dev/null
  source "$CONF"
fi

# --- Defaults ---
TRIALS=100
FOLDS=4
FOLD_MONTHS=2
RELEVANCE=""
SEED=42
FROM_MODEL=""
NARROW=false
# n_jobs=2 + num_threads=4 = 8 threads total, matching i7-6700 (4 phys / 8 LCPU)
N_JOBS=2
NUM_THREADS=4
OBJECTIVE=""
FIX_THRESHOLDS=""

SETUP_ONLY=false
FOREGROUND=false
STATUS_ONLY=false
FETCH_ONLY=false
WATCH_ONLY=false
SKIP_SYNC=false

# --- Local paths ---
PROJECT_DIR="$(pwd)"
ML_DIR="${PROJECT_DIR}/ml"
LOG_DIR="logs/tune"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --server) SERVER="$2"; shift 2 ;;
    --setup) SETUP_ONLY=true; shift ;;
    --foreground) FOREGROUND=true; shift ;;
    --watch) WATCH_ONLY=true; shift ;;
    --status) STATUS_ONLY=true; shift ;;
    --fetch) FETCH_ONLY=true; shift ;;
    --skip-sync) SKIP_SYNC=true; shift ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --fold-months) FOLD_MONTHS="$2"; shift 2 ;;
    --relevance) RELEVANCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --from-model) FROM_MODEL="$2"; shift 2 ;;
    --narrow) NARROW=true; shift ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    --num-threads) NUM_THREADS="$2"; shift 2 ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
    --fix-thresholds) FIX_THRESHOLDS="$2"; shift 2 ;;
    --help)
      cat <<'HELP'
Usage: ./scripts/server-tune.sh [options]

Runs P2 strategy Optuna tuning (tune_p2.py) on the remote server.

Modes:
  (default)         サーバーでnohup実行（即座に返る）
  --watch           ログ監視（tail -f、完了で自動終了）
  --status          簡易進捗確認
  --fetch           結果取得（ログをダウンロード）
  --foreground      SSH接続維持モード
  --setup           初回セットアップ（uv + workspace + deps）

Optuna options:
  --trials N        trial数 (default: 100)
  --folds N         WF-CV fold数 (default: 4)
  --fold-months N   fold幅（月数、default: 2）
  --relevance R     relevance scheme: linear|top_heavy|podium
  --seed N          random seed (default: 42)
  --objective O     tune_p2 objective: growth | kelly (default: growth)
  --fix-thresholds  閾値固定でハイパラのみ探索 (e.g., "gap23=0.13,ev=0.0,top3_conc=0.7")
  --from-model D    既存モデルのHPを初期trialとして投入（カンマ区切り可能）
  --narrow          --from-model の最初のモデル周辺だけを探索（要 --from-model）
  --n-jobs N        並列 trial 数 (default: 2、サーバーでのみ許可)
  --num-threads N   trial あたりの LightGBM スレッド数 (default: 4 = 8thread/2jobs on i7-6700)

General:
  --skip-sync       コード・データ同期スキップ
  --server HOST     サーバーホスト名 (default: one)
HELP
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() { ssh -o ConnectTimeout=10 "$SERVER" "$@"; }

if [ "$NARROW" = true ] && [ -z "$FROM_MODEL" ]; then
  echo "ERROR: --narrow requires --from-model" >&2
  exit 1
fi

# --- Validate SSH ---
log "Connecting to ${SERVER}..."
if ! remote true 2>/dev/null; then
  log "ERROR: Cannot connect to ${SERVER}"
  exit 1
fi
REMOTE_HOSTNAME=$(remote hostname -s)
log "Connected to ${REMOTE_HOSTNAME}"

REMOTE_DIR_RESOLVED=$(remote "echo ${REMOTE_DIR}")

# Remote file paths
REMOTE_LOG_FILE="${REMOTE_DIR_RESOLVED}/tune-run.log"
REMOTE_PID_FILE="${REMOTE_DIR_RESOLVED}/tune-run.pid"
REMOTE_SCRIPT_FILE="${REMOTE_DIR_RESOLVED}/run-tune.sh"

# ============================================================
# Setup
# ============================================================
_setup() {
  log "=== Setup: installing uv and creating workspace ==="

  log "Installing uv (if needed)..."
  remote bash <<'SETUP'
set -euo pipefail
if command -v uv &>/dev/null; then
  echo "uv already installed: $(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  echo "uv installed: $($HOME/.local/bin/uv --version)"
fi
SETUP

  log "Creating workspace directories..."
  remote "mkdir -p ${REMOTE_DIR_RESOLVED}/ml ${REMOTE_DIR_RESOLVED}/data"

  _sync_code

  log "Installing Python dependencies..."
  remote bash <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd ${REMOTE_DIR_RESOLVED}/ml
uv sync --frozen 2>&1 | tail -5
EOF

  log "=== Setup complete ==="
}

# ============================================================
# Sync
# ============================================================
_sync_code() {
  log "Syncing ML code + models..."
  rsync -az --delete \
    --exclude='.venv/' \
    --exclude='cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    "${ML_DIR}/" "${SERVER}:${REMOTE_DIR_RESOLVED}/ml/"
}

_sync_data() {
  log "Syncing database..."
  rsync -az --info=progress2 \
    "data/boatrace-tipster.db" \
    "${SERVER}:${REMOTE_DIR_RESOLVED}/data/boatrace-tipster.db"
}

# ============================================================
# Status / Watch / Fetch
# ============================================================
_status() {
  log "Checking server status..."
  local pid
  pid=$(remote "cat ${REMOTE_PID_FILE} 2>/dev/null" || echo "")
  if [ -n "$pid" ] && remote "kill -0 $pid 2>/dev/null"; then
    log "Running (PID: $pid)"
    echo ""
    remote "tail -20 ${REMOTE_LOG_FILE} 2>/dev/null" || echo "(no log yet)"
  elif remote "grep -q '=== Done ===' ${REMOTE_LOG_FILE} 2>/dev/null"; then
    log "Completed"
    echo ""
    remote "tail -10 ${REMOTE_LOG_FILE}"
  else
    log "Not running"
    if remote "test -f ${REMOTE_LOG_FILE}" 2>/dev/null; then
      echo ""
      remote "tail -10 ${REMOTE_LOG_FILE}"
    fi
  fi
}

_watch() {
  log "Watching server log (auto-exits on completion)..."
  remote -t "while [ ! -f ${REMOTE_LOG_FILE} ]; do sleep 1; done; tail -f ${REMOTE_LOG_FILE} | sed -u '/=== Done ===/q'"
  log "Completed. Run --fetch to download results."
}

_fetch() {
  mkdir -p "$LOG_DIR"
  local ts
  ts=$(date +%Y-%m-%d_%H%M)
  local log_file="${LOG_DIR}/${ts}_server-tune.log"
  local trials_json="${LOG_DIR}/${ts}_server-tune.trials.json"

  if ! remote "grep -q '=== Done ===' ${REMOTE_LOG_FILE} 2>/dev/null"; then
    log "WARNING: run may not be complete yet"
  fi

  rsync -az "${SERVER}:${REMOTE_LOG_FILE}" "${log_file}" 2>/dev/null && \
    log "Downloaded log → ${log_file}" || \
    log "WARNING: failed to download log"

  # Also fetch trials.json (for p2 runs with best_iter tracking)
  local remote_trials="${REMOTE_DIR_RESOLVED}/ml/models/tune_result/trials.json"
  if remote "test -f ${remote_trials}"; then
    rsync -az "${SERVER}:${remote_trials}" "${trials_json}" 2>/dev/null && \
      log "Downloaded trials.json → ${trials_json}" || \
      log "WARNING: failed to download trials.json"
  fi

  # Show results summary from log
  echo ""
  echo "=== Results ==="
  remote "sed -n '/^=\\{60\\}/,/^Total time/p' ${REMOTE_LOG_FILE} 2>/dev/null" || true
}

# ============================================================
# Run (detach / foreground)
# ============================================================
_build_cmd() {
  local cmd
  cmd="uv run --directory ml python -m scripts.tune_p2"
  cmd+=" --trials ${TRIALS}"
  cmd+=" --n-folds ${FOLDS}"
  cmd+=" --fold-months ${FOLD_MONTHS}"
  cmd+=" --seed ${SEED}"
  if [ -n "$OBJECTIVE" ]; then
    cmd+=" --objective ${OBJECTIVE}"
  fi
  if [ -n "$RELEVANCE" ]; then
    cmd+=" --relevance ${RELEVANCE}"
  fi
  if [ -n "$FIX_THRESHOLDS" ]; then
    cmd+=" --fix-thresholds '${FIX_THRESHOLDS}'"
  fi
  if [ -n "$FROM_MODEL" ]; then
    cmd+=" --from-model '${FROM_MODEL}'"
  fi
  if [ "$NARROW" = true ]; then
    cmd+=" --narrow"
  fi
  if [ "$N_JOBS" -gt 1 ]; then
    cmd+=" --n-jobs ${N_JOBS}"
  fi
  if [ "$NUM_THREADS" -gt 0 ]; then
    cmd+=" --num-threads ${NUM_THREADS}"
  fi
  echo "$cmd"
}

_detach_run() {
  local tune_cmd
  tune_cmd=$(_build_cmd)

  # Archive previous log using its mtime as YYYY-MM-DD_HHMM
  remote bash <<EOF
mkdir -p ${REMOTE_DIR_RESOLVED}/tune-logs
if [ -s "${REMOTE_LOG_FILE}" ]; then
  prev_ts=\$(date -r "${REMOTE_LOG_FILE}" '+%Y-%m-%d_%H%M' 2>/dev/null || echo "unknown")
  cp "${REMOTE_LOG_FILE}" "${REMOTE_DIR_RESOLVED}/tune-logs/\${prev_ts}_server-tune.log" 2>/dev/null || true
fi
EOF

  # Create run script
  remote bash <<EOF
cat > ${REMOTE_SCRIPT_FILE} << 'INNERSCRIPT'
#!/bin/bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export BOATRACE_TUNE_PARALLEL=1
cd ${REMOTE_DIR_RESOLVED}
LOG="${REMOTE_LOG_FILE}"
trap 'rm -f ${REMOTE_PID_FILE}' EXIT
echo "Started at \$(date '+%Y-%m-%d %H:%M:%S %Z')" > "\$LOG"
echo "Command: ${tune_cmd}" >> "\$LOG"
echo "" >> "\$LOG"
${tune_cmd} >> "\$LOG" 2>&1
echo "" >> "\$LOG"
echo "=== Done ===" >> "\$LOG"
date '+%Y-%m-%d %H:%M:%S %Z' >> "\$LOG"
INNERSCRIPT
chmod +x ${REMOTE_SCRIPT_FILE}
EOF

  remote "nohup ${REMOTE_SCRIPT_FILE} > /dev/null 2>&1 & echo \$! > ${REMOTE_PID_FILE}"
  local pid
  pid=$(remote "cat ${REMOTE_PID_FILE}")
  log "Started on ${REMOTE_HOSTNAME} (PID: ${pid})"
  log "  trials=${TRIALS} folds=${FOLDS}"
  log ""
  log "Check progress:  ./scripts/server-tune.sh --status"
  log "Watch log:       ./scripts/server-tune.sh --watch"
  log "Fetch results:   ./scripts/server-tune.sh --fetch"
}

_foreground_run() {
  local tune_cmd
  tune_cmd=$(_build_cmd)

  log "Running in foreground (${TRIALS} trials)..."
  remote bash <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export BOATRACE_TUNE_PARALLEL=1
cd ${REMOTE_DIR_RESOLVED}
${tune_cmd}
EOF
}

# ============================================================
# Main
# ============================================================
if [ "$SETUP_ONLY" = true ]; then
  _setup
  exit 0
fi

if [ "$STATUS_ONLY" = true ]; then
  _status
  exit 0
fi

if [ "$WATCH_ONLY" = true ]; then
  _watch
  exit 0
fi

if [ "$FETCH_ONLY" = true ]; then
  _fetch
  exit 0
fi

# --- Sync ---
if [ "$SKIP_SYNC" = false ]; then
  _sync_code
  _sync_data
fi

# --- Run ---
if [ "$FOREGROUND" = true ]; then
  _foreground_run
else
  _detach_run
fi
