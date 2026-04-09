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
MODEL="trifecta"  # trifecta | ranking | boat1
TRIALS=100
FOLDS=4
FOLD_MONTHS=2
RELEVANCE=""
SEED=42
TRAIN_START=""
WARM_START=false
OBJECTIVE=""
BETA=""
WITH_R2=false

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
    --model) MODEL="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --fold-months) FOLD_MONTHS="$2"; shift 2 ;;
    --relevance) RELEVANCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-start) TRAIN_START="$2"; shift 2 ;;
    --warm-start) WARM_START=true; shift ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --with-r2) WITH_R2=true; shift ;;
    --help)
      cat <<'HELP'
Usage: ./scripts/server-tune.sh [options]

Modes:
  (default)         サーバーでnohup実行（即座に返る）
  --watch           ログ監視（tail -f、完了で自動終了）
  --status          簡易進捗確認
  --fetch           結果取得（ログをダウンロード）
  --foreground      SSH接続維持モード
  --setup           初回セットアップ（uv + workspace + deps）

Optuna options:
  --model M         モデル: trifecta | ranking | boat1 (default: trifecta)
  --trials N        trial数 (default: 100)
  --folds N         WF-CV fold数 (default: 4)
  --fold-months N   fold幅（月数、default: 2）
  --relevance R     relevance scheme (default: top_heavy, ranking only)
  --seed N          random seed (default: 42)
  --train-start D   学習開始日 (default: all)
  --warm-start      現行model_metaのパラメータで初期化
  --with-r2         rank-2フォールバック有効化（trifecta, default: 無効）
  --objective O     boat1 objective: ev_roi | upset_fbeta (default: ev_roi)
  --beta F          F-beta for upset_fbeta (default: 1.5)

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
  log "Syncing ML code..."
  rsync -az --delete \
    --exclude='.venv/' \
    --exclude='models/' \
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

  if ! remote "grep -q '=== Done ===' ${REMOTE_LOG_FILE} 2>/dev/null"; then
    log "WARNING: run may not be complete yet"
  fi

  rsync -az "${SERVER}:${REMOTE_LOG_FILE}" "${LOG_DIR}/${ts}_server-tune.log" 2>/dev/null && \
    log "Downloaded log → ${LOG_DIR}/${ts}_server-tune.log" || \
    log "WARNING: failed to download log"

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
  case "$MODEL" in
    ranking)
      cmd="uv run --directory ml python -m scripts.train_eval"
      cmd+=" --mode optuna"
      cmd+=" --relevance ${RELEVANCE:-top_heavy}"
      ;;
    boat1)
      cmd="uv run --directory ml python -m scripts.train_boat1_binary"
      cmd+=" --mode optuna"
      if [ -n "$OBJECTIVE" ]; then
        cmd+=" --objective ${OBJECTIVE}"
      fi
      if [ -n "$BETA" ]; then
        cmd+=" --beta ${BETA}"
      fi
      ;;
    trifecta)
      cmd="uv run --directory ml python -m scripts.tune_trifecta"
      cmd+=" --trials ${TRIALS}"
      cmd+=" --n-folds ${FOLDS}"
      cmd+=" --fold-months ${FOLD_MONTHS}"
      cmd+=" --seed ${SEED}"
      if [ "$WARM_START" = true ]; then
        cmd+=" --warm-start"
      fi
      if [ -n "$OBJECTIVE" ]; then
        cmd+=" --objective ${OBJECTIVE}"
      fi
      if [ -n "$RELEVANCE" ]; then
        cmd+=" --relevance ${RELEVANCE}"
      fi
      if [ "$WITH_R2" = true ]; then
        cmd+=" --with-r2"
      fi
      echo "$cmd"
      return
      ;;
    *)
      echo "ERROR: unknown model '${MODEL}' (use: ranking | boat1 | trifecta)" >&2
      exit 1
      ;;
  esac
  cmd+=" --n-trials ${TRIALS}"
  cmd+=" --n-folds ${FOLDS}"
  cmd+=" --fold-months ${FOLD_MONTHS}"
  cmd+=" --seed ${SEED}"
  if [ -n "$TRAIN_START" ]; then
    cmd+=" --start-date ${TRAIN_START}"
  fi
  echo "$cmd"
}

_detach_run() {
  local tune_cmd
  tune_cmd=$(_build_cmd)

  # Archive previous log
  remote bash <<EOF
mkdir -p ${REMOTE_DIR_RESOLVED}/tune-logs
if [ -s "${REMOTE_LOG_FILE}" ]; then
  prev_ts=\$(head -1 "${REMOTE_LOG_FILE}" | tr ' :' '_-' || echo "unknown")
  cp "${REMOTE_LOG_FILE}" "${REMOTE_DIR_RESOLVED}/tune-logs/\${prev_ts}.log" 2>/dev/null || true
fi
EOF

  # Create run script
  remote bash <<EOF
cat > ${REMOTE_SCRIPT_FILE} << 'INNERSCRIPT'
#!/bin/bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd ${REMOTE_DIR_RESOLVED}
LOG="${REMOTE_LOG_FILE}"
trap 'rm -f ${REMOTE_PID_FILE}' EXIT
echo "Started at \$(date)" > "\$LOG"
echo "Command: ${tune_cmd}" >> "\$LOG"
echo "" >> "\$LOG"
${tune_cmd} >> "\$LOG" 2>&1
echo "" >> "\$LOG"
echo "=== Done ===" >> "\$LOG"
date >> "\$LOG"
INNERSCRIPT
chmod +x ${REMOTE_SCRIPT_FILE}
EOF

  remote "nohup ${REMOTE_SCRIPT_FILE} > /dev/null 2>&1 & echo \$! > ${REMOTE_PID_FILE}"
  local pid
  pid=$(remote "cat ${REMOTE_PID_FILE}")
  log "Started on ${REMOTE_HOSTNAME} (PID: ${pid})"
  log "  model=${MODEL} trials=${TRIALS} folds=${FOLDS}"
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
