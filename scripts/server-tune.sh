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
SEED=""  # empty = auto random; explicit --seed N overrides for reproducibility
FROM_MODEL=""
NARROW=false
# n_jobs=2 + num_threads=2 = 4 threads total (= half of i7-6700's 8 LCPU),
# leaving room for scraper / runner / interactive work on the same machine.
N_JOBS=2
NUM_THREADS=2
OBJECTIVE=""
FIX_THRESHOLDS=""
PHASE2_TOP=""            # empty = auto-scale from TRIALS (see resolve below).
                         # --phase2 N: override. --no-phase2: disable.
PHASE2_FROM="2026-01-01"
PHASE2_TO=""             # empty = today

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
    --phase2) PHASE2_TOP="$2"; shift 2 ;;
    --no-phase2) PHASE2_TOP="SKIP"; shift ;;
    --phase2-from) PHASE2_FROM="$2"; shift 2 ;;
    --phase2-to) PHASE2_TO="$2"; shift 2 ;;
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
  --seed N          random seed (default: 自動ランダム、明示で再現性確保)
  --objective O     tune_p2 objective: growth | kelly (default: growth)
  --fix-thresholds  閾値固定でハイパラのみ探索 (e.g., "gap23=0.13,ev=0.0,top3_conc=0.7")
  --from-model D    既存モデルのHPを初期trialとして投入（カンマ区切り可能）
  --narrow          --from-model の最初のモデル周辺だけを探索（要 --from-model）
  --n-jobs N        並列 trial 数 (default: 2、サーバーでのみ許可)
  --num-threads N   trial あたりの LightGBM スレッド数 (default: 2 = 4thread/2jobs、i7-6700 の半分)

Phase 2 (デフォルト ON、Phase 1 完了後にサーバで連続実行):
  --phase2 N        Kelly 上位 N 個を seed_stability_check で 5 seed 評価。
                    省略時は trials/25 で自動 scale (50→2, 400→16, 500→20)。
                    2026-04-14 実測で「trial 数 / ~25 個」が真に検証すべき候補数。
                    最終 ranking は stability_score (mean - std) で並ぶ。
  --no-phase2       Phase 2 を無効化 (Phase 1 だけ実行)
  --phase2-from D   Phase 2 OOS 評価期間 開始 (default: 2026-01-01)
  --phase2-to D     Phase 2 OOS 評価期間 終了 (default: 今日)

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
  log "Completed. Auto-fetching (Phase 1 + Phase 2 both ran on server)..."
  _fetch
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

  # Show last ~60 lines (includes Phase 2 stability ranking output).
  # Phase 1 + Phase 2 both ran on the server as part of the detach script,
  # so the log already contains the final stability_score ranking.
  echo ""
  echo "=== Results ==="
  remote "tail -60 ${REMOTE_LOG_FILE} 2>/dev/null" || true
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
  cmd+=" --run-prefix ${RUN_PREFIX}"
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
  # Always forward so the choice is visible in remote command logs and so the
  # behavior is decoupled from any future change to tune_p2's defaults.
  cmd+=" --n-jobs ${N_JOBS}"
  cmd+=" --num-threads ${NUM_THREADS}"
  echo "$cmd"
}

_detach_run() {
  local tune_cmd
  tune_cmd=$(_build_cmd)

  # Resolve Phase 2 top-N. Empty = auto-scale: max(TRIALS/25, 2).
  # This gives 50 trials → 2, 100 → 4, 400 → 16, 500 → 20 — scales with
  # how many "real candidates" a tune run typically produces (rough rule
  # of thumb from 2026-04-14: one usable HP per ~25-50 trials).
  # Explicit "SKIP" from --no-phase2 disables Phase 2 entirely.
  local phase2_top_resolved=""
  if [ "${PHASE2_TOP}" = "SKIP" ]; then
    phase2_top_resolved=""
  elif [ -z "${PHASE2_TOP}" ]; then
    phase2_top_resolved=$(( TRIALS / 25 ))
    if [ "${phase2_top_resolved}" -lt 2 ]; then
      phase2_top_resolved=2
    fi
  else
    phase2_top_resolved="${PHASE2_TOP}"
  fi

  # Resolve gap12_th from local active model so Phase 2 uses the current
  # production filter (not a hardcoded value that drifts).
  local active_model
  active_model="$(cd ml && uv run python -c "import json; print(json.load(open('models/active.json'))['model'])" 2>/dev/null || echo "p2_v2")"
  local gap12_th
  gap12_th="$(cd ml && uv run python -c "import json; m=json.load(open('models/${active_model}/ranking/model_meta.json')); print(m['strategy']['gap12_min_threshold'])" 2>/dev/null || echo "0.04")"
  local phase2_to="${PHASE2_TO:-$(date '+%Y-%m-%d')}"

  # Archive previous log using its mtime as YYYY-MM-DD_HHMM
  remote bash <<EOF
mkdir -p ${REMOTE_DIR_RESOLVED}/tune-logs
if [ -s "${REMOTE_LOG_FILE}" ]; then
  prev_ts=\$(date -r "${REMOTE_LOG_FILE}" '+%Y-%m-%d_%H%M' 2>/dev/null || echo "unknown")
  cp "${REMOTE_LOG_FILE}" "${REMOTE_DIR_RESOLVED}/tune-logs/\${prev_ts}_server-tune.log" 2>/dev/null || true
fi
EOF

  # Build the Phase 2 command (seed_stability_check) to run on server
  # right after Phase 1 completes. tateyamakun pattern: both phases run
  # inside the remote script so the user only kicks once and only
  # fetches once — no local A→B→C babysitting.
  local phase2_cmd=""
  if [ -n "${phase2_top_resolved}" ]; then
    phase2_cmd="nice -n 19 ionice -c 3 uv run --directory ml python -m scripts.seed_stability_check"
    phase2_cmd+=" --tune-log ${REMOTE_LOG_FILE}"
    phase2_cmd+=" --top-n ${phase2_top_resolved}"
    phase2_cmd+=" --from ${PHASE2_FROM}"
    phase2_cmd+=" --to ${phase2_to}"
    phase2_cmd+=" --gap12-th ${gap12_th}"
  fi

  # Create run script
  remote bash <<EOF
cat > ${REMOTE_SCRIPT_FILE} << 'INNERSCRIPT'
#!/bin/bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export BOATRACE_TUNE_PARALLEL=1
# Force line-buffered Python output so per-trial Optuna logs appear
# immediately in the remote log file (otherwise nohup file output is
# block-buffered and trials only surface in batches of ~20).
export PYTHONUNBUFFERED=1
cd ${REMOTE_DIR_RESOLVED}
LOG="${REMOTE_LOG_FILE}"
trap 'rm -f ${REMOTE_PID_FILE}' EXIT
echo "Started at \$(date '+%Y-%m-%d %H:%M:%S %Z')" > "\$LOG"
echo "Command: ${tune_cmd}" >> "\$LOG"
echo "" >> "\$LOG"
# Phase 1: Optuna tune (nice / ionice to yield CPU+IO to docker containers
# running on the same machine: runner / scraper / watchtower).
echo "=== Phase 1: Optuna tune ===" >> "\$LOG"
nice -n 19 ionice -c 3 ${tune_cmd} >> "\$LOG" 2>&1
echo "" >> "\$LOG"
# Phase 2: seed_stability_check on Kelly top-N of Phase 1 results.
# Runs on server so no local intervention is required.
if [ -n "${phase2_cmd}" ]; then
  echo "=== Phase 2: seed_stability_check ===" >> "\$LOG"
  date '+%Y-%m-%d %H:%M:%S %Z' >> "\$LOG"
  ${phase2_cmd} >> "\$LOG" 2>&1 || echo "Phase 2 failed" >> "\$LOG"
  echo "" >> "\$LOG"
fi
echo "=== Done ===" >> "\$LOG"
date '+%Y-%m-%d %H:%M:%S %Z' >> "\$LOG"
INNERSCRIPT
chmod +x ${REMOTE_SCRIPT_FILE}
EOF

  remote "nohup ${REMOTE_SCRIPT_FILE} > /dev/null 2>&1 & echo \$! > ${REMOTE_PID_FILE}"
  local pid
  pid=$(remote "cat ${REMOTE_PID_FILE}")
  log "Started on ${REMOTE_HOSTNAME} (PID: ${pid})"
  log "  Phase 1: ${TRIALS} trials × ${FOLDS} folds"
  if [ -n "${phase2_top_resolved}" ]; then
    log "  Phase 2: Kelly top ${phase2_top_resolved} × 5 seeds (gap12=${gap12_th}, to=${phase2_to})"
  else
    log "  Phase 2: skipped (--no-phase2)"
  fi
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
export PYTHONUNBUFFERED=1
cd ${REMOTE_DIR_RESOLVED}
nice -n 19 ionice -c 3 ${tune_cmd}
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

# --- Allocate seed (random by default to avoid duplicate runs) ---
if [ -z "$SEED" ]; then
  SEED=$(( (RANDOM << 15) | RANDOM ))
  log "Auto seed: ${SEED}"
else
  log "Using explicit seed: ${SEED}"
fi

# --- Allocate run prefix from local registry ---
# Tateyamakun-style: each tune run consumes one prefix from the local
# .run-counter so all dev models from this tune share the same identifier
# (e.g., ab_294, ab_266 all came from run "ab"). The counter lives only
# on local; rsync would otherwise overwrite a server-side increment.
RUN_PREFIX=$(cd "${ML_DIR}" && PYTHONPATH=src uv run python -c \
    "from boatrace_tipster_ml.registry import next_prefix; print(next_prefix())")
if [ -z "$RUN_PREFIX" ]; then
  echo "ERROR: failed to allocate run prefix from local registry" >&2
  exit 1
fi
log "Run prefix: ${RUN_PREFIX}"

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
