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
PHASE2_TOP="10"          # default 10 by Kelly. --phase2 N to override, --no-phase2 to disable
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
    --no-phase2) PHASE2_TOP=""; shift ;;
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

Phase 2 (--fetch 後に自動実行、デフォルト ON):
  --phase2 N        Kelly 上位 N 個を seed_stability_check で 5 seed 評価 (default: 10)。
                    最終 ranking は stability_score (mean - std) で並ぶ。
                    Phase 1 の WF-CV 値は seed luck で上下するため、本来の性能は
                    Phase 2 でしか測れない (2026-04-14 確定)。
  --no-phase2       Phase 2 を無効化 (Phase 1 ログだけ取得して終了)
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
  log "Tune completed. Auto-fetching + Phase 2..."
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

  # Show results summary from log
  echo ""
  echo "=== Results ==="
  remote "sed -n '/^=\\{60\\}/,/^Total time/p' ${REMOTE_LOG_FILE} 2>/dev/null" || true

  # Phase 2: Local seed_stability_check on Kelly-top trials.
  # Phase 1 (above) used a single LightGBM seed and is noise-dominated,
  # so its top trials may include winner's curse picks (e.g. 4-12 #266 was
  # growth #1 but OOS-worst). Phase 2 retrains the Kelly-top N candidates
  # with 5 seeds and ranks them by stability_score = mean - std.
  if [ -n "${PHASE2_TOP}" ]; then
    local to_date="${PHASE2_TO:-$(date '+%Y-%m-%d')}"
    local abs_log
    abs_log="$(realpath "${log_file}")"
    local phase2_log="${log_file%.log}.phase2.log"
    local abs_phase2
    abs_phase2="$(realpath -m "${phase2_log}")"

    # Read active model's gap12_min_threshold from its model_meta.json so
    # Phase 2 always evaluates at the current production filter, not a
    # hardcoded value that drifts when production filter changes.
    local active_model
    active_model="$(cd ml && uv run python -c "import json; print(json.load(open('models/active.json'))['model'])" 2>/dev/null || echo "p2_v2")"
    local gap12_th
    gap12_th="$(cd ml && uv run python -c "import json; m=json.load(open('models/${active_model}/ranking/model_meta.json')); print(m['strategy']['gap12_min_threshold'])" 2>/dev/null || echo "0.04")"

    log "=== Phase 2: seed_stability_check (top ${PHASE2_TOP} by Kelly) ==="
    log "  OOS period: ${PHASE2_FROM} ~ ${to_date}"
    log "  gap12_th: ${gap12_th} (from models/${active_model}/ranking/model_meta.json)"
    log "  Tune log: ${log_file}"
    log "  Phase 2 log: ${phase2_log}"
    (
      cd ml && PYTHONPATH=scripts:src uv run python scripts/seed_stability_check.py \
        --tune-log "${abs_log}" \
        --top-n "${PHASE2_TOP}" \
        --from "${PHASE2_FROM}" \
        --to "${to_date}" \
        --gap12-th "${gap12_th}"
    ) 2>&1 | tee "${abs_phase2}" || log "WARNING: phase2 seed_stability_check failed"
  fi
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
# nice / ionice: yield CPU and IO to the docker containers (runner /
# scraper / watchtower) running on the same machine. tune is a
# background research workload — production tasks must always win.
nice -n 19 ionice -c 3 ${tune_cmd} >> "\$LOG" 2>&1
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
