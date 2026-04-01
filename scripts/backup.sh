#!/usr/bin/env bash
# boatrace-tipster バックアップスクリプト
#
# DB・HTMLキャッシュをバックアップする。
# WALチェックポイント実行後にtar.gz + SHA256チェックサムを生成。
#
# 使い方:
#   ./scripts/backup.sh [オプション] [出力先ディレクトリ]
#
# オプション:
#   --rotate <日数>   指定日数超の古いバックアップを自動削除
#   --db-only         DBのみ（キャッシュを除外）
#
# 例:
#   ./scripts/backup.sh /mnt/h/backup/boatrace-tipster                           # フルバックアップ
#   ./scripts/backup.sh --db-only /mnt/h/backup/boatrace-tipster                 # DBのみ
#   ./scripts/backup.sh --rotate 21 /mnt/h/backup/boatrace-tipster               # 21日ローテーション
#   ./scripts/backup.sh --db-only "/mnt/c/Users/matsushita/OneDrive - codearts.jp/backup/boatrace-tipster/"
#
# cron設定例（毎日4:30）:
#   30 4 * * * /path/to/scripts/backup.sh --rotate 21 /storage/backup/daily >> /path/to/backup.log 2>&1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE=$(date +%Y%m%d)

# --- 引数パース ---
ROTATE_DAYS=0
DB_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --rotate) ROTATE_DAYS="$2"; shift 2 ;;
    --db-only) DB_ONLY=true; shift ;;
    -*) echo "Unknown option: $1"; exit 1 ;;
    *) break ;;
  esac
done

DEST_DIR="${1:-/tmp}"

if [[ "$DB_ONLY" == true ]]; then
  ARCHIVE_NAME="boatrace-tipster-db-${DATE}.tar.gz"
else
  ARCHIVE_NAME="boatrace-tipster-backup-${DATE}.tar.gz"
fi
ARCHIVE_PATH="${DEST_DIR}/${ARCHIVE_NAME}"

# --- バックアップ対象 ---
TARGETS=(
  "data/boatrace-tipster.db"  # SQLite DB（最重要・復旧に数十分〜数時間）
)

if [[ "$DB_ONLY" == false ]]; then
  TARGETS+=(
    "data/cache"              # HTMLキャッシュ（再取得に数時間、4GB）
  )
fi

# 存在チェック
for target in "${TARGETS[@]}"; do
  if [[ ! -e "${PROJECT_ROOT}/${target}" ]]; then
    echo "ERROR: ${target} が見つかりません" >&2
    exit 1
  fi
done

mkdir -p "${DEST_DIR}"

# --- SQLite WALチェックポイント ---
echo "[$(date '+%H:%M:%S')] WALチェックポイント実行中..."
sqlite3 "${PROJECT_ROOT}/data/boatrace-tipster.db" 'PRAGMA wal_checkpoint(TRUNCATE);'

# --- アーカイブ作成 ---
echo "[$(date '+%H:%M:%S')] バックアップ開始"
echo "対象:"
for target in "${TARGETS[@]}"; do
  SIZE=$(du -sh "${PROJECT_ROOT}/${target}" | cut -f1)
  echo "  ${target} (${SIZE})"
done

tar -czf "${ARCHIVE_PATH}" -C "${PROJECT_ROOT}" "${TARGETS[@]}"
ARCHIVE_SIZE=$(du -sh "${ARCHIVE_PATH}" | cut -f1)

# SHA256チェックサム
CHECKSUM=$(sha256sum "${ARCHIVE_PATH}" | cut -d' ' -f1)
echo "${CHECKSUM}  ${ARCHIVE_NAME}" > "${ARCHIVE_PATH}.sha256"

echo "  作成: ${ARCHIVE_PATH} (${ARCHIVE_SIZE})"
echo "  SHA256: ${CHECKSUM}"

# --- ローテーション ---
if [[ "$ROTATE_DAYS" -gt 0 ]]; then
  DELETED=$(find "${DEST_DIR}" -name "boatrace-tipster-*-*.tar.gz*" -mtime +"${ROTATE_DAYS}" -delete -print | wc -l)
  if [[ "${DELETED}" -gt 0 ]]; then
    echo "  削除: ${DELETED} 件の古いバックアップを削除（${ROTATE_DAYS}日超）"
  fi
fi

echo "[$(date '+%H:%M:%S')] バックアップ完了"
