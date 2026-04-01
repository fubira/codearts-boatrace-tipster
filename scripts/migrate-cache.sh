#!/usr/bin/env bash
# Migrate flat cache files into YYYYMM subdirectories.
# Before: data/cache/.../raceresult/rno=1&jcd=01&hd=20260225.html.gz
# After:  data/cache/.../raceresult/202602/rno=1&jcd=01&hd=20260225.html.gz
#
# Idempotent: skips files already in YYYYMM directories.
# Uses xargs for parallel mv.

set -euo pipefail

CACHE_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/cache"

if [ ! -d "$CACHE_DIR" ]; then
  echo "Cache directory not found: $CACHE_DIR"
  exit 1
fi

# Find files not already in a YYYYMM directory, containing hd= in filename
find "$CACHE_DIR" -name '*hd=*.html.gz' -type f \
  | grep -v '/[0-9]\{6\}/[^/]*$' \
  | xargs -P 8 -I{} bash -c '
    file="$1"
    filename="$(basename "$file")"
    parent="$(dirname "$file")"
    yyyymm=$(echo "$filename" | grep -oP "hd=\K\d{6}")
    dest="$parent/$yyyymm"
    mkdir -p "$dest"
    mv "$file" "$dest/$filename"
  ' _ {}

echo "Migration complete."
