"""Build a stats snapshot for lightweight inference.

Usage:
    uv run --directory ml python -m scripts.build_snapshot --through-date 2026-04-03
    uv run --directory ml python -m scripts.build_snapshot --through-date 2026-04-03 --output data/stats-snapshots/2026-04-04.db
"""

import argparse
from pathlib import Path

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.snapshot import build_snapshot

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SNAPSHOTS_DIR = str(_PROJECT_ROOT / "data" / "stats-snapshots")


def main():
    parser = argparse.ArgumentParser(description="Build stats snapshot for inference")
    parser.add_argument(
        "--through-date",
        required=True,
        help="Include data through this date (YYYY-MM-DD). Typically yesterday.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Main DB path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output snapshot path (default: data/stats-snapshots/<through_date>.db)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = f"{DEFAULT_SNAPSHOTS_DIR}/{args.through_date}.db"

    build_snapshot(args.db_path, output, args.through_date)


if __name__ == "__main__":
    main()
