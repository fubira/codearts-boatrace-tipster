"""Verify snapshot features match full pipeline output.

Compares build_features_df() vs build_features_from_snapshot() for a
given date. Reports max absolute difference per feature column.

Usage:
    uv run --directory ml python -m scripts.verify_snapshot --date 2026-04-03
"""

import argparse
import contextlib
import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURE_COLS
from boatrace_tipster_ml.boat1_features import BOAT1_FEATURE_COLS
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.snapshot_features import build_features_from_snapshot

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SNAPSHOTS_DIR = _PROJECT_ROOT / "data" / "stats-snapshots"

# Columns to skip in comparison (always NaN in snapshot mode)
SKIP_COLS = {"gate_bias", "upset_rate"}

# Tolerance for floating point comparison
ATOL = 1e-6


def verify(date: str, db_path: str, snapshot_path: str) -> bool:
    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    # Full pipeline
    print(f"Building full features for {date}...")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(db_path, start_date=date, end_date=next_day)
    print(f"  Full: {len(df_full)} entries in {time.time() - t0:.1f}s")

    # Snapshot pipeline
    print(f"Building snapshot features for {date}...")
    t1 = time.time()
    df_snap = build_features_from_snapshot(db_path, snapshot_path, date)
    print(f"  Snap: {len(df_snap)} entries in {time.time() - t1:.1f}s")

    if len(df_full) != len(df_snap):
        print(f"  FAIL: row count mismatch ({len(df_full)} vs {len(df_snap)})")
        return False

    if len(df_full) == 0:
        print(f"  SKIP: no data for {date}")
        return True

    # Align by race_id + boat_number
    df_full = df_full.sort_values(["race_id", "boat_number"]).reset_index(drop=True)
    df_snap = df_snap.sort_values(["race_id", "boat_number"]).reset_index(drop=True)

    # Compare all numeric columns that exist in both
    all_feature_cols = set(FEATURE_COLS) | set(BOAT1_FEATURE_COLS)
    # Also include intermediate computed columns used by both pipelines
    compare_cols = [
        c for c in df_full.columns
        if c in df_snap.columns
        and c not in SKIP_COLS
        and df_full[c].dtype in [np.float64, np.float32, np.int64, np.int32, "Int64"]
    ]

    ok = True
    mismatches = []

    for col in sorted(compare_cols):
        a = df_full[col].astype(float).values
        b = df_snap[col].astype(float).values

        # Both NaN = match
        both_nan = np.isnan(a) & np.isnan(b)
        # One NaN, other not = mismatch
        nan_mismatch = np.isnan(a) != np.isnan(b)
        n_nan_mismatch = nan_mismatch.sum()

        # Value difference where both non-NaN
        valid = ~np.isnan(a) & ~np.isnan(b)
        if valid.sum() > 0:
            max_diff = np.max(np.abs(a[valid] - b[valid]))
        else:
            max_diff = 0.0

        if n_nan_mismatch > 0 or max_diff > ATOL:
            important = col in all_feature_cols or col in [
                "racer_course_win_rate", "racer_course_top2_rate",
                "racer_course_top3_rate", "stadium_course_win_rate",
                "course_taking_rate", "recent_win_rate", "recent_top2_rate",
                "recent_avg_position", "course_avg_st", "st_stability",
                "self_exhibition_delta", "self_st_delta",
                "motor_quality_residual",
                "rolling_st_mean", "rolling_avg_position", "rolling_win_rate",
                "rolling_course_win_rate", "rolling_course_st",
                "tourn_exhibition_delta", "tourn_st_delta", "tourn_avg_position",
                "prev_day_exhibition_delta",
            ]
            status = "FAIL" if important else "warn"
            if important:
                ok = False
            mismatches.append((col, max_diff, n_nan_mismatch, status))

    if mismatches:
        print(f"\n  {'Column':40s} {'MaxDiff':>10s} {'NaN_mm':>8s} {'Status':>6s}")
        print(f"  {'-'*70}")
        for col, diff, n_nan, status in mismatches:
            print(f"  {col:40s} {diff:>10.6f} {n_nan:>8d} {status:>6s}")
    else:
        print(f"\n  All {len(compare_cols)} columns match within tolerance {ATOL}")

    speedup = (time.time() - t0 - (time.time() - t1)) / (time.time() - t1) if (time.time() - t1) > 0 else 0
    print(f"\n  Speedup: {(time.time()-t0-(time.time()-t1)):.1f}s → {time.time()-t1:.1f}s")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify snapshot features")
    parser.add_argument("--date", required=True, help="Date to verify (YYYY-MM-DD)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--snapshot", default=None, help="Snapshot path")
    args = parser.parse_args()

    # Find snapshot: use through_date = day before target
    if args.snapshot:
        snapshot_path = args.snapshot
    else:
        prev_day = (
            datetime.strptime(args.date, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        snapshot_path = str(DEFAULT_SNAPSHOTS_DIR / f"{prev_day}.db")

    snapshot_path = str(Path(snapshot_path).resolve()) if not Path(snapshot_path).is_absolute() else snapshot_path
    if not Path(snapshot_path).exists():
        print(f"Snapshot not found: {snapshot_path}")
        print(f"Build it first: python -m scripts.build_snapshot --through-date ...")
        sys.exit(1)

    ok = verify(args.date, args.db_path, snapshot_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
