"""Build and load stats snapshots for lightweight inference.

A snapshot captures sufficient statistics (cumulative sums/counts, daily
aggregates) so that prediction can skip the full 760k-row DB scan.

Snapshot is a SQLite file with tables:
  - cumulative_stats: per-group (sum, count, sum_sq) for cumulative features
  - motor_stats: per-(stadium, motor) residual stats
  - rolling_daily: per-(group, date) daily aggregates for rolling features
  - snapshot_meta: metadata (through_date, built_at, entry_count)
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .db import get_connection
from .features import (
    NON_FINISHER_RANK,
    ROLLING_WINDOW,
    ROLLING_WINDOW_COURSE,
    ROLLING_WINDOW_GENERAL,
    _BASE_QUERY,
)

# Rolling window buffer: keep extra days to avoid edge effects
_ROLLING_BUFFER = 5


# ---------------------------------------------------------------------------
# Build snapshot
# ---------------------------------------------------------------------------


def build_snapshot(
    db_path: str,
    cache_path: str,
    through_date: str,
) -> None:
    """Build a stats snapshot from the main DB.

    Args:
        db_path: Path to the main SQLite database.
        cache_path: Output path for the snapshot SQLite file.
        through_date: Include data up to and including this date (YYYY-MM-DD).
            Typically yesterday, so prediction on today excludes same-day data.
    """
    t0 = time.time()
    print(f"Building snapshot through {through_date}...")

    # Load all historical data
    conn = get_connection(db_path)
    try:
        df = conn.execute(_BASE_QUERY).fetchdf()
    finally:
        conn.close()

    df = df.sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)
    # Drop void races (all 6 entries have NULL finish_position = race not established)
    void_races = df.groupby("race_id")["finish_position"].apply(lambda x: x.isna().all())
    df = df[~df["race_id"].isin(void_races[void_races].index)].reset_index(drop=True)
    df["finish_position"] = df["finish_position"].fillna(NON_FINISHER_RANK).astype(int)

    # Drop races without exactly 6 entries
    race_counts = df.groupby("race_id").size()
    valid_races = race_counts[race_counts == 6].index
    df = df[df["race_id"].isin(valid_races)].reset_index(drop=True)

    # Filter to through_date
    df = df[df["race_date"] <= through_date].reset_index(drop=True)
    n_entries = len(df)
    print(f"  {n_entries} entries loaded ({n_entries // 6} races)")

    # Derived columns
    df["_is_win"] = (df["finish_position"] == 1).astype(int)
    df["_is_top2"] = (df["finish_position"] <= 2).astype(int)
    df["_is_top3"] = (df["finish_position"] <= 3).astype(int)
    df["_took_inner"] = (df["course_number"] < df["boat_number"]).astype(int)
    df["_pos_alpha"] = (df["boat_number"] - df["finish_position"]).astype(float)

    # Write to SQLite
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cache.unlink()

    sconn = sqlite3.connect(str(cache))
    _create_tables(sconn)

    # A. Cumulative stats
    print("  Computing cumulative stats...")
    _build_cumulative_stats(df, sconn)

    # B. Motor residual stats
    print("  Computing motor residual stats...")
    _build_motor_stats(df, sconn)

    # C. Rolling daily aggregates
    print("  Computing rolling daily aggregates...")
    _build_rolling_daily(df, through_date, sconn)

    # D. Metadata
    sconn.execute(
        "INSERT INTO snapshot_meta (key, value) VALUES (?, ?)",
        ("through_date", through_date),
    )
    sconn.execute(
        "INSERT INTO snapshot_meta (key, value) VALUES (?, ?)",
        ("built_at", datetime.now().isoformat()),
    )
    sconn.execute(
        "INSERT INTO snapshot_meta (key, value) VALUES (?, ?)",
        ("entry_count", str(n_entries)),
    )

    sconn.commit()
    sconn.close()

    elapsed = time.time() - t0
    size_mb = cache.stat().st_size / 1024 / 1024
    print(f"  Snapshot saved: {cache_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE cumulative_stats (
            stat_name TEXT NOT NULL,
            group_key TEXT NOT NULL,
            total_sum REAL NOT NULL,
            total_count INTEGER NOT NULL,
            total_sum_sq REAL,
            PRIMARY KEY (stat_name, group_key)
        )
    """)
    conn.execute("""
        CREATE TABLE motor_stats (
            stadium_id INTEGER NOT NULL,
            motor_number INTEGER NOT NULL,
            residual_sum REAL NOT NULL,
            residual_count INTEGER NOT NULL,
            PRIMARY KEY (stadium_id, motor_number)
        )
    """)
    conn.execute("""
        CREATE TABLE rolling_daily (
            stat_name TEXT NOT NULL,
            group_key TEXT NOT NULL,
            race_date TEXT NOT NULL,
            day_sum REAL NOT NULL,
            day_count INTEGER NOT NULL,
            PRIMARY KEY (stat_name, group_key, race_date)
        )
    """)
    conn.execute("""
        CREATE TABLE snapshot_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)


# ---------------------------------------------------------------------------
# Cumulative stats extraction
# ---------------------------------------------------------------------------

# Each entry: (stat_name, group_cols, value_col, aggregation_type)
# aggregation_type: "rate" (sum of 0/1), "mean" (sum of continuous), "std" (sum + sum_sq)
_CUMULATIVE_DEFS: list[tuple[str, list[str], str, str]] = [
    # Racer × course stats
    ("racer_course_win", ["racer_id", "course_number"], "_is_win", "rate"),
    ("racer_course_top2", ["racer_id", "course_number"], "_is_top2", "rate"),
    ("racer_course_top3", ["racer_id", "course_number"], "_is_top3", "rate"),
    # Stadium × course
    ("stadium_course_win", ["stadium_id", "course_number"], "_is_win", "rate"),
    # Per-racer
    ("racer_course_taking", ["racer_id"], "_took_inner", "rate"),
    ("racer_recent_win", ["racer_id"], "_is_win", "rate"),
    ("racer_recent_top2", ["racer_id"], "_is_top2", "rate"),
    # Cumulative means
    ("racer_avg_position", ["racer_id"], "finish_position", "mean"),
    ("racer_boat_avg_st", ["racer_id", "boat_number"], "start_timing", "mean"),
    ("racer_avg_exhibition", ["racer_id"], "exhibition_time", "mean"),
    ("racer_avg_exhibition_st", ["racer_id"], "exhibition_st", "mean"),
    # Cumulative std
    ("racer_st_stability", ["racer_id"], "start_timing", "std"),
    # Position alpha (skill net of lane advantage)
    ("racer_position_alpha", ["racer_id"], "_pos_alpha", "mean"),
]


def _build_cumulative_stats(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    rows = []
    for stat_name, group_cols, value_col, agg_type in _CUMULATIVE_DEFS:
        if agg_type in ("rate", "mean"):
            if agg_type == "rate":
                val = df[value_col].astype(float)
                valid = pd.Series(np.ones(len(df)), dtype=float)
            else:
                valid = df[value_col].notna().astype(float)
                val = df[value_col].fillna(0.0).astype(float)

            tmp = df[group_cols].copy()
            tmp["_val"] = val.values
            tmp["_valid"] = valid.values

            agg = tmp.groupby(group_cols, sort=False).agg(
                _sum=("_val", "sum"),
                _count=("_valid", "sum"),
            )
            for idx, row in agg.iterrows():
                key = _make_group_key(idx, group_cols)
                rows.append((stat_name, key, float(row["_sum"]), int(row["_count"]), None))

        elif agg_type == "std":
            valid = df[value_col].notna().astype(float)
            val = df[value_col].fillna(0.0).astype(float)
            val_sq = val**2

            tmp = df[group_cols].copy()
            tmp["_val"] = val.values
            tmp["_val_sq"] = val_sq.values
            tmp["_valid"] = valid.values

            agg = tmp.groupby(group_cols, sort=False).agg(
                _sum=("_val", "sum"),
                _count=("_valid", "sum"),
                _sum_sq=("_val_sq", "sum"),
            )
            for idx, row in agg.iterrows():
                key = _make_group_key(idx, group_cols)
                rows.append((stat_name, key, float(row["_sum"]), int(row["_count"]), float(row["_sum_sq"])))

    conn.executemany(
        "INSERT INTO cumulative_stats (stat_name, group_key, total_sum, total_count, total_sum_sq) VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def _make_group_key(idx, group_cols: list[str]) -> str:
    """Convert groupby index to string key like '1234:3'."""
    if len(group_cols) == 1:
        return str(idx)
    return ":".join(str(v) for v in idx)


# ---------------------------------------------------------------------------
# Motor residual stats
# ---------------------------------------------------------------------------


def _build_motor_stats(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Compute motor quality residual using the same cumulative approach as features.py.

    Replicates _add_motor_residual: per-row leak-safe racer avg exhibition → residual
    → per-motor cumulative mean of residuals. Stores the final accumulated
    (sum, count) per (stadium, motor).
    """
    # NOTE: Using features._cumulative_mean directly to ensure identical
    # leak-safe computation. If _cumulative_mean's interface changes,
    # this must be updated in tandem.
    from .features import _cumulative_mean

    # Step 1: Cumulative racer avg exhibition time (leak-safe, per row)
    racer_avg_ex = _cumulative_mean(df, ["racer_id"], "exhibition_time")

    # Step 2: Per-row residual
    residual = df["exhibition_time"] - racer_avg_ex

    # Step 3: Accumulate residuals per (stadium, motor)
    # Use the same logic as _cumulative_mean but manually aggregate the final totals
    valid = residual.notna()
    val = residual.fillna(0.0)

    tmp = df[["stadium_id", "motor_number"]].copy()
    tmp["_val"] = val.values
    tmp["_valid"] = valid.astype(int).values

    motor_agg = tmp.groupby(["stadium_id", "motor_number"], sort=False).agg(
        _sum=("_val", "sum"),
        _count=("_valid", "sum"),
    )

    rows = [
        (int(idx[0]), int(idx[1]), float(row["_sum"]), int(row["_count"]))
        for idx, row in motor_agg.iterrows()
    ]
    conn.executemany(
        "INSERT INTO motor_stats (stadium_id, motor_number, residual_sum, residual_count) VALUES (?, ?, ?, ?)",
        rows,
    )


# ---------------------------------------------------------------------------
# Rolling daily aggregates
# ---------------------------------------------------------------------------

# Each entry: (stat_name, group_cols, value_col, window_size)
_ROLLING_DEFS: list[tuple[str, list[str], str, int]] = [
    ("racer_st", ["racer_id"], "start_timing", ROLLING_WINDOW_GENERAL or ROLLING_WINDOW),
    ("racer_position", ["racer_id"], "finish_position", ROLLING_WINDOW_GENERAL or ROLLING_WINDOW),
    ("racer_win", ["racer_id"], "_is_win", ROLLING_WINDOW_GENERAL or ROLLING_WINDOW),
    ("racer_course_win", ["racer_id", "course_number"], "_is_win", ROLLING_WINDOW_COURSE or ROLLING_WINDOW),
    ("racer_course_st", ["racer_id", "course_number"], "start_timing", ROLLING_WINDOW_COURSE or ROLLING_WINDOW),
    ("racer_position_alpha", ["racer_id"], "_pos_alpha", ROLLING_WINDOW_GENERAL or ROLLING_WINDOW),
    ("racer_course_position_alpha", ["racer_id", "course_number"], "_pos_alpha", ROLLING_WINDOW_COURSE or ROLLING_WINDOW),
]


def _build_rolling_daily(df: pd.DataFrame, through_date: str, conn: sqlite3.Connection) -> None:
    all_rows = []

    for stat_name, group_cols, value_col, window in _ROLLING_DEFS:
        valid = df[value_col].notna()
        val = df[value_col].fillna(0.0).astype(float)

        tmp = df[group_cols + ["race_date"]].copy()
        tmp["_val"] = val.values
        tmp["_valid"] = valid.astype(int).values

        # Aggregate to daily level
        daily = (
            tmp.groupby(group_cols + ["race_date"], sort=False)
            .agg(day_sum=("_val", "sum"), day_count=("_valid", "sum"))
            .reset_index()
        )

        # Keep only last (window + buffer) race-days per group
        max_days = window + _ROLLING_BUFFER
        daily = daily.sort_values(group_cols + ["race_date"])
        daily = daily.groupby(group_cols, sort=False).tail(max_days)

        for _, row in daily.iterrows():
            key = _make_group_key(
                tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]],
                group_cols,
            )
            all_rows.append((stat_name, key, str(row["race_date"]), float(row["day_sum"]), int(row["day_count"])))

    conn.executemany(
        "INSERT INTO rolling_daily (stat_name, group_key, race_date, day_sum, day_count) VALUES (?, ?, ?, ?, ?)",
        all_rows,
    )


# ---------------------------------------------------------------------------
# Load snapshot
# ---------------------------------------------------------------------------


def load_snapshot(cache_path: str) -> dict:
    """Load a snapshot into memory for fast lookup.

    Returns:
        {
            "meta": {"through_date": "...", "built_at": "...", "entry_count": "..."},
            "cumulative": {(stat_name, group_key): (total_sum, total_count, total_sum_sq), ...},
            "motor": {(stadium_id, motor_number): (residual_sum, residual_count), ...},
            "rolling": {(stat_name, group_key): [(race_date, day_sum, day_count), ...], ...},
        }
    """
    conn = sqlite3.connect(cache_path)

    # Meta
    meta = {}
    for key, value in conn.execute("SELECT key, value FROM snapshot_meta"):
        meta[key] = value

    # Cumulative stats
    cumulative: dict[tuple[str, str], tuple[float, int, float | None]] = {}
    for row in conn.execute(
        "SELECT stat_name, group_key, total_sum, total_count, total_sum_sq FROM cumulative_stats"
    ):
        cumulative[(row[0], row[1])] = (row[2], row[3], row[4])

    # Motor stats
    motor: dict[tuple[int, int], tuple[float, int]] = {}
    for row in conn.execute(
        "SELECT stadium_id, motor_number, residual_sum, residual_count FROM motor_stats"
    ):
        motor[(row[0], row[1])] = (row[2], row[3])

    # Rolling daily: group by (stat_name, group_key), sorted by date
    rolling: dict[tuple[str, str], list[tuple[str, float, int]]] = {}
    for row in conn.execute(
        "SELECT stat_name, group_key, race_date, day_sum, day_count FROM rolling_daily ORDER BY stat_name, group_key, race_date"
    ):
        k = (row[0], row[1])
        if k not in rolling:
            rolling[k] = []
        rolling[k].append((row[2], row[3], row[4]))

    conn.close()

    return {
        "meta": meta,
        "cumulative": cumulative,
        "motor": motor,
        "rolling": rolling,
    }
