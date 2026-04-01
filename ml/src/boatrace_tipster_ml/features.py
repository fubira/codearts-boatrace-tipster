"""Feature extraction pipeline for boat race prediction.

Builds a feature DataFrame from SQLite DB via DuckDB.
Historical features use the cum_all - cum_daily pattern to prevent
temporal leakage (same-day races are fully excluded from history).
"""

import numpy as np
import pandas as pd

from .db import get_connection
from .feature_config import (
    compute_interaction_features,
    compute_relative_features,
    encode_race_grade,
    encode_racer_class,
    encode_weather,
    prepare_feature_matrix,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_BASE_QUERY = """
SELECT
    re.id AS entry_id,
    r.id AS race_id,
    r.race_date,
    r.race_number,
    r.stadium_id,
    r.race_grade,
    r.weather,
    r.wind_speed,
    r.wind_direction,
    r.wave_height,
    r.temperature,
    r.water_temperature,
    re.racer_id,
    re.boat_number,
    COALESCE(re.course_number, re.boat_number) AS course_number,
    re.motor_number,
    re.racer_class,
    re.racer_weight,
    re.flying_count,
    re.late_count,
    re.average_st,
    re.national_win_rate,
    re.national_top2_rate,
    re.national_top3_rate,
    re.local_win_rate,
    re.local_top2_rate,
    re.local_top3_rate,
    re.motor_top2_rate,
    re.motor_top3_rate,
    re.boat_top2_rate,
    re.boat_top3_rate,
    re.exhibition_time,
    re.exhibition_st,
    re.tilt,
    re.stabilizer,
    re.start_timing,
    re.finish_position,
    EXTRACT(MONTH FROM CAST(r.race_date AS DATE)) AS race_month
FROM db.races r
JOIN db.race_entries re ON re.race_id = r.id
ORDER BY r.race_date, r.id, re.id
"""

# Rank assigned to non-finishers (disqualified, capsized, flying start return)
NON_FINISHER_RANK = 7


def _load_all_data(conn) -> pd.DataFrame:
    df = conn.execute(_BASE_QUERY).fetchdf()
    df = df.sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)
    # Assign worst rank to non-finishers to keep 6 entries per race
    df["finish_position"] = df["finish_position"].fillna(NON_FINISHER_RANK).astype(int)
    # Drop races that don't have exactly 6 entries (data corruption)
    race_counts = df.groupby("race_id").size()
    valid_races = race_counts[race_counts == 6].index
    df = df[df["race_id"].isin(valid_races)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Leak-safe cumulative helpers
# ---------------------------------------------------------------------------


def _cumulative_rate(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> np.ndarray:
    """Compute leak-safe cumulative rate using cum_all - cum_daily pattern.

    Excludes all same-day races from history to prevent temporal leakage.
    Returns NaN for first-ever occurrence in a group.
    """
    cum_all = df.groupby(group_cols, sort=False)[value_col].cumsum()
    count_all = df.groupby(group_cols, sort=False).cumcount() + 1

    daily_group = group_cols + ["race_date"]
    cum_daily = df.groupby(daily_group, sort=False)[value_col].cumsum()
    count_daily = df.groupby(daily_group, sort=False).cumcount() + 1

    prior = cum_all - cum_daily
    prior_count = count_all - count_daily

    return np.where(prior_count > 0, prior / prior_count, np.nan)


def _cumulative_mean(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> np.ndarray:
    """Compute leak-safe cumulative mean for a continuous value.

    NaN values in value_col are excluded from the accumulation.
    """
    valid = df[value_col].notna()
    val = df[value_col].fillna(0.0)

    cum_sum_all = df.assign(_val=val).groupby(group_cols, sort=False)["_val"].cumsum()
    cum_cnt_all = df.assign(_valid=valid.astype(int)).groupby(group_cols, sort=False)[
        "_valid"
    ].cumsum()

    daily_group = group_cols + ["race_date"]
    cum_sum_daily = (
        df.assign(_val=val).groupby(daily_group, sort=False)["_val"].cumsum()
    )
    cum_cnt_daily = df.assign(_valid=valid.astype(int)).groupby(
        daily_group, sort=False
    )["_valid"].cumsum()

    prior_sum = cum_sum_all - cum_sum_daily
    prior_cnt = cum_cnt_all - cum_cnt_daily

    return np.where(prior_cnt > 0, prior_sum / prior_cnt, np.nan)


def _cumulative_std(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> np.ndarray:
    """Compute leak-safe cumulative standard deviation.

    Uses the formula: std = sqrt(E[X^2] - E[X]^2).
    Requires at least 2 prior observations.
    """
    valid = df[value_col].notna()
    val = df[value_col].fillna(0.0)
    val_sq = val**2

    tmp = df.assign(_val=val, _val_sq=val_sq, _valid=valid.astype(int))

    cum_sum_all = tmp.groupby(group_cols, sort=False)["_val"].cumsum()
    cum_sq_all = tmp.groupby(group_cols, sort=False)["_val_sq"].cumsum()
    cum_cnt_all = tmp.groupby(group_cols, sort=False)["_valid"].cumsum()

    daily_group = group_cols + ["race_date"]
    cum_sum_daily = tmp.groupby(daily_group, sort=False)["_val"].cumsum()
    cum_sq_daily = tmp.groupby(daily_group, sort=False)["_val_sq"].cumsum()
    cum_cnt_daily = tmp.groupby(daily_group, sort=False)["_valid"].cumsum()

    prior_sum = cum_sum_all - cum_sum_daily
    prior_sq = cum_sq_all - cum_sq_daily
    prior_cnt = cum_cnt_all - cum_cnt_daily

    mean = prior_sum / prior_cnt
    var = prior_sq / prior_cnt - mean**2
    var = np.maximum(var, 0)  # numerical safety

    result = np.sqrt(var)
    result[prior_cnt < 2] = np.nan
    return result


# ---------------------------------------------------------------------------
# Category B: Historical features
# ---------------------------------------------------------------------------


def _add_racer_course_stats(df: pd.DataFrame) -> None:
    """B1: Racer's historical win/top2/top3 rate by course number."""
    df["_is_win"] = (df["finish_position"] == 1).astype(int)
    df["_is_top2"] = (df["finish_position"] <= 2).astype(int)
    df["_is_top3"] = (df["finish_position"] <= 3).astype(int)

    group = ["racer_id", "course_number"]
    df["racer_course_win_rate"] = _cumulative_rate(df, group, "_is_win")
    df["racer_course_top2_rate"] = _cumulative_rate(df, group, "_is_top2")
    df["racer_course_top3_rate"] = _cumulative_rate(df, group, "_is_top3")


def _add_course_taking_rate(df: pd.DataFrame) -> None:
    """B2: Rate of racer taking an inner course than their boat number."""
    df["_took_inner"] = (df["course_number"] < df["boat_number"]).astype(int)
    df["course_taking_rate"] = _cumulative_rate(df, ["racer_id"], "_took_inner")


def _add_course_avg_st(df: pd.DataFrame) -> None:
    """B3: Racer's average ST by boat number (proxy for expected course)."""
    df["course_avg_st"] = _cumulative_mean(
        df, ["racer_id", "boat_number"], "start_timing"
    )


def _add_recent_form(df: pd.DataFrame) -> None:
    """B4: Recent 20-race form (win rate, top2 rate, avg position).

    Uses cum_all - cum_daily pattern (same as other historical features)
    to fully exclude same-day races, then takes the last-20 rolling window.
    """
    df["_is_win"] = (df["finish_position"] == 1).astype(float)
    df["_is_top2"] = (df["finish_position"] <= 2).astype(float)
    df["_pos"] = df["finish_position"].astype(float)

    # Use cumulative rate (same-day excluded) as the base
    df["recent_win_rate"] = _cumulative_rate(df, ["racer_id"], "_is_win")
    df["recent_top2_rate"] = _cumulative_rate(df, ["racer_id"], "_is_top2")
    df["recent_avg_position"] = _cumulative_mean(df, ["racer_id"], "_pos")



def _add_st_stability(df: pd.DataFrame) -> None:
    """B6: Standard deviation of racer's start timing (lower = more stable)."""
    df["st_stability"] = _cumulative_std(df, ["racer_id"], "start_timing")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_categoricals(df: pd.DataFrame) -> None:
    """Apply categorical encodings to raw columns."""
    df["race_grade_code"] = df["race_grade"].map(encode_race_grade)
    df["racer_class_code"] = df["racer_class"].map(encode_racer_class)
    df["weather_code"] = df["weather"].map(encode_weather)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

_TEMP_COLS = ["_is_win", "_is_top2", "_is_top3", "_took_inner", "_pos"]


def _cleanup_temp_cols(df: pd.DataFrame) -> None:
    """Remove temporary computation columns."""
    for col in _TEMP_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(
    db_path: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the full feature matrix from the database.

    Loads all data, computes historical features (leak-safe), encodes
    categoricals, filters to date range, then computes relative and
    interaction features.

    Args:
        db_path: Path to the SQLite database.
        start_date: Include races on or after this date (YYYY-MM-DD).
        end_date: Include races before this date (YYYY-MM-DD).

    Returns:
        (X, y, meta) tuple ready for model training.
    """
    conn = get_connection(db_path)

    try:
        print("Loading data from database...")
        df = _load_all_data(conn)
        print(f"  Loaded {len(df)} entries")
    finally:
        conn.close()

    # Historical features (must be computed on full dataset before filtering)
    print("Computing historical features...")
    _add_racer_course_stats(df)
    _add_course_taking_rate(df)
    _add_course_avg_st(df)
    _add_recent_form(df)
    _add_st_stability(df)

    _cleanup_temp_cols(df)
    _encode_categoricals(df)

    # Filter to target date range
    if start_date is not None:
        df = df[df["race_date"] >= start_date]
    if end_date is not None:
        df = df[df["race_date"] < end_date]
    df = df.reset_index(drop=True)
    print(f"  Filtered to {len(df)} entries")

    # Relative and interaction features (within-race, computed after filtering)
    df = compute_relative_features(df)
    df = compute_interaction_features(df)

    return prepare_feature_matrix(df)
