"""Feature extraction pipeline for boat race prediction.

Builds a feature DataFrame from SQLite DB via DuckDB.
Historical features use the cum_all - cum_daily pattern to prevent
temporal leakage (same-day races are fully excluded from history).
"""

import hashlib
import os
import time
from pathlib import Path

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
    r.race_title,
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
    re.bc_lap_time,
    re.bc_turn_time,
    re.bc_straight_time,
    re.bc_slit_diff,
    odds.odds AS tansho_odds,
    EXTRACT(MONTH FROM CAST(r.race_date AS DATE)) AS race_month
FROM db.races r
JOIN db.race_entries re ON re.race_id = r.id
LEFT JOIN db.race_odds odds
    ON odds.race_id = r.id
    AND odds.bet_type = '単勝'
    AND odds.combination = CAST(re.boat_number AS TEXT)
ORDER BY r.race_date, r.id, re.id
"""

# Rank assigned to non-finishers (disqualified, capsized, flying start return)
NON_FINISHER_RANK = 7


def _load_all_data(conn) -> pd.DataFrame:
    df = conn.execute(_BASE_QUERY).fetchdf()
    df = df.sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)
    # Drop void races (all 6 entries have NULL finish_position = race not established)
    void_races = df.groupby("race_id")["finish_position"].apply(lambda x: x.isna().all())
    df = df[~df["race_id"].isin(void_races[void_races].index)].reset_index(drop=True)
    # Assign worst rank to non-finishers to keep 6 entries per race
    df["finish_position"] = df["finish_position"].fillna(NON_FINISHER_RANK).astype(int)
    # Drop races that don't have exactly 6 entries (data corruption)
    race_counts = df.groupby("race_id").size()
    valid_races = race_counts[race_counts == 6].index
    df = df[df["race_id"].isin(valid_races)].reset_index(drop=True)
    # Popularity (betting rank) from tansho odds: lowest odds = rank 1
    # NaN for races without odds data
    df["popularity"] = df.groupby("race_id")["tansho_odds"].rank(method="min").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Leak-safe cumulative helpers
# ---------------------------------------------------------------------------


def _cumulative_rate_intraday(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> np.ndarray:
    """Cumulative rate WITH intentional intraday leakage.

    Includes same-day earlier races in the group (unlike _cumulative_rate).
    Used for "thin leakage" features that improve tree structure during training,
    then neutralized at evaluation/prediction time.

    The leakage: at race 8, this feature knows outcomes of races 1-7 today
    at the same (stadium, course) — subtly encodes today's conditions.
    """
    cum_all = df.groupby(group_cols, sort=False)[value_col].cumsum()
    count_all = df.groupby(group_cols, sort=False).cumcount() + 1

    # Subtract only current row (NOT same-day, so intraday data remains)
    prior = cum_all - df[value_col]
    prior_count = count_all - 1

    return np.where(prior_count > 0, prior / prior_count, np.nan)


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
# Rolling window helpers (race-day granularity)
# ---------------------------------------------------------------------------


def _rolling_mean_daily(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int = 10,
) -> np.ndarray:
    """Rolling mean over last N race-days, same-day excluded.

    Aggregates to (group, race_date) level, uses cumsum+shift trick
    for O(n) vectorized computation (no per-group Python lambdas).
    NaN values in value_col are excluded from sums and counts.
    """
    valid = df[value_col].notna()
    val = df[value_col].fillna(0.0)

    # Build daily summary
    tmp = df[group_cols + ["race_date"]].copy()
    tmp["_val"] = val
    tmp["_valid"] = valid.astype(int)

    daily = (
        tmp.groupby(group_cols + ["race_date"], sort=False)
        .agg(_dsum=("_val", "sum"), _dcnt=("_valid", "sum"))
        .reset_index()
        .sort_values(group_cols + ["race_date"])
    )

    # Vectorized rolling via cumsum + shift (all C-level, no Python lambdas)
    g = daily.groupby(group_cols, sort=False)
    cs_sum = g["_dsum"].cumsum()
    cs_cnt = g["_dcnt"].cumsum()

    # rolling_sum at day i = sum of last `window` days before day i
    #   = cumsum[i-1] - cumsum[i-1-window]
    # groupby.shift respects group boundaries (C-level, fast)
    g2 = daily.groupby(group_cols, sort=False)
    s1_sum = g2[cs_sum.name if hasattr(cs_sum, "name") else "_dsum"].shift(1)
    # Need to use the cumsum series directly; assign to daily first
    daily["_cs_sum"] = cs_sum.values
    daily["_cs_cnt"] = cs_cnt.values

    g3 = daily.groupby(group_cols, sort=False)
    rsum = g3["_cs_sum"].shift(1).fillna(0) - g3["_cs_sum"].shift(1 + window).fillna(0)
    rcnt = g3["_cs_cnt"].shift(1).fillna(0) - g3["_cs_cnt"].shift(1 + window).fillna(0)

    daily["_rmean"] = np.where(rcnt > 0, rsum / rcnt, np.nan)

    # Merge back to entry level
    merge_keys = group_cols + ["race_date"]
    result = df[merge_keys].merge(
        daily[merge_keys + ["_rmean"]],
        on=merge_keys,
        how="left",
    )
    return result["_rmean"].values


def _rolling_rate_daily(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int = 10,
) -> np.ndarray:
    """Rolling rate (0/1 indicator) over last N race-days, same-day excluded."""
    return _rolling_mean_daily(df, group_cols, value_col, window)


# ---------------------------------------------------------------------------
# Tournament ID generation
# ---------------------------------------------------------------------------


def _generate_tournament_id(df: pd.DataFrame) -> None:
    """Generate tournament_id from (stadium_id, race_title, consecutive dates).

    A tournament is a group of races at the same stadium with the same title
    on consecutive dates. A gap of 2+ days starts a new tournament.
    """
    # Get unique (stadium_id, race_title, race_date) combos, sorted
    race_info = (
        df[["stadium_id", "race_title", "race_date"]]
        .drop_duplicates()
        .sort_values(["stadium_id", "race_title", "race_date"])
        .reset_index(drop=True)
    )

    dates = pd.to_datetime(race_info["race_date"])
    tid = 0
    tournament_ids = []
    prev_key = None

    for i, row in race_info.iterrows():
        key = (row["stadium_id"], row["race_title"])
        if key != prev_key:
            tid += 1
        elif (dates.iloc[i] - dates.iloc[i - 1]).days > 1:
            tid += 1
        tournament_ids.append(tid)
        prev_key = key

    race_info["tournament_id"] = tournament_ids

    # Merge back to df
    df_merged = df.merge(
        race_info[["stadium_id", "race_title", "race_date", "tournament_id"]],
        on=["stadium_id", "race_title", "race_date"],
        how="left",
    )
    df["tournament_id"] = df_merged["tournament_id"].values


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


def _add_stadium_course_stats(df: pd.DataFrame) -> None:
    """B7: Stadium × course win rate (leak-safe cumulative).

    Captures venue-specific course advantages (e.g., Toda 1-course 44% vs Omura 64%).
    Varies per boat because each boat is at a different course.
    """
    df["_is_win"] = (df["finish_position"] == 1).astype(int)
    group = ["stadium_id", "course_number"]
    df["stadium_course_win_rate"] = _cumulative_rate(df, group, "_is_win")


def _add_course_taking_rate(df: pd.DataFrame) -> None:
    """B2: Rate of racer taking an inner course than their boat number."""
    df["_took_inner"] = (df["course_number"] < df["boat_number"]).astype(int)
    df["course_taking_rate"] = _cumulative_rate(df, ["racer_id"], "_took_inner")


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


ROLLING_WINDOW: int = 5    # race-days; default for per-racer features
ROLLING_WINDOW_GENERAL: int | None = None  # per-racer override (None = use ROLLING_WINDOW)
ROLLING_WINDOW_COURSE: int | None = 20     # per-racer-course features (wider for sample size)


def _add_rolling_features(df: pd.DataFrame) -> None:
    """B10: Rolling features over recent race-days.

    Captures short-term form that cumulative stats miss.
    Start timing from previous races is available pre-race (historical data).

    Window sizes:
        ROLLING_WINDOW_GENERAL (default 5): per-racer features (ST, position, win rate)
        ROLLING_WINDOW_COURSE (default 20): per-racer-course features (course win rate, ST)
    """
    w_general = ROLLING_WINDOW_GENERAL or ROLLING_WINDOW
    w_course = ROLLING_WINDOW_COURSE or ROLLING_WINDOW
    group = ["racer_id"]

    # Recent start timing (actual race ST from past races)
    df["rolling_st_mean"] = _rolling_mean_daily(df, group, "start_timing", w_general)

    # Recent finish position (short-term form)
    df["_pos"] = df["finish_position"].astype(float)
    df["rolling_avg_position"] = _rolling_mean_daily(df, group, "_pos", w_general)

    # Recent win rate
    df["_is_win"] = (df["finish_position"] == 1).astype(float)
    df["rolling_win_rate"] = _rolling_rate_daily(df, group, "_is_win", w_general)

    # Course-specific rolling: recent performance at this course position
    course_group = ["racer_id", "course_number"]
    df["rolling_course_win_rate"] = _rolling_rate_daily(
        df, course_group, "_is_win", w_course
    )
    df["rolling_course_st"] = _rolling_mean_daily(
        df, course_group, "start_timing", w_course
    )


def _add_tournament_features(df: pd.DataFrame) -> None:
    """B8: Tournament-scoped features (within same 開催).

    Captures motor development and racer adaptation within a tournament.
    Exhibition time trends are less noisy than finish positions (measured every race).
    """
    group = ["racer_id", "tournament_id"]

    # Exhibition time trend: today vs tournament prior days
    # Negative = boat getting faster = motor tuning working
    df["tourn_exhibition_delta"] = (
        df["exhibition_time"]
        - _cumulative_mean(df, group, "exhibition_time")
    )

    # Exhibition ST trend: today vs tournament prior days
    # Negative = starting earlier = racer adapting to conditions
    df["tourn_st_delta"] = (
        df["exhibition_st"]
        - _cumulative_mean(df, group, "exhibition_st")
    )

    # Average position within tournament (current form with this motor)
    df["_pos"] = df["finish_position"].astype(float)
    df["tourn_avg_position"] = _cumulative_mean(df, group, "_pos")


# ---------------------------------------------------------------------------
# Leaked features (thin leakage for tree structure improvement)
# ---------------------------------------------------------------------------


def _add_leaked_features(df: pd.DataFrame) -> None:
    """Intentionally leaked features for training-time tree structure improvement.

    These features include intraday race outcomes (same-day earlier races at
    the same stadium), providing subtle hints about today's conditions.
    At evaluation/prediction time, they are neutralized (replaced with per-race
    mean) so LambdaRank cannot use them for within-race discrimination.

    Pattern from tateyamakun: learn "what upsets look like" during training,
    then force the model to approximate that knowledge from non-leaked features.
    """
    df["_is_win"] = (df["finish_position"] == 1).astype(int)

    # gate_bias: win rate per (stadium, course) with intraday leakage
    # Captures venue × course advantage INCLUDING today's pattern
    df["gate_bias"] = _cumulative_rate_intraday(
        df, ["stadium_id", "course_number"], "_is_win"
    )

    # upset_rate: 1番人気の敗北率 per (stadium_id, course_number) with intraday leakage
    # Uses real odds-derived popularity when available, falls back to
    # national_win_rate proxy for races without odds data.
    has_odds = df["popularity"].notna()
    is_fav_odds = (df["popularity"].fillna(0) == 1).astype(int)
    # Fallback: highest national_win_rate in race = pseudo-favorite
    n_races = len(df) // 6
    nwr = df["national_win_rate"].fillna(0).values.reshape(n_races, 6)
    is_fav_fallback = np.zeros(len(df), dtype=int)
    is_fav_fallback[(np.arange(n_races) * 6 + np.argmax(nwr, axis=1))] = 1
    # Use odds-based when available, fallback otherwise
    df["_is_fav"] = np.where(has_odds, is_fav_odds, is_fav_fallback)
    df["_fav_won"] = (df["_is_fav"] & (df["finish_position"] == 1)).astype(int)

    # Per (stadium, course): cumulative favorite win rate with intraday leakage
    group = ["stadium_id", "course_number"]
    cum_fav_won = df.groupby(group, sort=False)["_fav_won"].cumsum()
    cum_fav = df.groupby(group, sort=False)["_is_fav"].cumsum()
    prior_fav_won = cum_fav_won - df["_fav_won"]
    prior_fav = cum_fav - df["_is_fav"]
    fav_wr = np.where(prior_fav > 0, prior_fav_won / prior_fav, 0.5)
    df["upset_rate"] = 1.0 - fav_wr

    df.drop(columns=["_is_fav", "_fav_won"], inplace=True)



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

_TEMP_COLS = ["_is_win", "_is_top2", "_is_top3", "_took_inner", "_pos", "tournament_id"]


def _cleanup_temp_cols(df: pd.DataFrame) -> None:
    """Remove temporary computation columns."""
    for col in _TEMP_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"
_CACHE_FILE = _CACHE_DIR / "features.pkl"
_CACHE_KEY_FILE = _CACHE_DIR / "features.key"


def _db_cache_key(db_path: str) -> str:
    """Generate a cache key from DB mtime + size + code mtime."""
    st = os.stat(db_path)
    code_mtime = os.stat(__file__).st_mtime_ns
    raw = f"{db_path}:{st.st_mtime_ns}:{st.st_size}:{code_mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cached_base(db_path: str) -> pd.DataFrame | None:
    """Load cached base DataFrame if DB hasn't changed."""
    if not _CACHE_FILE.exists() or not _CACHE_KEY_FILE.exists():
        return None
    stored_key = _CACHE_KEY_FILE.read_text().strip()
    if stored_key != _db_cache_key(db_path):
        return None
    print("Loading features from cache...")
    t0 = time.time()
    df = pd.read_pickle(_CACHE_FILE)
    print(f"  Loaded {len(df)} entries from cache ({time.time() - t0:.1f}s)")
    return df


def _save_cache(df: pd.DataFrame, db_path: str) -> None:
    """Save base DataFrame to pickle cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(_CACHE_FILE)
    _CACHE_KEY_FILE.write_text(_db_cache_key(db_path))


def build_features_df(
    db_path: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build the full feature DataFrame with all columns retained.

    Same pipeline as build_features() but returns the raw DataFrame
    before FEATURE_COLS filtering. Used by boat1 binary classifier
    which needs access to columns beyond the ranking model's feature set.

    Caches the base DataFrame (pre-filter) as pickle. Cache is
    invalidated when the DB file or feature code is modified.

    Args:
        db_path: Path to the SQLite database.
        start_date: Include races on or after this date (YYYY-MM-DD).
        end_date: Include races before this date (YYYY-MM-DD).

    Returns:
        DataFrame with all computed features (6 rows per race).
    """
    # Try loading from cache (pre-filter base)
    df = _load_cached_base(db_path)

    if df is None:
        conn = get_connection(db_path)

        try:
            print("Loading data from database...")
            df = _load_all_data(conn)
            print(f"  Loaded {len(df)} entries")
        finally:
            conn.close()

        # Generate tournament ID for tournament-scoped features
        _generate_tournament_id(df)

        # Historical features (must be computed on full dataset before filtering)
        print("Computing historical features...")
        _add_racer_course_stats(df)
        _add_stadium_course_stats(df)
        _add_course_taking_rate(df)
        _add_recent_form(df)
        _add_st_stability(df)
        _add_rolling_features(df)
        _add_tournament_features(df)
        _add_leaked_features(df)

        _cleanup_temp_cols(df)
        _encode_categoricals(df)

        _save_cache(df, db_path)
        print("  Saved features to cache")

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

    return df


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
    df = build_features_df(db_path, start_date=start_date, end_date=end_date)
    return prepare_feature_matrix(df)
