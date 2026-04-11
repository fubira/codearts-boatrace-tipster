"""Fast feature construction from a pre-built stats snapshot.

Produces the same DataFrame as build_features_df() but in ~1-2 seconds
instead of 30-60 seconds by avoiding the full 760k-row DB scan.

The snapshot contains:
  - cumulative_stats: per-group (sum, count) accumulated through yesterday
  - motor_stats: per-(stadium, motor) residual sufficient statistics
  - rolling_daily: per-(group, date) daily aggregates for rolling windows
Tournament features are computed on-the-fly from a small DB query.
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
)
from .features import (
    NON_FINISHER_RANK,
    ROLLING_WINDOW,
    ROLLING_WINDOW_COURSE,
    ROLLING_WINDOW_GENERAL,
    _BASE_QUERY,
)
from .snapshot import load_snapshot

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features_from_snapshot(
    db_path: str,
    cache_path: str,
    target_date: str,
) -> pd.DataFrame:
    """Build features for a single date using a pre-built snapshot.

    Produces a DataFrame identical to build_features_df(db_path,
    start_date=target_date, end_date=next_day).

    Args:
        db_path: Path to the main SQLite database.
        cache_path: Path to the snapshot SQLite file.
        target_date: The date to predict (YYYY-MM-DD).

    Returns:
        DataFrame with all computed features (6 rows per race).
    """
    snapshot = load_snapshot(cache_path)

    # Load today's entries from main DB (single date only)
    next_day = (
        pd.Timestamp(target_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Add date filter via parameter binding for safe, fast loading
    filtered_query = _BASE_QUERY.replace(
        "ORDER BY r.race_date, r.id, re.id",
        "WHERE r.race_date >= $1 AND r.race_date < $2\nORDER BY r.race_date, r.id, re.id",
    )

    conn = get_connection(db_path)
    try:
        df = conn.execute(filtered_query, [target_date, next_day]).fetchdf()
    finally:
        conn.close()

    # Apply same preprocessing as _load_all_data
    df = df.sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)
    df["finish_position"] = df["finish_position"].fillna(NON_FINISHER_RANK).astype(int)

    race_counts = df.groupby("race_id").size()
    valid_races = race_counts[race_counts == 6].index
    df = df[df["race_id"].isin(valid_races)].reset_index(drop=True)
    df["popularity"] = (
        df.groupby("race_id")["tansho_odds"].rank(method="min").astype("Int64")
    )

    if len(df) == 0:
        return df

    # --- Cumulative features ---
    _apply_cumulative_rate(df, snapshot, "racer_course_win", ["racer_id", "course_number"], "racer_course_win_rate")
    _apply_cumulative_rate(df, snapshot, "racer_course_top2", ["racer_id", "course_number"], "racer_course_top2_rate")
    _apply_cumulative_rate(df, snapshot, "racer_course_top3", ["racer_id", "course_number"], "racer_course_top3_rate")
    _apply_cumulative_rate(df, snapshot, "stadium_course_win", ["stadium_id", "course_number"], "stadium_course_win_rate")
    _apply_cumulative_rate(df, snapshot, "racer_course_taking", ["racer_id"], "course_taking_rate")
    _apply_cumulative_mean(df, snapshot, "racer_avg_course_diff", ["racer_id"], "avg_course_diff")
    _apply_cumulative_rate(df, snapshot, "racer_boat_course_taking", ["racer_id", "boat_number"], "course_taking_rate_at_boat")
    _apply_cumulative_rate(df, snapshot, "racer_recent_win", ["racer_id"], "recent_win_rate")
    _apply_cumulative_rate(df, snapshot, "racer_recent_top2", ["racer_id"], "recent_top2_rate")

    _apply_cumulative_mean(df, snapshot, "racer_avg_position", ["racer_id"], "recent_avg_position")
    _apply_cumulative_mean(df, snapshot, "racer_boat_avg_st", ["racer_id", "boat_number"], "course_avg_st")

    # Self-comparison: exhibition_time - racer's cumulative mean
    _apply_cumulative_mean(df, snapshot, "racer_avg_exhibition", ["racer_id"], "_racer_avg_ex")
    df["self_exhibition_delta"] = df["exhibition_time"] - df["_racer_avg_ex"]

    _apply_cumulative_mean(df, snapshot, "racer_avg_exhibition_st", ["racer_id"], "_racer_avg_exst")
    df["self_st_delta"] = df["exhibition_st"] - df["_racer_avg_exst"]

    # ST stability (cumulative std)
    _apply_cumulative_std(df, snapshot, "racer_st_stability", ["racer_id"], "st_stability")

    # Position alpha (skill net of lane advantage)
    _apply_cumulative_mean(df, snapshot, "racer_position_alpha", ["racer_id"], "position_alpha")

    # --- Motor quality residual ---
    _apply_motor_residual(df, snapshot)

    # --- Rolling features ---
    w_general = ROLLING_WINDOW_GENERAL or ROLLING_WINDOW
    w_course = ROLLING_WINDOW_COURSE or ROLLING_WINDOW

    _apply_rolling(df, snapshot, "racer_st", ["racer_id"], "rolling_st_mean", w_general)
    _apply_rolling(df, snapshot, "racer_position", ["racer_id"], "rolling_avg_position", w_general)
    _apply_rolling(df, snapshot, "racer_win", ["racer_id"], "rolling_win_rate", w_general)
    _apply_rolling(df, snapshot, "racer_course_win", ["racer_id", "course_number"], "rolling_course_win_rate", w_course)
    _apply_rolling(df, snapshot, "racer_course_st", ["racer_id", "course_number"], "rolling_course_st", w_course)
    _apply_rolling(df, snapshot, "racer_position_alpha", ["racer_id"], "rolling_position_alpha", w_general)

    # --- Tournament features (on-the-fly from DB) ---
    _apply_tournament_features(df, db_path, target_date)

    # --- Leaked features (not used, but keep columns for compatibility) ---
    df["gate_bias"] = np.nan
    df["upset_rate"] = np.nan

    # --- Cleanup temp columns ---
    for col in ["_racer_avg_ex", "_racer_avg_exst"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # --- Encodings ---
    df["race_grade_code"] = df["race_grade"].map(encode_race_grade)
    df["racer_class_code"] = df["racer_class"].map(encode_racer_class)
    df["weather_code"] = df["weather"].map(encode_weather)

    # --- Relative and interaction features ---
    df = compute_relative_features(df)
    df = compute_interaction_features(df)

    return df


# ---------------------------------------------------------------------------
# Cumulative feature application
# ---------------------------------------------------------------------------


def _make_key(row: pd.Series, group_cols: list[str]) -> str:
    if len(group_cols) == 1:
        return str(int(row[group_cols[0]]))
    return ":".join(str(int(row[c])) for c in group_cols)


def _apply_cumulative_rate(
    df: pd.DataFrame,
    snapshot: dict,
    stat_name: str,
    group_cols: list[str],
    output_col: str,
) -> None:
    """Look up cumulative rate from snapshot."""
    cum = snapshot["cumulative"]
    values = np.full(len(df), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        key = _make_key(row, group_cols)
        entry = cum.get((stat_name, key))
        if entry is not None:
            total_sum, total_count, _ = entry
            if total_count > 0:
                values[i] = total_sum / total_count

    df[output_col] = values


def _apply_cumulative_mean(
    df: pd.DataFrame,
    snapshot: dict,
    stat_name: str,
    group_cols: list[str],
    output_col: str,
) -> None:
    """Look up cumulative mean from snapshot (same as rate, but for continuous)."""
    _apply_cumulative_rate(df, snapshot, stat_name, group_cols, output_col)


def _apply_cumulative_std(
    df: pd.DataFrame,
    snapshot: dict,
    stat_name: str,
    group_cols: list[str],
    output_col: str,
) -> None:
    """Compute cumulative std from snapshot's (sum, count, sum_sq)."""
    cum = snapshot["cumulative"]
    values = np.full(len(df), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        key = _make_key(row, group_cols)
        entry = cum.get((stat_name, key))
        if entry is not None:
            total_sum, total_count, total_sum_sq = entry
            if total_count >= 2 and total_sum_sq is not None:
                mean = total_sum / total_count
                var = total_sum_sq / total_count - mean ** 2
                var = max(var, 0)  # numerical safety
                values[i] = np.sqrt(var)

    df[output_col] = values


# ---------------------------------------------------------------------------
# Motor residual
# ---------------------------------------------------------------------------


def _apply_motor_residual(df: pd.DataFrame, snapshot: dict) -> None:
    """Look up motor quality residual from snapshot."""
    motor = snapshot["motor"]
    values = np.full(len(df), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        entry = motor.get((int(row["stadium_id"]), int(row["motor_number"])))
        if entry is not None:
            residual_sum, residual_count = entry
            if residual_count > 0:
                values[i] = residual_sum / residual_count

    df["motor_quality_residual"] = values


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


def _apply_rolling(
    df: pd.DataFrame,
    snapshot: dict,
    stat_name: str,
    group_cols: list[str],
    output_col: str,
    window: int,
) -> None:
    """Compute rolling mean from snapshot's daily aggregates."""
    rolling = snapshot["rolling"]
    values = np.full(len(df), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        key = _make_key(row, group_cols)
        days = rolling.get((stat_name, key))
        if days is None:
            continue

        # Sum last `window` race-days (all days are before target_date
        # because snapshot was built through yesterday)
        total_sum = 0.0
        total_count = 0
        n_days = 0
        for date, day_sum, day_count in reversed(days):
            if n_days >= window:
                break
            total_sum += day_sum
            total_count += day_count
            n_days += 1

        if total_count > 0:
            values[i] = total_sum / total_count

    df[output_col] = values


# ---------------------------------------------------------------------------
# Tournament features (on-the-fly from DB)
# ---------------------------------------------------------------------------


def _apply_tournament_features(
    df: pd.DataFrame, db_path: str, target_date: str,
) -> None:
    """Compute tournament-scoped features from DB.

    Queries prior days of the same tournament (same stadium + race_title
    on consecutive dates) and computes cumulative means.
    """
    df["tourn_exhibition_delta"] = np.nan
    df["tourn_st_delta"] = np.nan
    df["tourn_avg_position"] = np.nan
    df["prev_day_exhibition_delta"] = np.nan

    if len(df) == 0:
        return

    # Get unique (stadium_id, race_title) for today's races
    race_info = df[["stadium_id", "race_title"]].drop_duplicates()

    # Compute date range for tournament lookup (10 days back)
    date_from = (pd.Timestamp(target_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    conn = get_connection(db_path)
    try:
        for _, info in race_info.iterrows():
            sid = int(info["stadium_id"])
            title = str(info["race_title"])

            # Find prior days of this tournament: same stadium + title,
            # dates before target_date, within last 10 days (tournaments are 3-7 days)
            prior_query = """
                SELECT re.racer_id, r.race_date,
                       re.exhibition_time, re.exhibition_st, re.finish_position
                FROM db.races r
                JOIN db.race_entries re ON re.race_id = r.id
                WHERE r.stadium_id = ?
                  AND r.race_title = ?
                  AND r.race_date < ?
                  AND r.race_date >= ?
                ORDER BY r.race_date
            """
            prior = conn.execute(prior_query, [sid, title, target_date, date_from]).fetchdf()

            if len(prior) == 0:
                continue

            # Fill non-finisher positions (same as _load_all_data)
            prior["finish_position"] = prior["finish_position"].fillna(NON_FINISHER_RANK)

            # Check date continuity (gap > 1 day = different tournament)
            prior_dates = sorted(prior["race_date"].unique())
            tournament_dates = []
            for d in reversed(prior_dates):
                if not tournament_dates:
                    tournament_dates.append(d)
                else:
                    prev = pd.Timestamp(tournament_dates[-1])
                    curr = pd.Timestamp(d)
                    if (prev - curr).days <= 1:
                        tournament_dates.append(d)
                    else:
                        break
            tournament_dates.reverse()

            if not tournament_dates:
                continue

            # Filter prior to tournament dates only
            prior = prior[prior["race_date"].isin(tournament_dates)]

            # Compute per-racer tournament stats
            racer_stats = (
                prior.groupby("racer_id")
                .agg(
                    ex_mean=("exhibition_time", "mean"),
                    exst_mean=("exhibition_st", "mean"),
                    pos_mean=("finish_position", "mean"),
                )
            )

            # Previous day stats
            last_date = tournament_dates[-1]
            prev_day = prior[prior["race_date"] == last_date]
            prev_day_ex = prev_day.groupby("racer_id")["exhibition_time"].mean()

            # Apply to today's entries at this stadium/title
            mask = (df["stadium_id"] == sid) & (df["race_title"] == title)
            for idx in df.index[mask]:
                rid = int(df.at[idx, "racer_id"])
                if rid in racer_stats.index:
                    stats = racer_stats.loc[rid]
                    ex_val = df.at[idx, "exhibition_time"]
                    exst_val = df.at[idx, "exhibition_st"]

                    if pd.notna(ex_val) and pd.notna(stats["ex_mean"]):
                        df.at[idx, "tourn_exhibition_delta"] = ex_val - stats["ex_mean"]
                    if pd.notna(exst_val) and pd.notna(stats["exst_mean"]):
                        df.at[idx, "tourn_st_delta"] = exst_val - stats["exst_mean"]
                    if pd.notna(stats["pos_mean"]):
                        df.at[idx, "tourn_avg_position"] = stats["pos_mean"]

                if rid in prev_day_ex.index:
                    ex_val = df.at[idx, "exhibition_time"]
                    if pd.notna(ex_val) and pd.notna(prev_day_ex[rid]):
                        df.at[idx, "prev_day_exhibition_delta"] = ex_val - prev_day_ex[rid]
    finally:
        conn.close()
