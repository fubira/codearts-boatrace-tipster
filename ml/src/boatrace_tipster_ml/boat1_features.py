"""Reshape 6-row-per-race data to 1-row-per-race for boat 1 binary classification.

Extracts boat 1 features, computes opponent aggregates, and builds
gap features for predicting whether boat 1 (1号艇) wins.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature column definition for boat 1 binary classifier
# ---------------------------------------------------------------------------

BOAT1_FEATURE_COLS: list[str] = [
    # --- A: Boat 1's own strength (~18) ---
    "b1_national_win_rate",
    "b1_national_top2_rate",
    "b1_national_top3_rate",
    "b1_local_win_rate",
    "b1_local_top3_rate",
    "b1_motor_top3_rate",
    "b1_exhibition_time",
    "b1_racer_course_win_rate",
    "b1_racer_course_top2_rate",
    "b1_recent_avg_position",
    "b1_stadium_course_win_rate",
    "b1_racer_class_code",
    "b1_racer_weight",
    "b1_average_st",
    "b1_st_stability",
    "b1_rolling_st_mean",
    "b1_rolling_win_rate",
    "b1_rolling_avg_position",
    "b1_rolling_course_win_rate",
    "b1_tourn_exhibition_delta",
    "b1_tourn_st_delta",
    "b1_tourn_avg_position",
    "b1_self_exhibition_delta",
    "b1_self_st_delta",
    # --- B: Relative position (z-scored within race) (3) ---
    "b1_rel_national_win_rate",
    "b1_rel_exhibition_time",
    "b1_rel_exhibition_st",
    # --- C: Opponent aggregates (8) ---
    "opp_max_national_win_rate",
    "opp_mean_national_win_rate",
    "opp_spread_national_win_rate",
    "opp_best_exhibition_time",
    "opp_best_exhibition_st",
    "opp_max_motor_top3_rate",
    "opp_n_a1",
    "opp_max_racer_course_win_rate",
    # --- D: Race context (6) ---
    "stadium_id",
    "race_grade_code",
    "weather_code",
    "wind_speed",
    "wave_height",
    "has_front_taking",
    # --- E: Gap features (2) ---
    "b1_vs_best_opp_win_rate",
    "b1_vs_best_opp_exhibition",
    # --- F: Kado threat (1) ---
    "opp_kado_exhibition_time",
]

# Columns from the full DataFrame to extract as boat 1 features (raw name → b1_ name)
_B1_EXTRACT_COLS: list[str] = [
    "national_win_rate",
    "national_top2_rate",
    "national_top3_rate",
    "local_win_rate",
    "local_top3_rate",
    "motor_top3_rate",
    "exhibition_time",
    "exhibition_st",
    "racer_course_win_rate",
    "racer_course_top2_rate",
    "recent_avg_position",
    "stadium_course_win_rate",
    "racer_class_code",
    "racer_weight",
    "average_st",
    "st_stability",
    "rolling_st_mean",
    "rolling_win_rate",
    "rolling_avg_position",
    "rolling_course_win_rate",
    "tourn_exhibition_delta",
    "tourn_st_delta",
    "tourn_avg_position",
    "self_exhibition_delta",
    "self_st_delta",
    # Relative features (already z-scored)
    "rel_national_win_rate",
    "rel_exhibition_time",
    "rel_exhibition_st",
]

FIELD_SIZE = 6


def reshape_to_boat1(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Reshape 6-row-per-race DataFrame to 1-row-per-race for boat 1 prediction.

    Args:
        df: Full feature DataFrame from build_features_df() (6 rows per race).

    Returns:
        (X_b1, y_b1, meta_b1) where:
            X_b1: Feature matrix (1 row per race, BOAT1_FEATURE_COLS columns)
            y_b1: Binary target (1 = boat 1 won, 0 = boat 1 lost)
            meta_b1: race_id, race_date (for time-series splitting)
    """
    # Separate boat 1 and opponents
    b1 = df[df["boat_number"] == 1].reset_index(drop=True)
    opps = df[df["boat_number"] != 1]

    # Target: did boat 1 win?
    y_b1 = (b1["finish_position"] == 1).astype(int)

    # --- A+B: Boat 1's own features ---
    result = pd.DataFrame({"race_id": b1["race_id"].values})
    for col in _B1_EXTRACT_COLS:
        if col in b1.columns:
            result[f"b1_{col}"] = b1[col].values

    # --- C: Opponent aggregates ---
    opp_agg = opps.groupby("race_id", sort=False).agg(
        opp_max_national_win_rate=("national_win_rate", "max"),
        opp_mean_national_win_rate=("national_win_rate", "mean"),
        opp_spread_national_win_rate=("national_win_rate", "std"),
        opp_best_exhibition_time=("exhibition_time", "min"),
        opp_best_exhibition_st=("exhibition_st", "min"),
        opp_max_motor_top3_rate=("motor_top3_rate", "max"),
        opp_max_racer_course_win_rate=("racer_course_win_rate", "max"),
    )
    # Count of A1-class opponents
    opp_a1 = opps.groupby("race_id", sort=False)["racer_class_code"].apply(
        lambda x: (x == 4).sum()
    ).rename("opp_n_a1")
    opp_agg = opp_agg.join(opp_a1)

    result = result.merge(opp_agg, on="race_id", how="left")

    # --- D: Race-level context ---
    for col in ["stadium_id", "race_grade_code", "weather_code", "wind_speed", "wave_height", "has_front_taking"]:
        if col in b1.columns:
            result[col] = b1[col].values

    # --- E: Gap features ---
    result["b1_vs_best_opp_win_rate"] = (
        result["b1_national_win_rate"] - result["opp_max_national_win_rate"]
    )
    result["b1_vs_best_opp_exhibition"] = (
        result["b1_exhibition_time"] - result["opp_best_exhibition_time"]
    )

    # --- F: Kado threat ---
    # Kado = innermost dash boat (min course_number >= 4)
    result["opp_kado_exhibition_time"] = _compute_kado_exhibition(df)

    # --- Meta ---
    meta_b1 = pd.DataFrame({
        "race_id": b1["race_id"].values,
        "race_date": b1["race_date"].values,
        "b1_tansho_odds": b1["tansho_odds"].values if "tansho_odds" in b1.columns else np.nan,
    })

    # Select final feature columns
    X_b1 = result[BOAT1_FEATURE_COLS].copy()

    # Fill categorical columns with 0 for LightGBM
    for col in ["stadium_id", "race_grade_code", "weather_code"]:
        X_b1[col] = X_b1[col].fillna(0).astype(int)

    return X_b1, y_b1, meta_b1


def _compute_kado_exhibition(df: pd.DataFrame) -> np.ndarray:
    """Extract exhibition time of the kado boat (innermost dash position).

    Kado = the boat with the smallest course_number >= 4 in each race.
    Returns NaN if no dash boat exists (shouldn't happen in valid races).
    """
    n_races = len(df) // FIELD_SIZE
    course = df["course_number"].values.reshape(n_races, FIELD_SIZE)
    exhibition = df["exhibition_time"].values.reshape(n_races, FIELD_SIZE)

    kado_ex = np.full(n_races, np.nan)
    for i in range(n_races):
        dash_mask = course[i] >= 4
        if dash_mask.any():
            dash_courses = course[i][dash_mask]
            kado_idx = np.where(course[i] == dash_courses.min())[0][0]
            kado_ex[i] = exhibition[i, kado_idx]

    return kado_ex
