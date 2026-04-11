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
    # --- A: Boat 1's own strength (14) ---
    "b1_local_win_rate",
    "b1_local_top3_rate",
    "b1_motor_top3_rate",
    "b1_racer_course_win_rate",
    "b1_racer_course_top2_rate",
    "b1_stadium_course_win_rate",
    "b1_racer_class_code",
    "b1_racer_weight",
    "b1_average_st",
    "b1_st_stability",
    "b1_rolling_st_mean",
    "b1_rolling_win_rate",
    "b1_rolling_course_win_rate",
    "b1_tourn_avg_position",
    # --- B: Relative position (z-scored within race) (2) ---
    "b1_rel_national_win_rate",
    "b1_rel_exhibition_time",
    # --- C: Opponent aggregates (4) ---
    "opp_max_national_win_rate",
    "opp_spread_national_win_rate",
    "opp_best_exhibition_st",
    "opp_max_racer_course_win_rate",
    # --- D: Race context (2) ---
    "wind_speed",
    "race_max_course_taking_rate",
    # --- E: Gap features (2) ---
    "b1_vs_best_opp_win_rate",
    "b1_vs_best_opp_exhibition",
    # --- F: New features (4) ---
    "opp_max_course_taking_rate",
    "headwind",
    "crosswind",
    "b1_flying_count",
    # --- G: BOATCAST exhibition z-scores (4) ---
    "b1_bc_lap_zscore",
    "b1_bc_turn_zscore",
    "b1_bc_straight_zscore",
    "b1_bc_slit_zscore",
]


# Columns from the full DataFrame to extract as boat 1 features (raw name → b1_ name)
_B1_EXTRACT_COLS: list[str] = [
    "national_win_rate",
    "local_win_rate",
    "local_top3_rate",
    "motor_top3_rate",
    "exhibition_time",
    "racer_course_win_rate",
    "racer_course_top2_rate",
    "stadium_course_win_rate",
    "racer_class_code",
    "racer_weight",
    "average_st",
    "st_stability",
    "rolling_st_mean",
    "rolling_win_rate",
    "rolling_course_win_rate",
    "tourn_avg_position",
    # Relative features (already z-scored)
    "rel_national_win_rate",
    "rel_exhibition_time",
    # Raw values needed for derived features
    "flying_count",
    # BOATCAST exhibition z-scores
    "bc_lap_zscore",
    "bc_turn_zscore",
    "bc_straight_zscore",
    "bc_slit_zscore",
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
    n_races = len(df) // FIELD_SIZE

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
        opp_spread_national_win_rate=("national_win_rate", "std"),
        opp_best_exhibition_st=("exhibition_st", "min"),
        opp_max_racer_course_win_rate=("racer_course_win_rate", "max"),
    )
    result = result.merge(opp_agg, on="race_id", how="left")

    # --- C2: Opponent course-taking rate (イン屋の脅威) ---
    opp_ct = opps.groupby("race_id", sort=False)["course_taking_rate"].max()
    result["opp_max_course_taking_rate"] = b1["race_id"].map(opp_ct.to_dict()).values

    # --- D: Race-level context ---
    for col in ["wind_speed", "race_max_course_taking_rate"]:
        if col in b1.columns:
            result[col] = b1[col].values

    # --- D2: Wind decomposition (headwind / crosswind) ---
    wind_dir = df["wind_direction"].fillna(0).values.reshape(n_races, FIELD_SIZE)[:, 0]
    wind_spd = df["wind_speed"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    angle = wind_dir * (2 * np.pi / 18)
    result["headwind"] = np.cos(angle) * wind_spd
    result["crosswind"] = np.sin(angle) * wind_spd

    # --- E: Gap features ---
    result["b1_vs_best_opp_win_rate"] = (
        result["b1_national_win_rate"] - result["opp_max_national_win_rate"]
    )
    opp_best_ex = opps.groupby("race_id", sort=False)["exhibition_time"].min()
    result["b1_vs_best_opp_exhibition"] = (
        b1["exhibition_time"].values - b1["race_id"].map(opp_best_ex.to_dict()).values
    )

    # --- Meta ---
    has_exhibition = b1["exhibition_time"].notna().values if "exhibition_time" in b1.columns else np.zeros(n_races, dtype=bool)
    meta_b1 = pd.DataFrame({
        "race_id": b1["race_id"].values,
        "race_date": b1["race_date"].values,
        "b1_tansho_odds": b1["tansho_odds"].values if "tansho_odds" in b1.columns else np.nan,
        "has_exhibition": has_exhibition,
    })

    # Select final feature columns (BOAT1_FEATURE_COLS + any additional extracted cols)
    all_cols = list(BOAT1_FEATURE_COLS)
    for col in result.columns:
        if col.startswith("b1_bc_") and col not in all_cols:
            all_cols.append(col)
    X_b1 = result[[c for c in all_cols if c in result.columns]].copy()

    return X_b1, y_b1, meta_b1
