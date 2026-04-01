"""Single source of truth for feature column definitions.

Feature order matters for model compatibility — append new features at the end
of each section, never reorder existing ones.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Feature columns (order matters for model compatibility)
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # --- Race conditions (4) ---
    "stadium_id",
    "race_number",
    "race_grade_code",
    "race_month",
    # --- Weather / environment (6) ---
    "wind_speed",
    "wind_direction",
    "wave_height",
    "temperature",
    "water_temperature",
    "weather_code",
    # --- Boat assignment (1) ---
    "boat_number",
    # --- Racer attributes (5) ---
    "racer_class_code",
    "racer_weight",
    "flying_count",
    "late_count",
    "average_st",
    # --- National performance (3) ---
    "national_win_rate",
    "national_top2_rate",
    "national_top3_rate",
    # --- Local (stadium) performance (3) ---
    "local_win_rate",
    "local_top2_rate",
    "local_top3_rate",
    # --- Equipment: motor (2) ---
    "motor_top2_rate",
    "motor_top3_rate",
    # --- Equipment: boat (2) ---
    "boat_top2_rate",
    "boat_top3_rate",
    # --- Exhibition data (4) ---
    "exhibition_time",
    "exhibition_st",
    "tilt",
    "stabilizer",
    # --- Course-specific racer history (3) ---
    "racer_course_win_rate",
    "racer_course_top2_rate",
    "racer_course_top3_rate",
    # --- Course behavior (2) ---
    "course_taking_rate",
    "course_avg_st",
    # --- Recent form (3) ---
    "recent_win_rate",
    "recent_top2_rate",
    "recent_avg_position",
    # --- Motor actual performance (2) ---
    "motor_actual_win_rate",
    "motor_actual_top2_rate",
    # --- Start stability (1) ---
    "st_stability",
    # --- Race-relative z-scores (4) ---
    "rel_national_win_rate",
    "rel_exhibition_time",
    "rel_motor_top2_rate",
    "rel_average_st",
    # --- Interaction features (4) ---
    "class_x_boat",
    "motor_x_boat",
    "wind_x_wave",
    "weight_x_boat",
]

# Columns that need 0-fill (not NaN) for LightGBM categorical handling
CATEGORICAL_FILL_COLS: list[str] = [
    "stadium_id",
    "race_grade_code",
    "weather_code",
    "racer_class_code",
]

# Relevance label schemes for LambdaRank
RELEVANCE_SCHEMES: dict[str, dict[int, float]] = {
    "linear": {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
    "top_heavy": {1: 10, 2: 6, 3: 3, 4: 2, 5: 1, 6: 0},
    "podium": {1: 5, 2: 3, 3: 1, 4: 0, 5: 0, 6: 0},
    "win_only": {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
}


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------


def encode_race_grade(grade: str | None) -> int:
    """Race grade to ordinal code: SG=5, G1=4, G2=3, G3=2, 一般=1, unknown=0."""
    mapping = {"SG": 5, "G1": 4, "G2": 3, "G3": 2, "一般": 1}
    return mapping.get(grade or "", 0)


def encode_racer_class(cls: str | None) -> int:
    """Racer class to ordinal code: A1=4, A2=3, B1=2, B2=1, unknown=0."""
    mapping = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    return mapping.get(cls or "", 0)


def encode_weather(weather: str | None) -> int:
    """Weather to code: 晴=1, 曇り=2, 雨=3, 雪=4, 霧=5, unknown=0."""
    mapping = {"晴": 1, "曇り": 2, "雨": 3, "雪": 4, "霧": 5}
    return mapping.get(weather or "", 0)


# ---------------------------------------------------------------------------
# Within-race z-scoring
# ---------------------------------------------------------------------------


def _race_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score a column within each race group."""
    grouped = df.groupby("race_id")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, 1)
    return (df[col] - mean) / std


def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race-relative z-scored features."""
    df = df.copy()
    df["rel_national_win_rate"] = _race_zscore(df, "national_win_rate")
    df["rel_exhibition_time"] = _race_zscore(df, "exhibition_time")
    df["rel_motor_top2_rate"] = _race_zscore(df, "motor_top2_rate")
    df["rel_average_st"] = _race_zscore(df, "average_st")
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features combining multiple signals."""
    df = df.copy()
    # Class advantage at inner courses (A1 racer in course 1 is dominant)
    df["class_x_boat"] = df["racer_class_code"] * (7 - df["boat_number"])
    # Motor quality at inner courses
    df["motor_x_boat"] = df["motor_top2_rate"] * (7 - df["boat_number"])
    # Wind and wave combined roughness
    df["wind_x_wave"] = df["wind_speed"] * df["wave_height"]
    # Lighter racers advantage at outer courses
    df["weight_x_boat"] = df["racer_weight"] * df["boat_number"]
    return df


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------


def prepare_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract feature matrix X, target y, and metadata from merged DataFrame.

    Returns:
        X: Feature matrix with FEATURE_COLS
        y: finish_position (target for relevance labeling)
        meta: race_id, racer_id, race_date, boat_number (for splitting/evaluation)
    """
    X = df[FEATURE_COLS].copy()

    for col in CATEGORICAL_FILL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)

    y = df["finish_position"]
    meta = df[["race_id", "racer_id", "race_date", "boat_number"]].copy()

    return X, y, meta
