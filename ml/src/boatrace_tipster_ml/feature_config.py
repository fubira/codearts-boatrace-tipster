"""Single source of truth for feature column definitions.

Feature order matters for model compatibility — append new features at the end
of each section, never reorder existing ones.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature columns (order matters for model compatibility)
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # --- Boat assignment (1) ---
    "boat_number",
    # --- Racer attributes (1) ---
    "racer_weight",
    # --- National performance (3) ---
    "national_win_rate",
    "national_top2_rate",
    "national_top3_rate",
    # --- Local (stadium) performance (2) ---
    "local_win_rate",
    "local_top3_rate",
    # --- Equipment: motor (1) ---
    "motor_top3_rate",
    # --- Exhibition data (1) ---
    "exhibition_time",
    # --- Course-specific racer history (2) ---
    "racer_course_win_rate",
    "racer_course_top2_rate",
    # --- Recent form (1) ---
    "recent_avg_position",
    # --- Race-relative z-scores (2) ---
    "rel_national_win_rate",
    "rel_exhibition_time",
    # --- Stadium × course (1) ---
    "stadium_course_win_rate",
    # --- Actual course position (1) ---
    "course_number",
    # --- Exhibition pre-race (1) ---
    "rel_exhibition_st",
    # --- Start formation (1) ---
    "kado_x_exhibition",
    # --- Tournament-scoped (3) ---
    "tourn_exhibition_delta",
    "tourn_st_delta",
    "tourn_avg_position",
    # --- Interaction features (3) ---
    "class_x_boat",
    "weight_x_boat",
    "wind_speed_x_boat",
    # --- Rolling: course-specific recent form (1) ---
    "rel_rolling_course_win",
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
    df["rel_motor_top3_rate"] = _race_zscore(df, "motor_top3_rate")
    df["rel_racer_course_win_rate"] = _race_zscore(df, "racer_course_win_rate")
    df["rel_exhibition_st"] = _race_zscore(df, "exhibition_st")
    df["rel_rolling_st"] = _race_zscore(df, "rolling_st_mean")
    df["rel_rolling_win_rate"] = _race_zscore(df, "rolling_win_rate")
    df["rel_rolling_avg_pos"] = _race_zscore(df, "rolling_avg_position")
    df["rel_rolling_course_win"] = _race_zscore(df, "rolling_course_win_rate")
    df["rel_rolling_course_st"] = _race_zscore(df, "rolling_course_st")
    # Form deltas: recent rolling vs career stats (self-comparison)
    # Positive = recent worse, negative = recent better
    df["st_form_delta"] = df["rolling_st_mean"] - df["average_st"]
    # Position delta: rolling recent vs cumulative career (both are avg finish position)
    df["pos_form_delta"] = df["rolling_avg_position"] - df["recent_avg_position"]
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------


# Water surface type: 海水=3, 汽水=2, 淡水=1
# Seawater stadiums have tidal effects and rougher conditions
_WATER_TYPE: dict[int, int] = {
    1: 1,   # 桐生 — 淡水
    2: 1,   # 戸田 — 淡水
    3: 2,   # 江戸川 — 汽水
    4: 3,   # 平和島 — 海水
    5: 1,   # 多摩川 — 淡水
    6: 2,   # 浜名湖 — 汽水
    7: 2,   # 蒲郡 — 汽水
    8: 3,   # 常滑 — 海水
    9: 3,   # 津 — 海水
    10: 1,  # 三国 — 淡水
    11: 1,  # びわこ — 淡水
    12: 1,  # 住之江 — 淡水
    13: 1,  # 尼崎 — 淡水
    14: 3,  # 鳴門 — 海水
    15: 3,  # 丸亀 — 海水
    16: 3,  # 児島 — 海水
    17: 3,  # 宮島 — 海水
    18: 3,  # 徳山 — 海水
    19: 3,  # 下関 — 海水
    20: 3,  # 若松 — 海水
    21: 1,  # 芦屋 — 淡水
    22: 2,  # 福岡 — 汽水
    23: 1,  # 唐津 — 淡水
    24: 3,  # 大村 — 海水
}


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
    # Wind disruption at inner courses (strong wind weakens inner advantage)
    df["wind_speed_x_boat"] = df["wind_speed"] * (7 - df["boat_number"])
    # Water surface type × course (seawater reduces inner-course advantage)
    df["water_type_x_boat"] = df["stadium_id"].map(_WATER_TYPE) * (7 - df["boat_number"])
    # Kado × exhibition: fast boat at kado position is a major threat
    _add_kado_features(df)
    # --- Wind decomposition ---
    # wind_direction: 0=tailwind, 9=headwind, 18 directions (20° each)
    # Decompose into headwind and crosswind (left) components
    angle = df["wind_direction"].fillna(0) * (2 * np.pi / 18)
    wind_crosswind = np.sin(angle)  # positive = wind from left of course
    # Crosswind × inner course: left crosswind pushes boat 1 outward at 1st mark
    df["crosswind_x_boat"] = wind_crosswind * df["wind_speed"] * (7 - df["boat_number"])
    # --- Wave height × inner course ---
    df["wave_height_x_boat"] = df["wave_height"] * (7 - df["boat_number"])
    # --- Front-taking flag ---
    # 1 if any boat in the race has course_number != boat_number
    df["_course_changed"] = (df["course_number"] != df["boat_number"]).astype(int)
    df["has_front_taking"] = df.groupby("race_id")["_course_changed"].transform("max")
    df.drop(columns="_course_changed", inplace=True)
    return df


def _add_kado_features(df: pd.DataFrame) -> None:
    """Kado interaction: exhibition time advantage at kado position.

    Kado = innermost dash starter (min course_number >= 4 in race).
    kado_x_exhibition = is_kado * rel_exhibition_time (z-scored, negative = faster).
    Non-kado boats get 0. Captures "fast boat at kado = dangerous."
    """
    n_races = len(df) // 6
    course_2d = df["course_number"].values.reshape(n_races, 6)
    is_kado = np.zeros(len(df), dtype=int)

    for i in range(n_races):
        dash_courses = course_2d[i][course_2d[i] >= 4]
        if len(dash_courses) > 0:
            kado_course = dash_courses.min()
            offset = i * 6
            for j in range(6):
                if course_2d[i, j] == kado_course:
                    is_kado[offset + j] = 1
                    break

    df["is_kado"] = is_kado
    # Interaction: kado × relative exhibition time
    rel_ex = df.get("rel_exhibition_time", 0)
    df["kado_x_exhibition"] = is_kado * rel_ex


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
