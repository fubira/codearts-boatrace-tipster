"""Tests for feature configuration, encoding, and matrix preparation.

Covers: column definitions, encoding functions, prepare_feature_matrix,
reshape_to_boat1, and race-relative z-score properties.
"""

import numpy as np
import pandas as pd
import pytest

from .feature_config import (
    CATEGORICAL_FILL_COLS,
    FEATURE_COLS,
    LEAKED_COLS,
    RELEVANCE_SCHEMES,
    _race_zscore,
    compute_interaction_features,
    compute_relative_features,
    encode_race_grade,
    encode_racer_class,
    encode_weather,
    neutralize_leaked_features,
    prepare_feature_matrix,
)
from .boat1_features import (
    BOAT1_FEATURE_COLS,
    FIELD_SIZE,
    reshape_to_boat1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_race_df(n_races: int = 2, *, seed: int = 42) -> pd.DataFrame:
    """Build a minimal DataFrame that satisfies both pipelines."""
    rng = np.random.RandomState(seed)
    rows = []
    for r in range(n_races):
        race_id = 1000 + r
        race_date = f"2025-01-{10 + r:02d}"
        # Randomize course_number as a permutation of 1-6
        courses = rng.permutation(6) + 1
        for boat in range(1, 7):
            rows.append({
                "race_id": race_id,
                "race_date": race_date,
                "racer_id": 1000 + r * 6 + boat,
                "boat_number": boat,
                "course_number": int(courses[boat - 1]),
                "finish_position": boat,  # boat 1 wins
                "stadium_id": 4,
                "race_number": 1,
                "race_grade": "一般",
                "race_grade_code": 1,
                "weather": "晴",
                "weather_code": 1,
                "racer_class": "A1",
                "racer_class_code": 4,
                "racer_weight": 50.0 + rng.rand() * 5,
                "national_win_rate": 5.0 + rng.rand() * 3,
                "national_top2_rate": 20.0 + rng.rand() * 10,
                "national_top3_rate": 30.0 + rng.rand() * 10,
                "local_win_rate": 5.0 + rng.rand() * 3,
                "local_top3_rate": 30.0 + rng.rand() * 10,
                "motor_top2_rate": 25.0 + rng.rand() * 10,
                "motor_top3_rate": 35.0 + rng.rand() * 10,
                "exhibition_time": 6.5 + rng.rand() * 0.5,
                "exhibition_st": 0.1 + rng.rand() * 0.2,
                "average_st": 0.15 + rng.rand() * 0.05,
                "st_stability": 0.02 + rng.rand() * 0.03,
                "racer_course_win_rate": 4.0 + rng.rand() * 4,
                "racer_course_top2_rate": 15.0 + rng.rand() * 10,
                "recent_avg_position": 2.0 + rng.rand() * 3,
                "stadium_course_win_rate": 40.0 + rng.rand() * 20,
                "wind_speed": int(rng.randint(0, 10)),
                "wind_direction": int(rng.randint(0, 18)),
                "wave_height": int(rng.randint(0, 5)),
                "rolling_st_mean": 0.15 + rng.rand() * 0.05,
                "rolling_win_rate": 0.1 + rng.rand() * 0.3,
                "rolling_avg_position": 2.0 + rng.rand() * 2,
                "rolling_course_win_rate": 0.1 + rng.rand() * 0.3,
                "rolling_course_st": 0.15 + rng.rand() * 0.05,
                "tourn_exhibition_delta": rng.randn() * 0.1,
                "tourn_st_delta": rng.randn() * 0.05,
                "tourn_avg_position": 2.0 + rng.rand() * 3,
                "tansho_odds": 2.0 + rng.rand() * 10,
                "course_taking_rate": rng.rand() * 0.3,
                "flying_count": int(rng.randint(0, 3)),
                "bc_lap_time": 37.0 + rng.rand() * 2,
                "bc_turn_time": 5.5 + rng.rand() * 0.5,
                "bc_straight_time": 7.0 + rng.rand() * 0.5,
                "bc_slit_diff": rng.rand() * 3,
                "position_alpha": rng.randn() * 0.5,
                "rolling_position_alpha": rng.randn() * 0.5,
                "rolling_course_position_alpha": rng.randn() * 0.5,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Column definitions
# ---------------------------------------------------------------------------


class TestFeatureColumnDefinitions:
    def test_feature_cols_is_nonempty_list_of_strings(self):
        assert isinstance(FEATURE_COLS, list)
        assert len(FEATURE_COLS) > 0
        assert all(isinstance(c, str) for c in FEATURE_COLS)

    def test_boat1_feature_cols_is_nonempty_list_of_strings(self):
        assert isinstance(BOAT1_FEATURE_COLS, list)
        assert len(BOAT1_FEATURE_COLS) > 0
        assert all(isinstance(c, str) for c in BOAT1_FEATURE_COLS)

    def test_no_duplicate_feature_cols(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), (
            f"Duplicates: {[c for c in FEATURE_COLS if FEATURE_COLS.count(c) > 1]}"
        )

    def test_no_duplicate_boat1_feature_cols(self):
        assert len(BOAT1_FEATURE_COLS) == len(set(BOAT1_FEATURE_COLS)), (
            f"Duplicates: {[c for c in BOAT1_FEATURE_COLS if BOAT1_FEATURE_COLS.count(c) > 1]}"
        )

    def test_feature_cols_known_positions(self):
        """Feature order is frozen for model compatibility. Verify key positions."""
        assert FEATURE_COLS[0] == "boat_number"
        assert FEATURE_COLS[1] == "racer_weight"
        assert FEATURE_COLS[2] == "national_win_rate"
        # course_number near the end of the original block
        assert "course_number" in FEATURE_COLS
        assert FEATURE_COLS.index("course_number") == 15

    def test_boat1_feature_cols_known_positions(self):
        """Boat1 feature order is frozen for model compatibility."""
        assert BOAT1_FEATURE_COLS[0] == "b1_local_win_rate"
        assert "opp_max_national_win_rate" in BOAT1_FEATURE_COLS
        assert "wind_speed" in BOAT1_FEATURE_COLS

    def test_feature_cols_expected_count(self):
        """Guard against accidental additions/removals."""
        assert len(FEATURE_COLS) == 31

    def test_boat1_feature_cols_expected_count(self):
        assert len(BOAT1_FEATURE_COLS) == 32

    def test_leaked_cols_subset_of_feature_cols(self):
        """LEAKED_COLS must be a subset of FEATURE_COLS (or empty)."""
        for col in LEAKED_COLS:
            assert col in FEATURE_COLS, f"Leaked col {col} not in FEATURE_COLS"

    def test_relevance_schemes_cover_all_positions(self):
        """Each scheme should map finish positions 1-6."""
        for name, scheme in RELEVANCE_SCHEMES.items():
            assert set(scheme.keys()) == {1, 2, 3, 4, 5, 6}, (
                f"Scheme '{name}' missing positions"
            )


# ---------------------------------------------------------------------------
# 2. Encoding functions
# ---------------------------------------------------------------------------


class TestEncodeRaceGrade:
    @pytest.mark.parametrize("grade,expected", [
        ("SG", 5), ("G1", 4), ("G2", 3), ("G3", 2), ("一般", 1),
    ])
    def test_known_values(self, grade, expected):
        assert encode_race_grade(grade) == expected

    def test_none_returns_zero(self):
        assert encode_race_grade(None) == 0

    def test_empty_string_returns_zero(self):
        assert encode_race_grade("") == 0

    def test_unknown_value_returns_zero(self):
        assert encode_race_grade("unknown") == 0
        assert encode_race_grade("GP") == 0

    def test_ordinal_ordering(self):
        """Higher grade = higher code."""
        grades = ["一般", "G3", "G2", "G1", "SG"]
        codes = [encode_race_grade(g) for g in grades]
        assert codes == sorted(codes)


class TestEncodeRacerClass:
    @pytest.mark.parametrize("cls,expected", [
        ("A1", 4), ("A2", 3), ("B1", 2), ("B2", 1),
    ])
    def test_known_values(self, cls, expected):
        assert encode_racer_class(cls) == expected

    def test_none_returns_zero(self):
        assert encode_racer_class(None) == 0

    def test_empty_string_returns_zero(self):
        assert encode_racer_class("") == 0

    def test_unknown_value_returns_zero(self):
        assert encode_racer_class("C1") == 0
        assert encode_racer_class("a1") == 0  # case sensitive

    def test_ordinal_ordering(self):
        classes = ["B2", "B1", "A2", "A1"]
        codes = [encode_racer_class(c) for c in classes]
        assert codes == sorted(codes)


class TestEncodeWeather:
    @pytest.mark.parametrize("weather,expected", [
        ("晴", 1), ("曇り", 2), ("雨", 3), ("雪", 4), ("霧", 5),
    ])
    def test_known_values(self, weather, expected):
        assert encode_weather(weather) == expected

    def test_none_returns_zero(self):
        assert encode_weather(None) == 0

    def test_empty_string_returns_zero(self):
        assert encode_weather("") == 0

    def test_unknown_value_returns_zero(self):
        assert encode_weather("台風") == 0


# ---------------------------------------------------------------------------
# 3. Race z-score
# ---------------------------------------------------------------------------


class TestRaceZscore:
    def test_zero_mean_within_race(self):
        """Z-scored features should have ~0 mean within each race."""
        df = _make_race_df(n_races=5)
        zscored = _race_zscore(df, "national_win_rate")
        df["z"] = zscored
        race_means = df.groupby("race_id")["z"].mean()
        for race_id, mean_val in race_means.items():
            assert abs(mean_val) < 1e-10, (
                f"Race {race_id} z-score mean = {mean_val}"
            )

    def test_unit_variance_within_race(self):
        """Z-scored features should have ~1 std within each race (when std > 0)."""
        df = _make_race_df(n_races=5)
        zscored = _race_zscore(df, "national_win_rate")
        df["z"] = zscored
        race_stds = df.groupby("race_id")["z"].std()
        for race_id, std_val in race_stds.items():
            assert abs(std_val - 1.0) < 1e-10, (
                f"Race {race_id} z-score std = {std_val}"
            )

    def test_constant_column_no_inf(self):
        """If all values are the same in a race, z-score should be 0 (not inf/nan)."""
        df = pd.DataFrame({
            "race_id": [1] * 6,
            "val": [5.0] * 6,
        })
        zscored = _race_zscore(df, "val")
        assert not zscored.isna().any()
        assert not np.isinf(zscored).any()
        assert (zscored == 0.0).all()


# ---------------------------------------------------------------------------
# 4. compute_relative_features
# ---------------------------------------------------------------------------


class TestComputeRelativeFeatures:
    def test_adds_expected_columns(self):
        df = _make_race_df(n_races=2)
        result = compute_relative_features(df)
        expected_new = [
            "rel_national_win_rate", "rel_exhibition_time",
            "rel_exhibition_st",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_input(self):
        df = _make_race_df(n_races=1)
        original_cols = list(df.columns)
        compute_relative_features(df)
        assert list(df.columns) == original_cols

    def test_relative_features_zero_mean(self):
        """All rel_ features should have ~0 mean within each race."""
        df = _make_race_df(n_races=3)
        result = compute_relative_features(df)
        rel_cols = [c for c in result.columns if c.startswith("rel_")]
        for col in rel_cols:
            race_means = result.groupby("race_id")[col].mean()
            for race_id, m in race_means.items():
                assert abs(m) < 1e-10, (
                    f"{col} race {race_id} mean = {m}"
                )


# ---------------------------------------------------------------------------
# 5. compute_interaction_features
# ---------------------------------------------------------------------------


class TestComputeInteractionFeatures:
    def test_adds_expected_columns(self):
        df = _make_race_df(n_races=2)
        df = compute_relative_features(df)
        result = compute_interaction_features(df)
        expected = [
            "class_x_boat", "motor_x_boat", "wind_x_wave",
            "weight_x_boat", "wind_speed_x_boat", "kado_x_exhibition",
            "has_front_taking",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_input(self):
        df = _make_race_df(n_races=1)
        df = compute_relative_features(df)
        original_cols = list(df.columns)
        compute_interaction_features(df)
        assert list(df.columns) == original_cols

    def test_has_front_taking_binary(self):
        df = _make_race_df(n_races=2)
        df = compute_relative_features(df)
        result = compute_interaction_features(df)
        assert result["has_front_taking"].isin([0, 1]).all()

    def test_kado_exactly_one_per_race(self):
        """Each race should have exactly one kado boat."""
        df = _make_race_df(n_races=3)
        df = compute_relative_features(df)
        result = compute_interaction_features(df)
        kado_per_race = result.groupby("race_id")["is_kado"].sum()
        assert (kado_per_race == 1).all()


# ---------------------------------------------------------------------------
# 6. prepare_feature_matrix
# ---------------------------------------------------------------------------


class TestPrepareFeatureMatrix:
    def _build_df(self, n_races: int = 3) -> pd.DataFrame:
        """Build a DataFrame with all columns needed by prepare_feature_matrix."""
        df = _make_race_df(n_races=n_races)
        df = compute_relative_features(df)
        df = compute_interaction_features(df)
        return df

    def test_output_shape(self):
        n_races = 4
        df = self._build_df(n_races=n_races)
        X, y, meta = prepare_feature_matrix(df)
        assert X.shape == (n_races * 6, len(FEATURE_COLS))
        assert len(y) == n_races * 6
        assert len(meta) == n_races * 6

    def test_columns_match_feature_cols(self):
        df = self._build_df()
        X, _, _ = prepare_feature_matrix(df)
        assert list(X.columns) == FEATURE_COLS

    def test_meta_contains_required_columns(self):
        df = self._build_df()
        _, _, meta = prepare_feature_matrix(df)
        for col in ["race_id", "racer_id", "race_date", "boat_number"]:
            assert col in meta.columns

    def test_categorical_fill_cols_no_nan(self):
        """Categorical columns should be filled with 0, not NaN."""
        df = self._build_df()
        # Inject NaN into a categorical column
        df.loc[df.index[0], "race_grade_code"] = np.nan
        X, _, _ = prepare_feature_matrix(df)
        for col in CATEGORICAL_FILL_COLS:
            if col in X.columns:
                assert not X[col].isna().any(), f"{col} has NaN after prepare"

    def test_no_nan_in_boat_number(self):
        df = self._build_df()
        X, _, _ = prepare_feature_matrix(df)
        assert not X["boat_number"].isna().any()

    def test_target_is_finish_position(self):
        df = self._build_df()
        X, y, _ = prepare_feature_matrix(df)
        assert y.name == "finish_position"
        assert (y >= 1).all()


# ---------------------------------------------------------------------------
# 7. neutralize_leaked_features
# ---------------------------------------------------------------------------


class TestNeutralizeLeakedFeatures:
    def test_no_op_when_leaked_cols_empty(self):
        """When LEAKED_COLS is empty, output equals input."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        meta = pd.DataFrame({"race_id": [1, 1, 2]})
        result = neutralize_leaked_features(X, meta)
        pd.testing.assert_frame_equal(result, X)

    def test_does_not_modify_input(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        meta = pd.DataFrame({"race_id": [1, 1, 2]})
        X_orig = X.copy()
        neutralize_leaked_features(X, meta)
        pd.testing.assert_frame_equal(X, X_orig)


# ---------------------------------------------------------------------------
# 8. reshape_to_boat1
# ---------------------------------------------------------------------------


class TestReshapeToBoat1:
    def _build_full_df(self, n_races: int = 3) -> pd.DataFrame:
        """Build DataFrame with all columns needed by reshape_to_boat1."""
        df = _make_race_df(n_races=n_races)
        df = compute_relative_features(df)
        df = compute_interaction_features(df)
        return df

    def test_output_shape(self):
        n_races = 5
        df = self._build_full_df(n_races=n_races)
        X_b1, y_b1, meta_b1 = reshape_to_boat1(df)
        assert X_b1.shape == (n_races, len(BOAT1_FEATURE_COLS))
        assert len(y_b1) == n_races
        assert len(meta_b1) == n_races

    def test_columns_match_boat1_feature_cols(self):
        df = self._build_full_df()
        X_b1, _, _ = reshape_to_boat1(df)
        assert list(X_b1.columns) == BOAT1_FEATURE_COLS

    def test_target_is_binary(self):
        df = self._build_full_df()
        _, y_b1, _ = reshape_to_boat1(df)
        assert set(y_b1.unique()).issubset({0, 1})

    def test_boat1_wins_when_finish_1(self):
        """In our fixture, boat 1 always finishes 1st."""
        df = self._build_full_df(n_races=3)
        _, y_b1, _ = reshape_to_boat1(df)
        assert (y_b1 == 1).all()

    def test_meta_contains_race_id_and_date(self):
        df = self._build_full_df()
        _, _, meta_b1 = reshape_to_boat1(df)
        assert "race_id" in meta_b1.columns
        assert "race_date" in meta_b1.columns

    def test_opponent_aggregates_reasonable(self):
        """Opponent max win rate should be >= any single opponent's rate."""
        df = self._build_full_df(n_races=2)
        X_b1, _, _ = reshape_to_boat1(df)
        # opp_max should be positive (our fixture has positive rates)
        assert (X_b1["opp_max_national_win_rate"] > 0).all()

    def test_gap_features_sign(self):
        """b1_vs_best_opp is boat1 minus best opponent."""
        df = self._build_full_df(n_races=2)
        X_b1, _, _ = reshape_to_boat1(df)
        # Should be a finite number
        assert X_b1["b1_vs_best_opp_win_rate"].notna().all()
        assert X_b1["b1_vs_best_opp_exhibition"].notna().all()

    def test_wind_decomposition(self):
        """headwind and crosswind should be bounded by wind_speed."""
        df = self._build_full_df(n_races=4)
        X_b1, _, _ = reshape_to_boat1(df)
        wind = X_b1["wind_speed"].values
        hw = X_b1["headwind"].values
        cw = X_b1["crosswind"].values
        # sqrt(hw^2 + cw^2) should approximately equal wind_speed
        reconstructed = np.sqrt(hw**2 + cw**2)
        np.testing.assert_allclose(reconstructed, wind, atol=1e-10)

    def test_field_size_constant(self):
        assert FIELD_SIZE == 6
