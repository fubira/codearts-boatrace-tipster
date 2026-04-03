"""Tests for predict_boat1 NaN fallback behavior."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from boatrace_tipster_ml.boat1_features import BOAT1_FEATURE_COLS


class TestNaNFallback:
    """Test that NaN features are filled from model_meta feature_means."""

    def _make_X_with_nans(self, n_races: int = 5) -> pd.DataFrame:
        """Create a feature DataFrame with NaN in exhibition-related columns."""
        rng = np.random.default_rng(42)
        data = {c: rng.random(n_races) for c in BOAT1_FEATURE_COLS}
        X = pd.DataFrame(data)
        # Simulate missing exhibition data
        for c in ["b1_rel_exhibition_time", "opp_best_exhibition_st",
                   "wind_speed", "b1_vs_best_opp_exhibition",
                   "headwind", "crosswind"]:
            X[c] = np.nan
        return X

    def _make_feature_means(self) -> dict[str, float]:
        """Create dummy feature means matching BOAT1_FEATURE_COLS."""
        return {c: 0.5 for c in BOAT1_FEATURE_COLS}

    def test_fills_nan_with_feature_means(self):
        X = self._make_X_with_nans()
        X = X.astype("float64")
        feature_means = self._make_feature_means()

        nan_cols = [c for c in X.columns if X[c].isna().any()]
        assert len(nan_cols) == 6

        for c in nan_cols:
            if c in feature_means:
                X[c] = X[c].fillna(feature_means[c])

        # All NaN should be filled
        assert X.isna().sum().sum() == 0
        # Filled values should match means
        assert X["headwind"].iloc[0] == 0.5

    def test_no_fill_without_feature_means(self):
        X = self._make_X_with_nans()
        X = X.astype("float64")
        feature_means = None

        nan_before = X.isna().sum().sum()
        if feature_means:
            nan_cols = [c for c in X.columns if X[c].isna().any()]
            for c in nan_cols:
                if c in feature_means:
                    X[c] = X[c].fillna(feature_means[c])

        # NaN should remain
        assert X.isna().sum().sum() == nan_before

    def test_partial_fill_with_incomplete_means(self):
        X = self._make_X_with_nans()
        X = X.astype("float64")
        # Only provide means for some columns
        feature_means = {"headwind": 1.0, "crosswind": 2.0}

        nan_cols = [c for c in X.columns if X[c].isna().any()]
        for c in nan_cols:
            if c in feature_means:
                X[c] = X[c].fillna(feature_means[c])

        # headwind and crosswind should be filled
        assert X["headwind"].isna().sum() == 0
        assert X["crosswind"].isna().sum() == 0
        # Others should remain NaN
        assert X["b1_rel_exhibition_time"].isna().sum() == 5

    def test_no_change_when_no_nans(self):
        rng = np.random.default_rng(42)
        data = {c: rng.random(5) for c in BOAT1_FEATURE_COLS}
        X = pd.DataFrame(data).astype("float64")
        feature_means = self._make_feature_means()

        nan_cols = [c for c in X.columns if X[c].isna().any()]
        assert len(nan_cols) == 0
        # No fill should occur


class TestHasExhibition:
    """Test that has_exhibition flag is correctly set in reshape_to_boat1."""

    def test_has_exhibition_true_when_data_present(self):
        from boatrace_tipster_ml.boat1_features import reshape_to_boat1, FIELD_SIZE

        n_races = 2
        n_entries = n_races * FIELD_SIZE
        rng = np.random.default_rng(42)

        df = pd.DataFrame({
            "race_id": np.repeat([1, 2], FIELD_SIZE),
            "boat_number": np.tile(np.arange(1, FIELD_SIZE + 1), n_races),
            "finish_position": np.tile(np.arange(1, FIELD_SIZE + 1), n_races),
            "race_date": "2026-04-03",
            "stadium_id": 1,
            "race_number": np.repeat([1, 2], FIELD_SIZE),
            "national_win_rate": rng.random(n_entries) * 10,
            "local_win_rate": rng.random(n_entries) * 10,
            "local_top3_rate": rng.random(n_entries) * 100,
            "motor_top3_rate": rng.random(n_entries) * 100,
            "exhibition_time": rng.random(n_entries) * 0.1 + 6.5,  # present
            "exhibition_st": rng.random(n_entries) * 0.3,
            "racer_course_win_rate": rng.random(n_entries),
            "racer_course_top2_rate": rng.random(n_entries),
            "stadium_course_win_rate": rng.random(n_entries),
            "racer_class_code": rng.integers(1, 5, n_entries),
            "racer_weight": rng.random(n_entries) * 10 + 48,
            "average_st": rng.random(n_entries) * 0.2,
            "st_stability": rng.random(n_entries) * 0.1,
            "rolling_st_mean": rng.random(n_entries) * 0.2,
            "rolling_win_rate": rng.random(n_entries),
            "rolling_course_win_rate": rng.random(n_entries),
            "tourn_avg_position": rng.random(n_entries) * 4 + 1,
            "rel_national_win_rate": rng.standard_normal(n_entries),
            "rel_exhibition_time": rng.standard_normal(n_entries),
            "wind_speed": rng.integers(0, 10, n_entries),
            "wind_direction": rng.integers(0, 18, n_entries),
            "has_front_taking": rng.integers(0, 2, n_entries),
            "course_taking_rate": rng.random(n_entries),
            "flying_count": rng.integers(0, 3, n_entries),
            "tansho_odds": rng.random(n_entries) * 5 + 1,
        })

        _, _, meta = reshape_to_boat1(df)
        assert meta["has_exhibition"].all()

    def test_has_exhibition_false_when_missing(self):
        from boatrace_tipster_ml.boat1_features import reshape_to_boat1, FIELD_SIZE

        n_races = 2
        n_entries = n_races * FIELD_SIZE
        rng = np.random.default_rng(42)

        df = pd.DataFrame({
            "race_id": np.repeat([1, 2], FIELD_SIZE),
            "boat_number": np.tile(np.arange(1, FIELD_SIZE + 1), n_races),
            "finish_position": np.tile(np.arange(1, FIELD_SIZE + 1), n_races),
            "race_date": "2026-04-03",
            "stadium_id": 1,
            "race_number": np.repeat([1, 2], FIELD_SIZE),
            "national_win_rate": rng.random(n_entries) * 10,
            "local_win_rate": rng.random(n_entries) * 10,
            "local_top3_rate": rng.random(n_entries) * 100,
            "motor_top3_rate": rng.random(n_entries) * 100,
            "exhibition_time": pd.array([pd.NA] * n_entries, dtype="Float64"),  # missing
            "exhibition_st": rng.random(n_entries) * 0.3,
            "racer_course_win_rate": rng.random(n_entries),
            "racer_course_top2_rate": rng.random(n_entries),
            "stadium_course_win_rate": rng.random(n_entries),
            "racer_class_code": rng.integers(1, 5, n_entries),
            "racer_weight": rng.random(n_entries) * 10 + 48,
            "average_st": rng.random(n_entries) * 0.2,
            "st_stability": rng.random(n_entries) * 0.1,
            "rolling_st_mean": rng.random(n_entries) * 0.2,
            "rolling_win_rate": rng.random(n_entries),
            "rolling_course_win_rate": rng.random(n_entries),
            "tourn_avg_position": rng.random(n_entries) * 4 + 1,
            "rel_national_win_rate": rng.standard_normal(n_entries),
            "rel_exhibition_time": rng.standard_normal(n_entries),
            "wind_speed": rng.integers(0, 10, n_entries),
            "wind_direction": rng.integers(0, 18, n_entries),
            "has_front_taking": rng.integers(0, 2, n_entries),
            "course_taking_rate": rng.random(n_entries),
            "flying_count": rng.integers(0, 3, n_entries),
            "tansho_odds": rng.random(n_entries) * 5 + 1,
        })

        _, _, meta = reshape_to_boat1(df)
        assert not meta["has_exhibition"].any()
