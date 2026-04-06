"""Tests for boat 1 binary classifier model.

Covers: train_boat1_model, save/load round-trip, prediction shape and range,
early stopping with validation set, metrics dict contents, and edge cases
(all-win / all-lose targets).
"""

import numpy as np
import pandas as pd
import pytest

from .boat1_features import BOAT1_FEATURE_COLS, reshape_to_boat1
from .boat1_model import (
    BOAT1_DEFAULT_PARAMS,
    load_boat1_model,
    save_boat1_model,
    train_boat1_model,
)
from .feature_config import (
    compute_interaction_features,
    compute_relative_features,
)
from .test_feature_config import _make_race_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELD_SIZE = 6


def _build_boat1_data(
    n_races: int = 30,
    *,
    seed: int = 42,
    boat1_always_wins: bool | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build synthetic boat1 data via the real reshape pipeline.

    Args:
        boat1_always_wins: If True, boat 1 always finishes 1st.
            If False, boat 1 never finishes 1st. If None, mixed.
    """
    df = _make_race_df(n_races=n_races, seed=seed)
    rng = np.random.RandomState(seed)

    if boat1_always_wins is True:
        # Default fixture already has boat 1 finishing 1st
        pass
    elif boat1_always_wins is False:
        # Swap finish positions so boat 1 never wins
        for race_id in df["race_id"].unique():
            mask = df["race_id"] == race_id
            b1_mask = mask & (df["boat_number"] == 1)
            b2_mask = mask & (df["boat_number"] == 2)
            # Swap boat 1 (pos=1) and boat 2 (pos=2)
            df.loc[b1_mask, "finish_position"] = 2
            df.loc[b2_mask, "finish_position"] = 1
    else:
        # Mixed: randomly assign some races where boat 1 loses
        for race_id in df["race_id"].unique():
            if rng.rand() < 0.5:
                mask = df["race_id"] == race_id
                b1_mask = mask & (df["boat_number"] == 1)
                b2_mask = mask & (df["boat_number"] == 2)
                df.loc[b1_mask, "finish_position"] = 2
                df.loc[b2_mask, "finish_position"] = 1

    df = compute_relative_features(df)
    df = compute_interaction_features(df)
    X_b1, y_b1, meta_b1 = reshape_to_boat1(df)
    return X_b1, y_b1, meta_b1


# ---------------------------------------------------------------------------
# 1. train_boat1_model returns model and metrics
# ---------------------------------------------------------------------------


class TestTrainBoat1Model:
    def test_returns_model_and_metrics(self):
        X, y, _ = _build_boat1_data(n_races=30)
        model, metrics = train_boat1_model(X, y)
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_contains_feature_importance(self):
        X, y, _ = _build_boat1_data(n_races=30)
        # Use small min_child_samples so the model can split with few rows
        _, metrics = train_boat1_model(
            X, y, extra_params={"min_child_samples": 5},
        )
        assert "feature_importance" in metrics
        fi = metrics["feature_importance"]
        assert isinstance(fi, dict)
        assert len(fi) == len(BOAT1_FEATURE_COLS)
        # Importance values should sum to ~1 (normalized)
        total = sum(fi.values())
        assert abs(total - 1.0) < 1e-6

    def test_can_override_n_estimators(self):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, n_estimators=10)
        assert model.n_estimators == 10

    def test_can_override_learning_rate(self):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, learning_rate=0.1)
        assert model.learning_rate == 0.1

    def test_extra_params_override_defaults(self):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, extra_params={"max_depth": 3})
        assert model.max_depth == 3


# ---------------------------------------------------------------------------
# 2. Predictions are probabilities in [0, 1]
# ---------------------------------------------------------------------------


class TestPredictionRange:
    def test_probabilities_in_unit_interval(self):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, n_estimators=20)
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_predict_proba_two_columns(self):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, n_estimators=20)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        # Columns should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Prediction shape (one per race)
# ---------------------------------------------------------------------------


class TestPredictionShape:
    def test_one_prediction_per_race(self):
        n_races = 20
        X, y, _ = _build_boat1_data(n_races=n_races)
        model, _ = train_boat1_model(X, y, n_estimators=20)
        proba = model.predict_proba(X)[:, 1]
        assert len(proba) == n_races


# ---------------------------------------------------------------------------
# 4. Save/load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_predictions_identical_after_reload(self, tmp_path):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, n_estimators=20)
        original_pred = model.predict_proba(X)[:, 1]

        out_dir = str(tmp_path / "boat1_model")
        save_boat1_model(model, out_dir)
        loaded = load_boat1_model(out_dir)

        loaded_pred = loaded.predict_proba(X)[:, 1]
        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_creates_directory(self, tmp_path):
        X, y, _ = _build_boat1_data(n_races=30)
        model, _ = train_boat1_model(X, y, n_estimators=5)

        nested = str(tmp_path / "a" / "b" / "c")
        path = save_boat1_model(model, nested)
        assert "model.pkl" in path

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_boat1_model(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# 5. Validation set and early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_val_auc_in_metrics(self):
        X, y, _ = _build_boat1_data(n_races=60)
        split = len(X) // 2
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        _, metrics = train_boat1_model(
            X_train, y_train, X_val, y_val, n_estimators=200,
        )
        assert "val_auc" in metrics
        assert 0.0 <= metrics["val_auc"] <= 1.0

    def test_early_stopping_fewer_trees(self):
        """With early stopping, model should use fewer trees than max."""
        X, y, _ = _build_boat1_data(n_races=80, seed=99)
        split = len(X) // 2
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        max_trees = 500
        model, _ = train_boat1_model(
            X_train, y_train, X_val, y_val,
            n_estimators=max_trees, early_stopping_rounds=10,
        )
        # Early stopping should kick in before reaching max
        actual_trees = model.best_iteration_
        assert actual_trees < max_trees

    def test_no_val_set_no_val_auc(self):
        X, y, _ = _build_boat1_data(n_races=30)
        _, metrics = train_boat1_model(X, y, n_estimators=20)
        assert "val_auc" not in metrics


# ---------------------------------------------------------------------------
# 6. Metrics dict expected keys
# ---------------------------------------------------------------------------


class TestMetricsDict:
    def test_feature_importance_keys_match_columns(self):
        X, y, _ = _build_boat1_data(n_races=30)
        _, metrics = train_boat1_model(X, y, n_estimators=20)
        fi = metrics["feature_importance"]
        assert set(fi.keys()) == set(BOAT1_FEATURE_COLS)

    def test_feature_importance_all_non_negative(self):
        X, y, _ = _build_boat1_data(n_races=30)
        _, metrics = train_boat1_model(X, y, n_estimators=20)
        fi = metrics["feature_importance"]
        assert all(v >= 0 for v in fi.values())


# ---------------------------------------------------------------------------
# 7. Edge cases: all-win and all-lose targets
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_boat1_wins(self):
        """Model trains without error when boat 1 always wins (y=all 1)."""
        X, y, _ = _build_boat1_data(n_races=30, boat1_always_wins=True)
        assert (y == 1).all()
        model, metrics = train_boat1_model(X, y, n_estimators=10)
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()
        assert "feature_importance" in metrics

    def test_all_boat1_loses(self):
        """Model trains without error when boat 1 never wins (y=all 0)."""
        X, y, _ = _build_boat1_data(n_races=30, boat1_always_wins=False)
        assert (y == 0).all()
        model, metrics = train_boat1_model(X, y, n_estimators=10)
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()
        assert "feature_importance" in metrics


# ---------------------------------------------------------------------------
# 8. Default params sanity
# ---------------------------------------------------------------------------


class TestDefaultParams:
    def test_objective_is_binary(self):
        assert BOAT1_DEFAULT_PARAMS["objective"] == "binary"

    def test_verbose_is_silent(self):
        assert BOAT1_DEFAULT_PARAMS["verbose"] == -1

    def test_random_state_is_set(self):
        assert "random_state" in BOAT1_DEFAULT_PARAMS
