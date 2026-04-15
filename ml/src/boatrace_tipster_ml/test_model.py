"""Tests for model utilities.

Covers: position-to-relevance conversion, query group computation,
model save/load round-trip, model meta save/load round-trip.
"""

import numpy as np
import pandas as pd
import pytest

from .model import (
    _compute_query_groups,
    _position_to_relevance,
    load_model,
    load_model_meta,
    save_model,
    save_model_meta,
)


# ---------------------------------------------------------------------------
# _position_to_relevance: scheme correctness
# ---------------------------------------------------------------------------


class TestPositionToRelevance:
    """Each relevance scheme maps finish positions to expected scores."""

    @pytest.fixture()
    def positions(self) -> pd.Series:
        return pd.Series([1, 2, 3, 4, 5, 6])

    # -- linear --------------------------------------------------------

    def test_linear_values(self, positions):
        rel = _position_to_relevance(positions, scheme="linear")
        np.testing.assert_array_equal(rel, [6, 5, 4, 3, 2, 1])

    def test_linear_monotonically_decreasing(self, positions):
        rel = _position_to_relevance(positions, scheme="linear")
        assert all(rel[i] > rel[i + 1] for i in range(5))

    def test_linear_all_different(self, positions):
        rel = _position_to_relevance(positions, scheme="linear")
        assert len(set(rel)) == 6

    # -- podium --------------------------------------------------------

    def test_podium_values(self, positions):
        rel = _position_to_relevance(positions, scheme="podium")
        np.testing.assert_array_equal(rel, [5, 3, 1, 0, 0, 0])

    def test_podium_top3_positive(self, positions):
        rel = _position_to_relevance(positions, scheme="podium")
        assert all(r > 0 for r in rel[:3])

    def test_podium_bottom3_zero(self, positions):
        rel = _position_to_relevance(positions, scheme="podium")
        assert all(r == 0 for r in rel[3:])

    # -- top_heavy -----------------------------------------------------

    def test_top_heavy_values(self, positions):
        rel = _position_to_relevance(positions, scheme="top_heavy")
        np.testing.assert_array_equal(rel, [10, 6, 3, 2, 1, 0])

    def test_top_heavy_first_more_than_linear(self, positions):
        heavy = _position_to_relevance(positions, scheme="top_heavy")
        linear = _position_to_relevance(positions, scheme="linear")
        assert heavy[0] > linear[0]

    # -- win_only ------------------------------------------------------

    def test_win_only_values(self, positions):
        rel = _position_to_relevance(positions, scheme="win_only")
        np.testing.assert_array_equal(rel, [1, 0, 0, 0, 0, 0])

    def test_win_only_first_positive(self, positions):
        rel = _position_to_relevance(positions, scheme="win_only")
        assert rel[0] > 0

    def test_win_only_rest_zero(self, positions):
        rel = _position_to_relevance(positions, scheme="win_only")
        assert all(r == 0 for r in rel[1:])

    # -- general properties --------------------------------------------

    @pytest.mark.parametrize("scheme", ["linear", "podium", "top_heavy", "win_only"])
    def test_all_non_negative(self, positions, scheme):
        rel = _position_to_relevance(positions, scheme=scheme)
        assert all(r >= 0 for r in rel)

    @pytest.mark.parametrize("scheme", ["linear", "podium", "top_heavy", "win_only"])
    def test_non_finisher_gets_zero(self, scheme):
        """Position 7 (non-finisher) should map to relevance 0."""
        rel = _position_to_relevance(pd.Series([7]), scheme=scheme)
        assert rel[0] == 0

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown relevance scheme"):
            _position_to_relevance(pd.Series([1]), scheme="nonexistent")


# ---------------------------------------------------------------------------
# _compute_query_groups
# ---------------------------------------------------------------------------


class TestComputeQueryGroups:
    def test_single_race(self):
        race_ids = pd.Series(["R1"] * 6)
        groups = _compute_query_groups(race_ids)
        np.testing.assert_array_equal(groups, [6])

    def test_multiple_races(self):
        race_ids = pd.Series(["R1"] * 6 + ["R2"] * 6 + ["R3"] * 6)
        groups = _compute_query_groups(race_ids)
        np.testing.assert_array_equal(groups, [6, 6, 6])

    def test_groups_sum_to_total(self):
        race_ids = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6)
        groups = _compute_query_groups(race_ids)
        assert groups.sum() == len(race_ids)

    def test_uneven_groups(self):
        """Non-standard group sizes (e.g. void entries)."""
        race_ids = pd.Series(["R1"] * 5 + ["R2"] * 6)
        groups = _compute_query_groups(race_ids)
        assert sorted(groups) == [5, 6]
        assert groups.sum() == 11


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------


class TestModelSaveLoad:
    def test_round_trip(self, tmp_path):
        import lightgbm as lgb

        # Train a minimal LGBMRanker so we have a real model object
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [6, 5, 4, 3, 2, 1]})
        y = np.array([5, 4, 3, 2, 1, 0])  # relevance
        group = np.array([6])

        model = lgb.LGBMRanker(n_estimators=5, verbose=-1)
        model.fit(X, y, group=group)

        out_dir = str(tmp_path / "model_out")
        save_model(model, out_dir)
        loaded = load_model(out_dir)

        # Predictions should be identical
        original_pred = model.predict(X)
        loaded_pred = loaded.predict(X)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_creates_directory(self, tmp_path):
        import lightgbm as lgb

        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
        y = np.array([5, 4, 3, 2, 1, 0])
        model = lgb.LGBMRanker(n_estimators=2, verbose=-1)
        model.fit(X, y, group=np.array([6]))

        nested = str(tmp_path / "a" / "b" / "c")
        path = save_model(model, nested)
        assert "model.pkl" in path


# ---------------------------------------------------------------------------
# save_model_meta / load_model_meta round-trip
# ---------------------------------------------------------------------------


class TestModelMetaSaveLoad:
    def test_round_trip(self, tmp_path):
        out_dir = str(tmp_path)
        features = ["f1", "f2", "f3"]
        hyperparams = {"n_estimators": 500, "learning_rate": 0.05}
        training = {"train_size": 10000, "val_size": 2000}

        save_model_meta(out_dir, features, hyperparams, training)
        meta = load_model_meta(out_dir)

        assert meta is not None
        assert meta["feature_columns"] == features
        assert meta["hyperparameters"] == hyperparams
        assert meta["training"] == training
        assert "created_at" in meta

    def test_with_feature_means(self, tmp_path):
        out_dir = str(tmp_path)
        means = {"f1": 1.5, "f2": 3.0}

        save_model_meta(out_dir, ["f1", "f2"], {}, {}, feature_means=means)
        meta = load_model_meta(out_dir)

        assert meta is not None
        assert meta["feature_means"] == means

    def test_without_feature_means(self, tmp_path):
        out_dir = str(tmp_path)
        save_model_meta(out_dir, ["f1"], {}, {})
        meta = load_model_meta(out_dir)

        assert meta is not None
        assert "feature_means" not in meta

    def test_load_missing_returns_none(self, tmp_path):
        meta = load_model_meta(str(tmp_path))
        assert meta is None

    def test_float_precision_preserved(self, tmp_path):
        """LightGBM HP like subsample/colsample_bytree must round-trip at
        full double precision. Silent rounding (e.g., :.2g, round(x, 2))
        causes measurable growth drift on retrained models."""
        out_dir = str(tmp_path)
        hp = {
            "subsample": 0.7302700512037856,
            "colsample_bytree": 0.6240721896783721,
            "reg_alpha": 8.715701573782917e-06,
            "reg_lambda": 0.09885538716630769,
            "learning_rate": 0.006172936736632081,
        }
        save_model_meta(out_dir, ["f1"], hp, {})
        meta = load_model_meta(out_dir)

        assert meta is not None
        for key, expected in hp.items():
            actual = meta["hyperparameters"][key]
            assert actual == expected, (
                f"{key} lost precision: got {actual!r}, expected {expected!r}"
            )
