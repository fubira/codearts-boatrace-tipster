"""LightGBM model wrapper for boat race prediction.

Uses LambdaRank (learning-to-rank) instead of multiclass classification.
Each race is a query group (6 boats); labels are relevance scores derived
from finish position.
"""

import json
import pickle
from datetime import timedelta, timezone
from datetime import datetime as dt
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .feature_config import RELEVANCE_SCHEMES

MAX_RELEVANCE_LABEL = 31  # LightGBM default label upper bound
FIELD_SIZE = 6  # Boat racing always has 6 boats

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.001,
    "reg_lambda": 0.001,
    "max_depth": 8,
    "random_state": 42,
    "verbose": -1,
}


def _position_to_relevance(
    positions: pd.Series,
    scheme: str = "linear",
) -> np.ndarray:
    """Convert finish positions to relevance scores for LambdaRank.

    Uses scheme definitions from feature_config.RELEVANCE_SCHEMES.
    """
    mapping = RELEVANCE_SCHEMES.get(scheme)
    if mapping is None:
        raise ValueError(f"Unknown relevance scheme: {scheme}")

    return np.array([mapping.get(int(p), 0) for p in positions], dtype=float)


def _compute_query_groups(race_ids: pd.Series) -> np.ndarray:
    """Compute group sizes for LambdaRank from race_id column."""
    return race_ids.groupby(race_ids).count().values


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    train_start: str | None = None,
    val_start: str = "2025-09-01",
    test_start: str = "2025-10-01",
) -> dict:
    """Split data chronologically into train/val/test sets.

    Splits by fixed date boundaries to prevent temporal leakage.
    """
    race_dates = pd.to_datetime(meta["race_date"])
    val_dt = pd.Timestamp(val_start)
    test_dt = pd.Timestamp(test_start)

    train_mask = race_dates < val_dt
    if train_start is not None:
        train_mask = train_mask & (race_dates >= pd.Timestamp(train_start))
    val_mask = (race_dates >= val_dt) & (race_dates < test_dt)
    test_mask = race_dates >= test_dt

    return {
        "train": {"X": X[train_mask], "y": y[train_mask], "meta": meta[train_mask]},
        "val": {"X": X[val_mask], "y": y[val_mask], "meta": meta[val_mask]},
        "test": {"X": X[test_mask], "y": y[test_mask], "meta": meta[test_mask]},
    }


def walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    fold_months: int = 2,
    n_folds: int = 4,
    test_end: str | None = None,
    min_train_months: int = 12,
    train_start: str | None = None,
) -> list[dict]:
    """Generate walk-forward time-series splits.

    Walks backward from test_end, creating n_folds of fold_months each.
    Each fold uses all prior data as training and 1 month before as validation.
    """
    race_dates = pd.to_datetime(meta["race_date"])

    if test_end is None:
        last_date = race_dates.max()
    else:
        last_date = pd.Timestamp(test_end)

    folds = []
    for i in range(n_folds):
        test_end_dt = last_date - pd.DateOffset(months=fold_months * i)
        test_start_dt = test_end_dt - pd.DateOffset(months=fold_months)
        val_start_dt = test_start_dt - pd.DateOffset(months=1)
        train_end_dt = val_start_dt

        earliest = race_dates.min()
        train_months = (train_end_dt - earliest).days / 30
        if train_months < min_train_months:
            break

        train_mask = race_dates < train_end_dt
        if train_start is not None:
            train_mask = train_mask & (race_dates >= pd.Timestamp(train_start))
        val_mask = (race_dates >= val_start_dt) & (race_dates < test_start_dt)
        test_mask = (race_dates >= test_start_dt) & (race_dates < test_end_dt)

        if test_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        folds.append({
            "train": {"X": X[train_mask], "y": y[train_mask], "meta": meta[train_mask]},
            "val": {"X": X[val_mask], "y": y[val_mask], "meta": meta[val_mask]},
            "test": {"X": X[test_mask], "y": y[test_mask], "meta": meta[test_mask]},
            "period": {
                "train_end": str(train_end_dt.date()),
                "val": f"{val_start_dt.date()} ~ {test_start_dt.date()}",
                "test": f"{test_start_dt.date()} ~ {test_end_dt.date()}",
            },
        })

    return list(reversed(folds))


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    meta_val: pd.DataFrame | None = None,
    *,
    n_estimators: int | None = None,
    learning_rate: float | None = None,
    extra_params: dict | None = None,
    relevance_scheme: str = "linear",
    early_stopping_rounds: int | None = None,
) -> tuple[lgb.LGBMRanker, dict]:
    """Train a LightGBM LambdaRank model.

    Returns (model, metrics).
    """
    params = DEFAULT_PARAMS.copy()
    if extra_params:
        params.update(extra_params)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if learning_rate is not None:
        params["learning_rate"] = learning_rate

    y_rel = _position_to_relevance(y_train, scheme=relevance_scheme)
    groups_train = _compute_query_groups(meta_train["race_id"])

    model = lgb.LGBMRanker(**params)

    fit_kwargs: dict = {
        "X": X_train,
        "y": y_rel,
        "group": groups_train,
    }

    callbacks = []
    if X_val is not None and y_val is not None and meta_val is not None:
        y_rel_val = _position_to_relevance(y_val, scheme=relevance_scheme)
        groups_val = _compute_query_groups(meta_val["race_id"])
        fit_kwargs["eval_set"] = [(X_val, y_rel_val)]
        fit_kwargs["eval_group"] = [groups_val]
        fit_kwargs["eval_at"] = [1, 3]
        if early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    model.fit(**fit_kwargs)

    importance = dict(
        zip(
            X_train.columns,
            model.feature_importances_ / model.feature_importances_.sum(),
        )
    )

    metrics = {"feature_importance": importance}
    return model, metrics


def save_model(model: lgb.LGBMRanker, output_dir: str) -> str:
    """Save model to disk."""
    path = Path(output_dir) / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return str(path)


def load_model(model_dir: str) -> lgb.LGBMRanker:
    """Load model from disk."""
    path = Path(model_dir) / "model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_meta(
    output_dir: str,
    feature_columns: list[str],
    hyperparameters: dict,
    training: dict,
    feature_means: dict[str, float] | None = None,
    **extra,
) -> str:
    """Save model metadata for inference-time feature compatibility."""
    meta = {
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters,
        "training": training,
        "created_at": dt.now(timezone(timedelta(hours=9))).isoformat(),
    }
    if feature_means is not None:
        meta["feature_means"] = feature_means
    # Additional fields (architecture, stage1_features, stage1_feature_means, etc.)
    for k, v in extra.items():
        if v is not None:
            meta[k] = v
    path = Path(output_dir) / "model_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return str(path)


def load_model_meta(model_dir: str) -> dict | None:
    """Load model_meta.json if present."""
    path = Path(model_dir) / "model_meta.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


_LGB_PARAM_KEYS = {
    "num_leaves", "max_depth", "min_child_samples",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
}


def load_training_params(model_dir: str) -> dict:
    """Load training hyperparameters from model_meta.json.

    Returns dict with keys matching train_model() arguments:
        extra_params: dict of LightGBM tree params
        n_estimators: int
        learning_rate: float
        relevance_scheme: str
    """
    meta = load_model_meta(model_dir)
    if not meta or "hyperparameters" not in meta:
        return {
            "extra_params": {},
            "n_estimators": DEFAULT_PARAMS["n_estimators"],
            "learning_rate": DEFAULT_PARAMS["learning_rate"],
            "relevance_scheme": "linear",
        }
    hp = dict(meta["hyperparameters"])
    return {
        "extra_params": {k: v for k, v in hp.items() if k in _LGB_PARAM_KEYS},
        "n_estimators": hp.get("n_estimators", DEFAULT_PARAMS["n_estimators"]),
        "learning_rate": hp.get("learning_rate", DEFAULT_PARAMS["learning_rate"]),
        "relevance_scheme": hp.get("relevance_scheme", "linear"),
    }
