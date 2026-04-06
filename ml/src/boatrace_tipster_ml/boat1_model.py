"""LightGBM binary classifier for boat 1 win prediction.

Predicts whether boat 1 (1号艇) wins a race. Used with selective betting:
bet 単勝 on boat 1 only when the model is confident (prob >= threshold).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from .db import get_connection
from .evaluate import _load_payouts

FIELD_SIZE = 6
BET_UNIT = 100

BOAT1_DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "verbose": -1,
}


def train_boat1_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    *,
    n_estimators: int | None = None,
    learning_rate: float | None = None,
    extra_params: dict | None = None,
    early_stopping_rounds: int = 50,
) -> tuple[LGBMClassifier, dict]:
    """Train a LightGBM binary classifier for boat 1 win prediction.

    Returns (model, metrics) where metrics includes feature importance and val AUC.
    """
    params = BOAT1_DEFAULT_PARAMS.copy()
    if extra_params:
        params.update(extra_params)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if learning_rate is not None:
        params["learning_rate"] = learning_rate

    model = LGBMClassifier(**params)

    fit_kwargs: dict = {"X": X_train, "y": y_train}

    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [
            __import__("lightgbm").early_stopping(early_stopping_rounds, verbose=False),
        ]

    model.fit(**fit_kwargs)

    # Feature importance (normalized)
    importance_raw = model.feature_importances_
    total = importance_raw.sum()
    importance = dict(
        zip(X_train.columns, importance_raw / total if total > 0 else importance_raw)
    )

    metrics: dict = {"feature_importance": importance}

    # Validation AUC
    if X_val is not None and y_val is not None:
        val_prob = model.predict_proba(X_val)[:, 1]
        metrics["val_auc"] = roc_auc_score(y_val, val_prob)

    return model, metrics


def evaluate_boat1(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    db_path: str | None = None,
    payouts_cache: dict | None = None,
) -> dict:
    """Evaluate boat 1 binary classifier with classification and business metrics.

    Returns dict with:
        - auc: ROC AUC
        - accuracy: overall accuracy
        - base_hit_rate: actual boat 1 win rate in test set
        - thresholds: list of {threshold, n_bets, hit_rate, roi, profit} dicts
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_true = y.values

    # Classification metrics
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = (y_pred == y_true).mean()
    base_hit_rate = y_true.mean()

    result: dict = {
        "auc": auc,
        "accuracy": accuracy,
        "base_hit_rate": base_hit_rate,
        "n_races": len(y_true),
    }

    # Business metrics: threshold-based ROI with actual payouts
    if db_path or payouts_cache is not None:
        race_ids = meta["race_id"].values
        payouts_db = payouts_cache if payouts_cache is not None else _load_payouts(db_path, race_ids)

        thresholds = _threshold_analysis(y_true, y_prob, race_ids, payouts_db)
        result["thresholds"] = thresholds

        tansho_odds = meta["b1_tansho_odds"].values if "b1_tansho_odds" in meta.columns else None
        ev_results = _ev_analysis(y_true, y_prob, race_ids, payouts_db, tansho_odds=tansho_odds)
        result["ev_analysis"] = ev_results

    # Calibration analysis (no payouts needed)
    result["calibration"] = _calibration_analysis(y_true, y_prob)

    return result


def _threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    race_ids: np.ndarray,
    payouts_db: dict,
) -> list[dict]:
    """Grid search thresholds and compute ROI using actual tansho payouts."""
    results = []

    for threshold in np.arange(0.50, 0.76, 0.01):
        mask = y_prob >= threshold
        n_bets = mask.sum()
        if n_bets == 0:
            continue

        hit_rate = y_true[mask].mean()

        # ROI from actual payouts
        total_bet = 0
        total_payout = 0
        hits = 0
        for i in np.where(mask)[0]:
            rid = int(race_ids[i])
            rp = payouts_db.get(rid)
            if not rp:
                continue
            tansho_payouts = rp.get("単勝")
            if not tansho_payouts:
                continue
            total_bet += BET_UNIT
            # Boat 1 tansho combination is "1"
            payout = tansho_payouts.get("1")
            if payout and y_true[i] == 1:
                total_payout += payout
                hits += 1

        roi = total_payout / total_bet if total_bet > 0 else 0.0
        profit = total_payout - total_bet

        results.append({
            "threshold": round(float(threshold), 2),
            "n_bets": int(n_bets),
            "hit_rate": float(hit_rate),
            "roi": float(roi),
            "profit": int(profit),
            "actual_bets": total_bet // BET_UNIT,
        })

    return results


def _ev_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    race_ids: np.ndarray,
    payouts_db: dict,
    *,
    tansho_odds: np.ndarray | None = None,
) -> list[dict]:
    """EV-based betting: bet when predicted_prob × expected_payout > bet_unit.

    Uses pre-race tansho odds (from meta) to compute expected payout.
    Expected payout = odds × 100 (per 100 yen bet).
    EV = model_prob × expected_payout - 100.
    Actual ROI is computed from race_payouts (actual winning payouts).

    Args:
        tansho_odds: Pre-race tansho odds for boat 1 (from meta["b1_tansho_odds"]).
    """
    n = len(y_true)

    if tansho_odds is None or np.all(np.isnan(tansho_odds)):
        return []

    # Expected payout from odds (odds are in 100-yen units already)
    # e.g., odds=1.5 → payout=150 yen per 100 yen bet
    expected_payout = tansho_odds * BET_UNIT
    ev = y_prob * expected_payout - BET_UNIT

    valid = ~np.isnan(ev)
    results = []

    for ev_threshold in [-20, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50]:
        mask = valid & (ev >= ev_threshold)
        n_bets = mask.sum()
        if n_bets == 0:
            continue

        # Actual ROI from real payouts (not odds-based)
        total_bet = 0
        total_payout = 0
        for i in np.where(mask)[0]:
            rid = int(race_ids[i])
            rp = payouts_db.get(rid)
            if not rp:
                continue
            tansho = rp.get("単勝")
            if not tansho:
                continue
            total_bet += BET_UNIT
            if y_true[i] == 1:
                p = tansho.get("1")
                if p:
                    total_payout += p

        roi = total_payout / total_bet if total_bet > 0 else 0.0
        hit_rate = y_true[mask].mean()
        avg_odds = tansho_odds[mask].mean()
        avg_ev = ev[mask].mean()
        avg_prob = y_prob[mask].mean()
        market_prob = (1.0 / tansho_odds[mask]).mean()  # implied prob from odds

        results.append({
            "ev_threshold": ev_threshold,
            "n_bets": int(n_bets),
            "actual_bets": total_bet // BET_UNIT,
            "hit_rate": float(hit_rate),
            "roi": float(roi),
            "profit": total_payout - total_bet,
            "avg_odds": float(avg_odds),
            "avg_ev": float(avg_ev),
            "avg_model_prob": float(avg_prob),
            "avg_market_prob": float(market_prob),
        })

    return results


def _calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Calibration analysis: compare predicted probabilities with actual outcomes.

    Bins predictions into quantile-based buckets and computes actual win rate
    in each bin. Perfect calibration: predicted_prob == actual_rate.
    """
    # Use quantile-based bins for equal sample sizes
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)

    results = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n = mask.sum()
        if n == 0:
            continue

        avg_pred = y_prob[mask].mean()
        actual_rate = y_true[mask].mean()
        gap = actual_rate - avg_pred

        results.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": int(n),
            "avg_pred": float(avg_pred),
            "actual_rate": float(actual_rate),
            "gap": float(gap),
        })

    return results


def find_best_threshold(thresholds: list[dict], min_bets: int = 500) -> dict | None:
    """Find the threshold with best ROI among those with sufficient bets."""
    candidates = [t for t in thresholds if t["actual_bets"] >= min_bets]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t["roi"])


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_boat1_model(model: LGBMClassifier, output_dir: str) -> str:
    """Save boat1 binary classifier to disk."""
    path = Path(output_dir) / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return str(path)


def load_boat1_model(model_dir: str) -> LGBMClassifier:
    """Load boat1 binary classifier from disk."""
    path = Path(model_dir) / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


_LGB_PARAM_KEYS = {
    "num_leaves", "max_depth", "min_child_samples",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
}


def load_boat1_training_params(model_dir: str) -> dict:
    """Load boat1 training hyperparameters from model_meta.json.

    Returns dict with keys matching train_boat1_model() arguments:
        extra_params: dict of LightGBM tree params
        n_estimators: int
        learning_rate: float
    """
    from .model import load_model_meta

    meta = load_model_meta(model_dir)
    if not meta or "hyperparameters" not in meta:
        return {
            "extra_params": {},
            "n_estimators": BOAT1_DEFAULT_PARAMS["n_estimators"],
            "learning_rate": BOAT1_DEFAULT_PARAMS["learning_rate"],
        }
    hp = dict(meta["hyperparameters"])
    return {
        "extra_params": {k: v for k, v in hp.items() if k in _LGB_PARAM_KEYS},
        "n_estimators": hp.get("n_estimators", BOAT1_DEFAULT_PARAMS["n_estimators"]),
        "learning_rate": hp.get("learning_rate", BOAT1_DEFAULT_PARAMS["learning_rate"]),
    }
