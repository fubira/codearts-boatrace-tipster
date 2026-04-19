"""Unified P2 ranker training pipeline.

train_ranking / train_dev_model / seed_stability_check が独立に実装していた
train ロジックを統合。val split 方式・early stopping 設定・feature_means
計算を一箇所にまとめ、同じ HP + 同じ DB で同じ model が生成されることを
保証する (2026-04-19、train_ranking の early_stopping=200 が feature 差を
masking していた inconsistency を受けて統合)。

Rules (CLAUDE.md との整合):
- val split: train data (race_date < end_date) 内の直近 val_months
- early_stopping_rounds: None (tune_p2 / train_dev_model と整合)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .feature_config import FEATURES
from .model import train_model

FIELD_SIZE = 6


def split_train_val(
    df: pd.DataFrame,
    end_date: str,
    val_months: int = 2,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (train_df, val_mask) with standardized val split.

    Train window: race_date < end_date. Val: last `val_months` within train.
    """
    train_df = df[df["race_date"] < end_date].copy()
    val_start = pd.Timestamp(end_date) - pd.DateOffset(months=val_months)
    val_mask = train_df["race_date"] >= str(val_start.date())
    return train_df, val_mask


def train_p2_ranker(
    df: pd.DataFrame,
    *,
    hp: dict,
    n_estimators: int,
    learning_rate: float,
    relevance_scheme: str,
    end_date: str,
    val_months: int = 2,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train a P2 LambdaRank model with the standardized pipeline.

    Args:
      df: features dataframe (full history, filtered by end_date inside).
      hp: LightGBM extra params (num_leaves, max_depth, ...).
      n_estimators / learning_rate: core LightGBM params.
      relevance_scheme: "podium" / "linear" / "top_heavy".
      end_date: training cutoff (exclusive, YYYY-MM-DD).
      val_months: val window length (default 2).
      seed: when given, overrides `random_state` in hp.

    Returns dict:
      - model: LightGBM Booster
      - feature_means: {col: mean}
      - n_train: train race count (entries/6)
      - n_val: val race count
      - date_range: "min ~ max" of train data race_date
    """
    train_df, val_mask = split_train_val(df, end_date, val_months)

    X = train_df[FEATURES].copy()
    y = train_df["finish_position"]
    meta = train_df[["race_id", "racer_id", "race_date", "boat_number"]].copy()

    extra = dict(hp)
    if seed is not None:
        extra["random_state"] = seed

    model, _ = train_model(
        X[~val_mask], y[~val_mask], meta[~val_mask],
        X[val_mask], y[val_mask], meta[val_mask],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        relevance_scheme=relevance_scheme,
        extra_params=extra,
        early_stopping_rounds=None,
    )

    feature_means = {
        c: float(X[c].astype("float64").mean()) for c in FEATURES
    }
    n_train = int((~val_mask).sum() // FIELD_SIZE)
    n_val = int(val_mask.sum() // FIELD_SIZE)
    dates_sorted = sorted(train_df["race_date"].unique())
    date_range = f"{dates_sorted[0]} ~ {dates_sorted[-1]}" if dates_sorted else ""

    return {
        "model": model,
        "feature_means": feature_means,
        "n_train": n_train,
        "n_val": n_val,
        "date_range": date_range,
    }
