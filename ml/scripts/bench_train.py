"""Benchmark a single full-training run with varying LightGBM num_threads.

Goal: identify whether Phase 1/Phase 2 wall time is bottlenecked by
LightGBM thread scaling, by data loading, or by something else.

Measures:
  - build_features_df time
  - feature_means computation time
  - train_model time (LightGBM core) at num_threads ∈ {1, 2, 4, 8}
  - predict time
  - evaluate_p2_strategy time

Runs with p2_v2 #294 HP (leaves=89, depth=7, n_est=1333, lr=0.0062).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model
from scripts.tune_p2 import FEATURES, _load_trifecta_odds, evaluate_p2_strategy

FIELD_SIZE = 6

HP = {
    "num_leaves": 89,
    "max_depth": 7,
    "min_child_samples": 64,
    "subsample": 0.7302700512037856,
    "colsample_bytree": 0.6240721896783721,
    "reg_alpha": 8.715701573782917e-06,
    "reg_lambda": 0.09885538716630769,
}
N_EST = 1333
LR = 0.0062
RELEVANCE = "podium"


def bench_load():
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(DEFAULT_DB_PATH, end_date="2026-01-01")
    t1 = time.time()
    print(f"  build_features_df(end_date=2026-01-01): {t1-t0:.2f}s, rows={len(df)}")

    t0 = time.time()
    odds = _load_trifecta_odds(DEFAULT_DB_PATH)
    t1 = time.time()
    print(f"  _load_trifecta_odds: {t1-t0:.2f}s, entries={len(odds)}")

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(DEFAULT_DB_PATH)
    t1 = time.time()
    print(f"  build_features_df(full): {t1-t0:.2f}s, rows={len(df_full)}")

    return df, df_full, odds


def bench_train(df, num_threads):
    val_start = pd.Timestamp("2026-01-01") - pd.DateOffset(months=2)
    val_mask = df["race_date"] >= str(val_start.date())

    t0 = time.time()
    X = df[FEATURES].copy()
    y = df["finish_position"]
    meta = df[["race_id", "racer_id", "race_date", "boat_number"]].copy()
    t1 = time.time()
    copy_time = t1 - t0

    hp = dict(HP)
    hp["num_threads"] = num_threads

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            X[~val_mask], y[~val_mask], meta[~val_mask],
            X[val_mask], y[val_mask], meta[val_mask],
            n_estimators=N_EST, learning_rate=LR,
            relevance_scheme=RELEVANCE, extra_params=hp,
            early_stopping_rounds=None,
        )
    t1 = time.time()
    train_time = t1 - t0

    return copy_time, train_time, model


def bench_predict_and_eval(model, df_full, odds):
    oos = df_full[(df_full["race_date"] >= "2026-01-01") & (df_full["race_date"] <= "2026-04-13")]
    oos = oos.sort_values(["race_id", "boat_number"]).reset_index(drop=True)

    t0 = time.time()
    X = oos[FEATURES]
    scores = model.predict(X)
    t1 = time.time()
    predict_time = t1 - t0

    t0 = time.time()
    meta = oos[["race_id", "boat_number"]].copy()
    meta["finish_position"] = oos["finish_position"].values
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=odds,
        gap23_threshold=0.13,
        ev_threshold=0.0,
        top3_conc_threshold=0.60,
        gap12_min_threshold=0.04,
    )
    t1 = time.time()
    eval_time = t1 - t0

    return predict_time, eval_time, result


def main():
    print(f"CPU: {os.cpu_count()} logical cores")
    print(f"n_estimators: {N_EST}")
    print()

    print("Loading data...")
    df, df_full, odds = bench_load()
    print()

    print("Training at various num_threads:")
    for n_threads in [1, 2, 4, 8]:
        copy_t, train_t, model = bench_train(df, n_threads)
        pred_t, eval_t, result = bench_predict_and_eval(model, df_full, odds)
        print(f"  num_threads={n_threads}: "
              f"copy={copy_t:.2f}s train={train_t:.2f}s "
              f"predict={pred_t:.2f}s eval={eval_t:.2f}s "
              f"races={result['races']} ROI={result['roi']:.3f}")


if __name__ == "__main__":
    main()
