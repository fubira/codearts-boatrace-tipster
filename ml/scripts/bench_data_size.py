"""Benchmark training time vs OOS P/L for different train_start cutoffs.

Question: do we really need 24 months of training data? If the model
trained on 12 months gives ~same OOS P/L, we halve LightGBM training
time per trial — a massive tune speedup.

Measures for each train_start ∈ {2024-01-01, 2024-07-01, 2025-01-01, 2025-04-01}:
  - n_train (races after cutoff)
  - train_model wall time (num_threads=2, matching tune config)
  - OOS 2026-01-01..04-13 P/L, ROI, hit%

Uses p2_v2 #294 HP.
"""

from __future__ import annotations

import contextlib
import io
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
    "num_threads": 2,  # match server tune config
}
N_EST = 1333
LR = 0.0062
RELEVANCE = "podium"


def train_and_eval(df, df_full, odds, train_start: str, end_date: str = "2026-01-01"):
    train_df = df_full[(df_full["race_date"] >= train_start) & (df_full["race_date"] < end_date)]

    val_start_dt = pd.Timestamp(end_date) - pd.DateOffset(months=2)
    val_mask = train_df["race_date"] >= str(val_start_dt.date())

    X = train_df[FEATURES].copy()
    y = train_df["finish_position"]
    meta = train_df[["race_id", "racer_id", "race_date", "boat_number"]].copy()
    n_train = int((~val_mask).sum() // FIELD_SIZE)
    n_val = int(val_mask.sum() // FIELD_SIZE)

    # Train
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            X[~val_mask], y[~val_mask], meta[~val_mask],
            X[val_mask], y[val_mask], meta[val_mask],
            n_estimators=N_EST, learning_rate=LR,
            relevance_scheme=RELEVANCE, extra_params=dict(HP),
            early_stopping_rounds=None,
        )
    train_time = time.time() - t0

    # Eval on OOS
    oos = df_full[(df_full["race_date"] >= "2026-01-01") & (df_full["race_date"] <= "2026-04-13")]
    oos = oos.sort_values(["race_id", "boat_number"]).reset_index(drop=True)
    scores = model.predict(oos[FEATURES])
    eval_meta = oos[["race_id", "boat_number"]].copy()
    eval_meta["finish_position"] = oos["finish_position"].values
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=eval_meta,
        trifecta_odds=odds,
        gap23_threshold=0.13,
        ev_threshold=0.0,
        top3_conc_threshold=0.60,
        gap12_min_threshold=0.04,
    )
    pl = result["payout"] - result["cost"]
    hit_pct = 100 * result["wins"] / result["races"] if result["races"] else 0.0

    return {
        "train_start": train_start,
        "n_train": n_train,
        "n_val": n_val,
        "train_time": train_time,
        "races": result["races"],
        "wins": result["wins"],
        "hit_pct": hit_pct,
        "roi": result["roi"],
        "pl": pl,
    }


def main():
    print("Loading full features...", file=sys.stderr)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(DEFAULT_DB_PATH)
    print(f"  {time.time()-t0:.1f}s rows={len(df_full)}", file=sys.stderr)

    odds = _load_trifecta_odds(DEFAULT_DB_PATH)

    cutoffs = [
        "2024-01-01",  # 24mo (current, BC missing for first 3mo)
        "2024-02-01",  # 23mo
        "2024-03-01",  # 22mo
        "2024-04-01",  # 21mo ← BC data fully available from here
        "2024-05-01",  # 20mo
        "2024-07-01",  # 18mo (control from previous bench)
    ]
    results = []
    for c in cutoffs:
        print(f"\n[{c}] training...", file=sys.stderr)
        r = train_and_eval(df_full, df_full, odds, c)
        results.append(r)
        print(f"  n_train={r['n_train']} time={r['train_time']:.1f}s "
              f"races={r['races']} hit%={r['hit_pct']:.1f} "
              f"ROI={r['roi']:.3f} P/L={r['pl']:+,.0f}",
              file=sys.stderr)

    # Table
    print()
    print("=" * 100)
    print(f"{'train_start':<12} {'n_train':>8} {'train(s)':>10} "
          f"{'vs full':>9} {'races':>6} {'hit%':>6} {'ROI':>6} {'P/L':>11}")
    print("-" * 100)
    full_time = results[0]["train_time"]
    for r in results:
        ratio = r["train_time"] / full_time
        print(f"{r['train_start']:<12} {r['n_train']:>8} {r['train_time']:>10.1f} "
              f"{ratio:>8.2f}x {r['races']:>6} {r['hit_pct']:>5.1f}% "
              f"{r['roi']:>6.3f} {r['pl']:>+11,.0f}")


if __name__ == "__main__":
    main()
