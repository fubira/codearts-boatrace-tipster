"""Re-run p2_v2 #294 HP through WF-CV at the EXACT original tune fold
dates (test_end=2026-04-11) and compare to original tune log values.

Original tune log (2026-04-12_1713) reported for #294:
    growth=0.003004 ROI=158% P/L=+50,550 races=734 conc=0.448

Today's same HP (warm-start trial 0 of 4-14 tune) reports:
    growth=0.001509 ROI=130% P/L=+25,370 races=755

If today's run with original fold dates matches 0.003 → fold-date shift
is the cause. If it still gives ~0.0015 → DB content or feature pipeline
changed.
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
from boatrace_tipster_ml.model import train_model, walk_forward_splits
from scripts.tune_p2 import FEATURES, _load_trifecta_odds, evaluate_p2_strategy

FIELD_SIZE = 6
BANKROLL = 70000.0
DAYS_PER_FOLD = 60

# p2_v2 #294 HPs (from logs/tune/2026-04-12_1713_server-tune.log)
HP = {
    "num_leaves": 89,
    "max_depth": 7,
    "min_child_samples": 64,
    "subsample": 0.7302700512037856,
    "colsample_bytree": 0.6240721896783721,
    "reg_alpha": 8.715701573782917e-06,
    "reg_lambda": 0.09885538716630769,
}
N_EST_UPPER = 1333
LR = 0.006172936736632081
RELEVANCE = "podium"
GAP23 = 0.13
EV = 0.0
CONC = 0.4480319663154616
GAP12 = 0.0  # original had no gap12


def _train(fold):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            fold["train"]["X"], fold["train"]["y"], fold["train"]["meta"],
            fold["val"]["X"], fold["val"]["y"], fold["val"]["meta"],
            n_estimators=N_EST_UPPER, learning_rate=LR,
            relevance_scheme=RELEVANCE, extra_params=dict(HP),
            early_stopping_rounds=200,
        )
    return model.predict(fold["test"]["X"]), getattr(model, "best_iteration_", N_EST_UPPER)


def _eval(rank_scores, test_meta, trifecta_odds):
    return evaluate_p2_strategy(
        rank_scores=rank_scores, meta_rank=test_meta,
        trifecta_odds=trifecta_odds,
        gap23_threshold=GAP23, ev_threshold=EV,
        top3_conc_threshold=CONC, gap12_min_threshold=GAP12,
    )


def main():
    print("Loading features...", file=sys.stderr)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(DEFAULT_DB_PATH)
    print(f"  features in {time.time()-t0:.1f}s, rows={len(df)}", file=sys.stderr)

    trifecta_odds = _load_trifecta_odds(DEFAULT_DB_PATH)

    X = df[FEATURES].copy()
    y = df["finish_position"]
    meta = df[["race_id", "racer_id", "race_date", "boat_number", "finish_position"]].copy()

    print()
    for label, test_end in [
        ("ORIGINAL fold dates (test_end=2026-04-11)", "2026-04-11"),
        ("TODAY fold dates (test_end=None → max date)", None),
    ]:
        print(f"=== {label} ===")
        folds = walk_forward_splits(
            X, y, meta, n_folds=4, fold_months=2, test_end=test_end,
        )
        for i, fold in enumerate(folds):
            n_test = len(fold["test"]["X"]) // FIELD_SIZE
            n_train = len(fold["train"]["X"]) // FIELD_SIZE
            print(f"  Fold {i+1}: test={fold['period']['test']} test_R={n_test} train_R={n_train}")

        fold_profits = []
        fold_rois = []
        fold_races = []
        fold_best = []
        for i, fold in enumerate(folds):
            t1 = time.time()
            scores, best_it = _train(fold)
            test_meta = fold["test"]["meta"].copy()
            test_meta["finish_position"] = fold["test"]["y"].values
            r = _eval(scores, test_meta, trifecta_odds)
            fold_profits.append(r["payout"] - r["cost"])
            fold_rois.append(r["roi"] if r["cost"] > 0 else 0)
            fold_races.append(r["races"])
            fold_best.append(best_it)
            print(f"    f{i+1} best_it={best_it} buys={r['races']} "
                  f"ROI={r['roi']:.3f} profit={r['payout']-r['cost']:+,.0f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

        rates = [
            np.log(max(1 + (fp / DAYS_PER_FOLD) / BANKROLL, 1e-6))
            for fp in fold_profits
        ]
        growth = float(np.mean(rates))
        total_profit = sum(fold_profits)
        total_races = sum(fold_races)
        mean_roi = float(np.mean(fold_rois))
        print(f"  TOTAL: growth={growth:.6f} mean_ROI={mean_roi:.3f} "
              f"races={total_races} profit={total_profit:+,.0f}")
        print()

    print("EXPECTED from original 4-12 tune log:")
    print("  #294: growth=0.003004 ROI=158% P/L=+50,550 races=734")


if __name__ == "__main__":
    main()
