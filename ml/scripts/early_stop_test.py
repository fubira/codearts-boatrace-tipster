"""Test whether per-fold early stopping is structurally biasing WF-CV
against shallow/regularized HPs (specifically p2_v2 HP).

For each selected trial:
  variant A: train with early_stopping_rounds=200 (matches tune_p2)
  variant B: train with early_stopping_rounds=None, n_estimators = upper bound
             (matches production p2_v2 which trains for full n_estimators)

Compare WF-CV growth across the two variants for each HP. Hypothesis:
shallow+regularized HPs (p2_v2 trial 0) recover significantly under
variant B, while deep+complex HPs do not.

Usage:
    cd ml && uv run python scripts/early_stop_test.py \\
        --tune-log ../logs/tune/2026-04-14_1036_server-tune.log \\
        --trials 0,369 --conc 0.60 --gap12 0.04
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits
from scripts.train_dev_model import parse_tune_log, params_to_hp
from scripts.tune_p2 import FEATURES, _load_trifecta_odds, evaluate_p2_strategy

FIELD_SIZE = 6
BANKROLL = 70000.0


def _fold_growth(fold_profits, days_per_fold):
    rates = [
        np.log(max(1 + (fp / days_per_fold) / BANKROLL, 1e-6))
        for fp in fold_profits
    ]
    return float(np.mean(rates)) if rates else -999.0


def _train(fold, hp, lr, n_est, relevance, *, early_stop):
    extra = dict(hp)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            fold["train"]["X"], fold["train"]["y"], fold["train"]["meta"],
            fold["val"]["X"], fold["val"]["y"], fold["val"]["meta"],
            n_estimators=n_est, learning_rate=lr,
            relevance_scheme=relevance, extra_params=extra,
            early_stopping_rounds=200 if early_stop else None,
        )
    rank_scores = model.predict(fold["test"]["X"])
    best_it = getattr(model, "best_iteration_", None) or n_est
    return rank_scores, best_it


def _evaluate_all_folds(rank_scores_per_fold, fold_metas, trifecta_odds,
                         gap23, ev, conc, gap12, days_per_fold):
    fold_profits = []
    fold_rois = []
    fold_races = []
    for rank_scores, test_meta in zip(rank_scores_per_fold, fold_metas):
        result = evaluate_p2_strategy(
            rank_scores=rank_scores, meta_rank=test_meta,
            trifecta_odds=trifecta_odds,
            gap23_threshold=gap23, ev_threshold=ev,
            top3_conc_threshold=conc, gap12_min_threshold=gap12,
        )
        fold_profits.append(result["payout"] - result["cost"])
        fold_rois.append(result["roi"] if result["cost"] > 0 else 0)
        fold_races.append(result["races"])
    return {
        "growth": _fold_growth(fold_profits, days_per_fold),
        "mean_roi": float(np.mean(fold_rois)),
        "races": sum(fold_races),
        "profit": sum(fold_profits),
        "fold_rois": fold_rois,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-log", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--conc", type=float, default=0.60)
    ap.add_argument("--gap12", type=float, default=0.04)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--fold-months", type=int, default=2)
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    wanted = [int(t.strip()) for t in args.trials.split(",")]

    print(f"Parsing {args.tune_log}...", file=sys.stderr)
    log_info = parse_tune_log(Path(args.tune_log))
    fix_th = log_info["fix_thresholds"]
    gap23_th = fix_th.get("gap23", 0.13)
    ev_th = fix_th.get("ev", 0.0)

    by_num = log_info["trials"]
    selected = [(n, by_num[n]) for n in wanted if n in by_num]
    if not selected:
        sys.exit(1)

    print("Loading features...", file=sys.stderr)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)
    print(f"  features in {time.time()-t0:.1f}s", file=sys.stderr)

    trifecta_odds = _load_trifecta_odds(args.db_path)

    X = df[FEATURES].copy()
    y = df["finish_position"]
    meta = df[["race_id", "racer_id", "race_date", "boat_number", "finish_position"]].copy()
    folds = walk_forward_splits(X, y, meta, n_folds=args.n_folds, fold_months=args.fold_months)
    fold_metas = []
    for fold in folds:
        m = fold["test"]["meta"].copy()
        m["finish_position"] = fold["test"]["y"].values
        fold_metas.append(m)
    print(f"Folds: {len(folds)}", file=sys.stderr)

    days_per_fold = args.fold_months * 30
    rows = []

    for ti, (num, trial) in enumerate(selected, 1):
        params = trial["params"]
        hp, lr, n_est_upper, _, _ = params_to_hp(params, conc_default=0.0, gap12_default=0.0)
        ua = trial.get("user_attrs") or {}
        relevance = params.get("relevance") or ua.get("relevance") or "podium"
        print(f"\n[{ti}/{len(selected)}] trial #{num} "
              f"leaves={params['num_leaves']} depth={params['max_depth']} "
              f"l2={params['reg_lambda']:.4f} lr={lr:.4f} n_est_ub={n_est_upper}",
              file=sys.stderr)

        # variant A: with early stopping
        t1 = time.time()
        scores_A = []
        best_iters_A = []
        for fi, fold in enumerate(folds):
            rs, bi = _train(fold, hp, lr, n_est_upper, relevance, early_stop=True)
            scores_A.append(rs)
            best_iters_A.append(bi)
            print(f"  A f{fi+1} {time.time()-t1:.0f}s best_it={bi}", file=sys.stderr)
        res_A = _evaluate_all_folds(
            scores_A, fold_metas, trifecta_odds,
            gap23_th, ev_th, args.conc, args.gap12, days_per_fold,
        )

        # variant B: full n_estimators (matches production p2_v2 setup)
        t1 = time.time()
        scores_B = []
        for fi, fold in enumerate(folds):
            rs, bi = _train(fold, hp, lr, n_est_upper, relevance, early_stop=False)
            scores_B.append(rs)
            print(f"  B f{fi+1} {time.time()-t1:.0f}s iters={bi}", file=sys.stderr)
        res_B = _evaluate_all_folds(
            scores_B, fold_metas, trifecta_odds,
            gap23_th, ev_th, args.conc, args.gap12, days_per_fold,
        )

        rows.append({
            "trial": num,
            "n_est_upper": n_est_upper,
            "best_iters_A": best_iters_A,
            "A": res_A,
            "B": res_B,
        })

    print()
    print("=" * 95)
    print(f"Early stopping comparison (conc={args.conc} gap12={args.gap12} gap23={gap23_th} ev={ev_th})")
    print("=" * 95)
    print(f"{'trial':>5} {'variant':>8} {'n_est':>6} {'growth':>10} {'roi':>7} {'races':>6} {'profit':>11}")
    print("-" * 95)
    for r in rows:
        print(f"  #{r['trial']:>3} {'A(es)':>8} {r['best_iters_A']!s:>6} {r['A']['growth']:>10.6f} {r['A']['mean_roi']:>7.3f} {r['A']['races']:>6} {r['A']['profit']:>+11,.0f}")
        print(f"  {'':>4} {'B(full)':>8} {r['n_est_upper']:>6} {r['B']['growth']:>10.6f} {r['B']['mean_roi']:>7.3f} {r['B']['races']:>6} {r['B']['profit']:>+11,.0f}")
        delta = r['B']['growth'] - r['A']['growth']
        ratio = (r['B']['growth'] / r['A']['growth']) if r['A']['growth'] != 0 else float('nan')
        print(f"  {'':>4} {'Δ':>8} {'':>6} {delta:>+10.6f} {'':>7} {'':>6} (ratio B/A = {ratio:.2f}x)")
        print()


if __name__ == "__main__":
    main()
