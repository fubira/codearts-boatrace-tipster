"""Conc threshold sweep for fixed HPs.

For each (HP, conc) pair, run the same WF-CV that tune_p2 uses and report
growth/ROI/races. Shares fold structure with tune_p2 so numbers are directly
comparable to a tune log.

Each HP is trained ONCE per fold; conc is a post-hoc filter so we evaluate
all conc values from a single set of fold predictions.

Usage:
    cd ml && uv run python scripts/conc_sweep.py \\
        --tune-log ../logs/tune/2026-04-14_1036_server-tune.log \\
        --trials 0,369,240,309,403,237 \\
        --conc-values 0.40,0.45,0.50,0.55,0.60,0.65 \\
        --gap12 0.04
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

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits
from scripts.train_dev_model import parse_tune_log, params_to_hp
from scripts.tune_p2 import _load_trifecta_odds, evaluate_p2_strategy

FIELD_SIZE = 6
BANKROLL = 70000.0


def _fold_growth(fold_profits: list[float], days_per_fold: int) -> float:
    rates = [
        np.log(max(1 + (fp / days_per_fold) / BANKROLL, 1e-6))
        for fp in fold_profits
    ]
    return float(np.mean(rates)) if rates else -999.0


def _train_and_predict(fold, hp, lr, n_est, relevance):
    """Train one fold model and return (rank_scores on test, best_iter)."""
    extra = dict(hp)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            fold["train"]["X"], fold["train"]["y"], fold["train"]["meta"],
            fold["val"]["X"], fold["val"]["y"], fold["val"]["meta"],
            n_estimators=n_est,
            learning_rate=lr,
            relevance_scheme=relevance,
            extra_params=extra,
            early_stopping_rounds=200,
        )
    rank_scores = model.predict(fold["test"]["X"])
    best_it = getattr(model, "best_iteration_", None) or n_est
    return rank_scores, best_it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-log", required=True)
    ap.add_argument("--trials", required=True,
                    help="Comma-separated trial numbers (e.g. '0,369,240,309')")
    ap.add_argument("--conc-values", default="0.40,0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--gap12-values", default="0.04",
                    help="Comma-separated gap12_min_threshold values to sweep")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--fold-months", type=int, default=2)
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    wanted = [int(t.strip()) for t in args.trials.split(",")]
    conc_values = [float(c.strip()) for c in args.conc_values.split(",")]
    gap12_values = [float(g.strip()) for g in args.gap12_values.split(",")]

    print(f"Parsing {args.tune_log}...", file=sys.stderr)
    log_info = parse_tune_log(Path(args.tune_log))
    fix_th = log_info["fix_thresholds"]
    gap23_th = fix_th.get("gap23", 0.13)
    ev_th = fix_th.get("ev", 0.0)

    by_num = log_info["trials"]
    selected = []
    for num in wanted:
        if num not in by_num:
            print(f"WARN: trial #{num} not in log, skip", file=sys.stderr)
            continue
        selected.append((num, by_num[num]))
    if not selected:
        print("No trials.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading features...", file=sys.stderr)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)
    print(f"  features in {time.time()-t0:.1f}s, rows={len(df)}", file=sys.stderr)

    print(f"Loading 3連単 odds...", file=sys.stderr)
    trifecta_odds = _load_trifecta_odds(args.db_path)

    X = df[FEATURES].copy()
    y = df["finish_position"]
    meta = df[["race_id", "racer_id", "race_date", "boat_number", "finish_position"]].copy()

    folds = walk_forward_splits(X, y, meta, n_folds=args.n_folds, fold_months=args.fold_months)
    print(f"Folds: {len(folds)}", file=sys.stderr)
    for i, fold in enumerate(folds):
        n_test = len(fold["test"]["X"]) // FIELD_SIZE
        print(f"  Fold {i+1}: test={fold['period']['test']} ({n_test}R)", file=sys.stderr)

    days_per_fold = args.fold_months * 30
    all_rows: list[dict] = []

    for ti, (num, trial) in enumerate(selected, 1):
        params = trial["params"]
        # params_to_hp expects fix-threshold defaults
        hp, lr, n_est, conc_orig, gap12_orig = params_to_hp(
            params, conc_default=0.0, gap12_default=0.0,
        )
        ua = trial.get("user_attrs") or {}
        relevance = params.get("relevance") or ua.get("relevance") or "podium"

        print(f"\n[{ti}/{len(selected)}] trial #{num} "
              f"(leaves={params.get('num_leaves')} depth={params.get('max_depth')} "
              f"l2={params.get('reg_lambda'):.4f} lr={lr:.4f} n_est_ub={n_est})",
              file=sys.stderr)
        print(f"    relevance={relevance} reported growth={ua.get('growth')}",
              file=sys.stderr)

        # Train each fold once, store (rank_scores, test_meta_with_y)
        fold_predictions = []
        train_t0 = time.time()
        for fi, fold in enumerate(folds):
            rank_scores, best_it = _train_and_predict(fold, hp, lr, n_est, relevance)
            test_meta = fold["test"]["meta"].copy()
            test_meta["finish_position"] = fold["test"]["y"].values
            fold_predictions.append((rank_scores, test_meta, best_it))
            print(f"    fold {fi+1} trained ({time.time()-train_t0:.0f}s, best_it={best_it})",
                  file=sys.stderr)

        # Evaluate at every (conc, gap12) pair
        for gv in gap12_values:
            for cv in conc_values:
                fold_profits = []
                fold_rois = []
                fold_races = []
                for rank_scores, test_meta, _ in fold_predictions:
                    result = evaluate_p2_strategy(
                        rank_scores=rank_scores,
                        meta_rank=test_meta,
                        trifecta_odds=trifecta_odds,
                        gap23_threshold=gap23_th,
                        ev_threshold=ev_th,
                        top3_conc_threshold=cv,
                        gap12_min_threshold=gv,
                    )
                    fp = result["payout"] - result["cost"]
                    fold_profits.append(fp)
                    fold_rois.append(result["roi"] if result["cost"] > 0 else 0)
                    fold_races.append(result["races"])

                growth = _fold_growth(fold_profits, days_per_fold)
                mean_roi = float(np.mean(fold_rois))
                total_races = sum(fold_races)
                total_profit = sum(fold_profits)

                all_rows.append({
                    "trial": num,
                    "conc": cv,
                    "gap12": gv,
                    "growth": growth,
                    "roi": mean_roi,
                    "races": total_races,
                    "profit": total_profit,
                    "fold_rois": fold_rois,
                })

    # Print pivot table per gap12 value
    for gv in gap12_values:
        print()
        print("=" * 90)
        print(f"Conc sweep result (gap12={gv:.2f}, gap23={gap23_th:.2f}, ev={ev_th:.2f})")
        print("=" * 90)

        print()
        print("Growth by (trial × conc):")
        header = "  trial  | " + " | ".join(f"conc={c:.2f}" for c in conc_values)
        print(header)
        print("-" * len(header))
        for num, _ in selected:
            cells = []
            for cv in conc_values:
                row = next(r for r in all_rows if r["trial"] == num and r["conc"] == cv and r["gap12"] == gv)
                cells.append(f"{row['growth']:.6f}")
            print(f"  #{num:>5} | " + " | ".join(c.center(8) for c in cells))

        print()
        print("ROI by (trial × conc):")
        print(header)
        print("-" * len(header))
        for num, _ in selected:
            cells = []
            for cv in conc_values:
                row = next(r for r in all_rows if r["trial"] == num and r["conc"] == cv and r["gap12"] == gv)
                cells.append(f"{row['roi']:.3f}")
            print(f"  #{num:>5} | " + " | ".join(c.center(8) for c in cells))

        print()
        print("Races by (trial × conc):")
        print(header)
        print("-" * len(header))
        for num, _ in selected:
            cells = []
            for cv in conc_values:
                row = next(r for r in all_rows if r["trial"] == num and r["conc"] == cv and r["gap12"] == gv)
                cells.append(f"{row['races']}")
            print(f"  #{num:>5} | " + " | ".join(c.center(8) for c in cells))

        print()
        print("Profit by (trial × conc):")
        print(header)
        print("-" * len(header))
        for num, _ in selected:
            cells = []
            for cv in conc_values:
                row = next(r for r in all_rows if r["trial"] == num and r["conc"] == cv and r["gap12"] == gv)
                cells.append(f"{row['profit']:+,.0f}")
            print(f"  #{num:>5} | " + " | ".join(c.center(9) for c in cells))


if __name__ == "__main__":
    main()
