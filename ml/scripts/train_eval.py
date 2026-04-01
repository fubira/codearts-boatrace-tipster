"""Train and evaluate a boatrace prediction model.

Usage:
    uv run --directory ml python -m scripts.train_eval [options]

Options:
    --mode single|wfcv     Evaluation mode (default: single)
    --start-date DATE      Training data start date (default: all)
    --n-folds N            WF-CV fold count (default: 4)
    --fold-months N        WF-CV fold width in months (default: 2)
    --n-estimators N       LightGBM tree count (default: 500)
    --learning-rate F      Learning rate (default: 0.05)
    --relevance SCHEME     Relevance scheme (default: linear)
    --seed N               Random seed (default: 42)
    --params JSON          Extra LightGBM params as JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from boatrace_tipster_ml.evaluate import evaluate_model
from boatrace_tipster_ml.features import build_features
from boatrace_tipster_ml.model import (
    time_series_split,
    train_model,
    walk_forward_splits,
)

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "boatrace-tipster.db")


def _print_metrics(result: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    acc = result["topNAccuracy"]
    hit = result["multiHitRates"]
    print(f"{prefix}Top1={acc['1']:.1%} Top2={acc['2']:.1%} Top3={acc['3']:.1%} nDCG={result['avgNDCG']:.4f}")
    print(f"{prefix}2連単={hit['2連単']:.1%} 3連単={hit['3連単']:.1%}")

    if "payoutROI" in result:
        print(f"{prefix}Payout ROI:")
        for name, s in result["payoutROI"].items():
            print(f"  {name}: ROI={s['recoveryRate']:.1%} hit={s['hitRate']:.1%} bets={s['betCount']}")

    if "confidenceAnalysis" in result:
        print(f"{prefix}Confidence Analysis:")
        ca = result["confidenceAnalysis"]
        # Show 2連単 by percentile
        for entry in ca:
            if entry["betType"] == "2連単":
                print(
                    f"  p{entry['percentile']:>2d} (>{entry['threshold']:.3f}): "
                    f"ROI={entry['recoveryRate']:.1%} hit={entry['hitRate']:.1%} "
                    f"bets={entry['betCount']}"
                )


def run_single(args, X, y, meta) -> dict:
    """Single train/val/test split evaluation."""
    splits = time_series_split(X, y, meta)

    for name, data in splits.items():
        n_races = len(data["X"]) // 6
        print(f"  {name}: {n_races} races")

    print("\nTraining...")
    t0 = time.time()
    model, metrics = train_model(
        splits["train"]["X"],
        splits["train"]["y"],
        splits["train"]["meta"],
        splits["val"]["X"],
        splits["val"]["y"],
        splits["val"]["meta"],
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        relevance_scheme=args.relevance,
        extra_params=json.loads(args.params) if args.params else None,
    )
    print(f"  Trained in {time.time() - t0:.1f}s")

    # Feature importance
    imp = sorted(metrics["feature_importance"].items(), key=lambda x: -x[1])
    print("\nFeature importance (top 15):")
    for name, score in imp[:15]:
        bar = "█" * int(score * 100)
        print(f"  {name:30s} {score:.4f} {bar}")

    low_imp = [name for name, score in imp if score < 0.01]
    if low_imp:
        print(f"\n  Low importance (<1%): {', '.join(low_imp)}")

    print("\nEvaluating on test set...")
    result = evaluate_model(
        model,
        splits["test"]["X"],
        splits["test"]["y"],
        splits["test"]["meta"],
        db_path=DB_PATH,
    )
    _print_metrics(result)

    return result


def run_wfcv(args, X, y, meta) -> dict:
    """Walk-Forward Cross-Validation evaluation."""
    folds = walk_forward_splits(
        X,
        y,
        meta,
        n_folds=args.n_folds,
        fold_months=args.fold_months,
        train_start=args.start_date,
    )

    if not folds:
        print("ERROR: No valid folds generated")
        sys.exit(1)

    print(f"Generated {len(folds)} folds:")
    for i, fold in enumerate(folds):
        p = fold["period"]
        n_train = len(fold["train"]["X"]) // 6
        n_test = len(fold["test"]["X"]) // 6
        print(f"  Fold {i+1}: train={n_train}R test={p['test']} ({n_test}R)")

    fold_results = []
    for i, fold in enumerate(folds):
        print(f"\n--- Fold {i+1}/{len(folds)} ({fold['period']['test']}) ---")

        model, _ = train_model(
            fold["train"]["X"],
            fold["train"]["y"],
            fold["train"]["meta"],
            fold["val"]["X"],
            fold["val"]["y"],
            fold["val"]["meta"],
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            relevance_scheme=args.relevance,
            extra_params=json.loads(args.params) if args.params else None,
        )

        result = evaluate_model(
            model,
            fold["test"]["X"],
            fold["test"]["y"],
            fold["test"]["meta"],
            db_path=DB_PATH,
        )
        _print_metrics(result, f"Fold {i+1}")
        fold_results.append(result)

    # Aggregate across folds
    print("\n" + "=" * 60)
    print("Walk-Forward CV Summary")
    print("=" * 60)

    ndcgs = [r["avgNDCG"] for r in fold_results]
    top1s = [r["topNAccuracy"]["1"] for r in fold_results]
    exacta_hits = [r["multiHitRates"]["2連単"] for r in fold_results]

    print(f"nDCG:   {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f} (min={np.min(ndcgs):.4f} max={np.max(ndcgs):.4f})")
    print(f"Top1:   {np.mean(top1s):.1%} ± {np.std(top1s):.1%}")
    print(f"2連単:  {np.mean(exacta_hits):.1%} ± {np.std(exacta_hits):.1%}")

    # Aggregate payout ROI
    if all("payoutROI" in r for r in fold_results):
        print("\nPayout ROI (per fold):")
        bet_types = fold_results[0]["payoutROI"].keys()
        for bt in bet_types:
            rois = [r["payoutROI"][bt]["recoveryRate"] for r in fold_results if bt in r["payoutROI"]]
            if rois:
                print(
                    f"  {bt}: median={np.median(rois):.1%} "
                    f"mean={np.mean(rois):.1%} ± {np.std(rois):.1%} "
                    f"min={np.min(rois):.1%} max={np.max(rois):.1%}"
                )

    return {"folds": fold_results, "summary": {"ndcg_mean": np.mean(ndcgs), "top1_mean": np.mean(top1s)}}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate boatrace prediction model")
    parser.add_argument("--mode", choices=["single", "wfcv"], default="single")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--relevance", default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", default=None, help="Extra LightGBM params as JSON string")
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Trees: {args.n_estimators}, LR: {args.learning_rate}")
    print(f"Relevance: {args.relevance}")
    if args.start_date:
        print(f"Train start: {args.start_date}")
    print()

    t0 = time.time()
    X, y, meta = build_features(DB_PATH, start_date=args.start_date)
    print(f"Features: {X.shape[0]} entries ({X.shape[0]//6} races), {X.shape[1]} features\n")

    if args.mode == "single":
        run_single(args, X, y, meta)
    else:
        run_wfcv(args, X, y, meta)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
