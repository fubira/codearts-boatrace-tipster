"""Train and evaluate a boat 1 win prediction binary classifier.

Usage:
    uv run --directory ml python -m scripts.train_boat1_binary [options]

Options:
    --mode single|wfcv|optuna  Evaluation mode (default: single)
    --start-date DATE          Training data start date (default: all)
    --n-folds N                WF-CV fold count (default: 4)
    --fold-months N            WF-CV fold width in months (default: 2)
    --n-estimators N           LightGBM tree count (default: 500)
    --learning-rate F          Learning rate (default: 0.05)
    --seed N                   Random seed (default: 42)
    --params JSON              Extra LightGBM params as JSON
    --n-trials N               Optuna trial count (default: 100)
    --objective ev_roi|upset_fbeta  Optuna objective (default: ev_roi)
    --beta F                   F-beta beta for upset_fbeta (default: 1.5)
"""

import argparse
import json
import sys
import time

import numpy as np

from boatrace_tipster_ml.boat1_features import BOAT1_FEATURE_COLS, reshape_to_boat1
from boatrace_tipster_ml.boat1_model import (
    evaluate_boat1,
    find_best_threshold,
    save_boat1_model,
    train_boat1_model,
)
from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.evaluate import load_payouts
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import save_model_meta, time_series_split, walk_forward_splits

DB_PATH = DEFAULT_DB_PATH


def _print_thresholds(thresholds: list[dict]) -> None:
    print(f"  {'Thresh':>6s}  {'Selected':>8s}  {'HitRate':>7s}  {'ROI':>7s}  {'Profit':>10s}")
    print(f"  {'-' * 48}")
    for t in thresholds:
        profit_str = f"+¥{t['profit']:,}" if t["profit"] >= 0 else f"-¥{abs(t['profit']):,}"
        print(
            f"  {t['threshold']:.2f}    {t['actual_bets']:>8d}  "
            f"{t['hit_rate']:>6.1%}  {t['roi']:>6.1%}  {profit_str:>10s}"
        )


def _print_ev(ev_results: list[dict]) -> None:
    print(f"  {'EV≥':>5s}  {'Bets':>6s}  {'HitRate':>7s}  {'ROI':>7s}  {'Profit':>10s}  "
          f"{'AvgOdds':>7s}  {'Model%':>7s}  {'Mkt%':>7s}  {'Edge':>6s}")
    print(f"  {'-' * 76}")
    for e in ev_results:
        profit_str = f"+¥{e['profit']:,}" if e["profit"] >= 0 else f"-¥{abs(e['profit']):,}"
        edge = e["avg_model_prob"] - e["avg_market_prob"]
        print(
            f"  {e['ev_threshold']:>+5d}  {e['actual_bets']:>6d}  "
            f"{e['hit_rate']:>6.1%}  {e['roi']:>6.1%}  {profit_str:>10s}  "
            f"{e['avg_odds']:>6.1f}  {e['avg_model_prob']:>6.1%}  "
            f"{e['avg_market_prob']:>6.1%}  {edge:>+5.1%}"
        )


def _print_calibration(calibration: list[dict]) -> None:
    print(f"  {'Bin':>13s}  {'N':>6s}  {'Pred%':>6s}  {'Actual%':>7s}  {'Gap':>6s}  {'':>20s}")
    print(f"  {'-' * 65}")
    for c in calibration:
        gap_bar = ""
        gap_pct = c["gap"] * 100
        if gap_pct > 0:
            gap_bar = "+" * min(int(gap_pct * 2), 20)
        else:
            gap_bar = "-" * min(int(-gap_pct * 2), 20)
        print(
            f"  [{c['bin_lo']:.2f}-{c['bin_hi']:.2f}]  {c['n']:>6d}  "
            f"{c['avg_pred']:>5.1%}  {c['actual_rate']:>6.1%}  "
            f"{c['gap']:>+5.1%}  {gap_bar}"
        )


def _print_importance(importance: dict, top_n: int = 15) -> None:
    imp = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\nFeature importance (top {top_n}):")
    for name, score in imp[:top_n]:
        bar = "█" * int(score * 100)
        print(f"  {name:35s} {score:.4f} {bar}")


def run_single(args, X, y, meta) -> dict:
    """Single train/val/test split evaluation."""
    splits = time_series_split(X, y, meta)

    for name, data in splits.items():
        print(f"  {name}: {len(data['X'])} races")

    print(f"\n  Boat 1 win rate: {y.mean():.1%} (overall)")
    for name, data in splits.items():
        rate = data["y"].mean()
        print(f"  {name}: {rate:.1%}")

    print("\nTraining...")
    t0 = time.time()
    model, metrics = train_boat1_model(
        splits["train"]["X"],
        splits["train"]["y"],
        splits["val"]["X"],
        splits["val"]["y"],
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        extra_params=json.loads(args.params) if args.params else None,
    )
    print(f"  Trained in {time.time() - t0:.1f}s")

    if "val_auc" in metrics:
        print(f"  Val AUC: {metrics['val_auc']:.4f}")

    _print_importance(metrics["feature_importance"])

    print("\nEvaluating on test set...")
    result = evaluate_boat1(
        model,
        splits["test"]["X"],
        splits["test"]["y"],
        splits["test"]["meta"],
        db_path=DB_PATH,
    )

    print(f"\nClassification Metrics:")
    print(f"  AUC:       {result['auc']:.4f}")
    print(f"  Accuracy:  {result['accuracy']:.1%}")
    print(f"  Base rate: {result['base_hit_rate']:.1%} (boat 1 wins)")
    print(f"  N races:   {result['n_races']}")

    if "thresholds" in result:
        print(f"\nThreshold Analysis (単勝1号艇):")
        _print_thresholds(result["thresholds"])

        best = find_best_threshold(result["thresholds"])
        if best:
            print(f"\n  Best threshold (≥500 bets): {best['threshold']:.2f} "
                  f"→ ROI {best['roi']:.1%}, hit {best['hit_rate']:.1%}, "
                  f"{best['actual_bets']} bets")

    if "ev_analysis" in result:
        print(f"\nEV Analysis (predicted_prob × payout - 100):")
        _print_ev(result["ev_analysis"])

    if "calibration" in result:
        print(f"\nCalibration (predicted prob vs actual win rate):")
        _print_calibration(result["calibration"])

    return result


def run_wfcv(args, X, y, meta) -> dict:
    """Walk-Forward Cross-Validation evaluation."""
    folds = walk_forward_splits(
        X, y, meta,
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
        n_train = len(fold["train"]["X"])
        n_test = len(fold["test"]["X"])
        print(f"  Fold {i+1}: train={n_train}R test={p['test']} ({n_test}R)")

    fold_results = []
    for i, fold in enumerate(folds):
        print(f"\n--- Fold {i+1}/{len(folds)} ({fold['period']['test']}) ---")

        model, metrics = train_boat1_model(
            fold["train"]["X"],
            fold["train"]["y"],
            fold["val"]["X"],
            fold["val"]["y"],
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            extra_params=json.loads(args.params) if args.params else None,
        )

        result = evaluate_boat1(
            model,
            fold["test"]["X"],
            fold["test"]["y"],
            fold["test"]["meta"],
            db_path=DB_PATH,
        )
        print(f"  AUC={result['auc']:.4f} Acc={result['accuracy']:.1%} "
              f"Base={result['base_hit_rate']:.1%}")

        if "thresholds" in result:
            best = find_best_threshold(result["thresholds"], min_bets=100)
            if best:
                print(f"  Best: t={best['threshold']:.2f} ROI={best['roi']:.1%} "
                      f"hit={best['hit_rate']:.1%} bets={best['actual_bets']}")

        fold_results.append(result)

    # Aggregate
    print("\n" + "=" * 60)
    print("Walk-Forward CV Summary")
    print("=" * 60)

    aucs = [r["auc"] for r in fold_results]
    print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # ROI at fixed thresholds across folds
    print(f"\nROI by threshold (across folds):")
    for threshold in [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]:
        rois = []
        bets = []
        for r in fold_results:
            if "thresholds" not in r:
                continue
            matching = [t for t in r["thresholds"] if abs(t["threshold"] - threshold) < 0.005]
            if matching:
                rois.append(matching[0]["roi"])
                bets.append(matching[0]["actual_bets"])
        if rois:
            print(f"  t={threshold:.2f}: ROI={np.mean(rois):.1%} ± {np.std(rois):.1%} "
                  f"bets={[int(b) for b in bets]}")

    # EV analysis across folds
    ev_thresholds_to_check = [-20, -10, 0, 10, 20, 30, 50]
    has_ev = all("ev_analysis" in r and r["ev_analysis"] for r in fold_results)
    if has_ev:
        print(f"\nEV-based ROI (across folds):")
        print(f"  {'EV≥':>5s}  {'MeanROI':>8s}  {'StdROI':>8s}  {'Folds ROI':>40s}  {'Bets':>20s}")
        print(f"  {'-' * 90}")
        for ev_thr in ev_thresholds_to_check:
            rois = []
            bets = []
            for r in fold_results:
                matching = [e for e in r["ev_analysis"] if e["ev_threshold"] == ev_thr]
                if matching:
                    rois.append(matching[0]["roi"])
                    bets.append(matching[0]["actual_bets"])
            if len(rois) == len(fold_results):
                rois_str = " ".join(f"{r:.0%}" for r in rois)
                bets_str = " ".join(str(int(b)) for b in bets)
                print(f"  {ev_thr:>+5d}  {np.mean(rois):>7.1%}  {np.std(rois):>7.1%}  "
                      f"{rois_str:>40s}  {bets_str:>20s}")

    return {"folds": fold_results}


def run_save(args, X, y, meta, df) -> None:
    """Train on all available data and save model for production use."""
    dates = sorted(df["race_date"].unique())
    # Find date ~2 months before end
    end_date = dates[-1]
    val_start_idx = max(0, len(dates) - 60)  # ~2 months of race-days
    val_start = dates[val_start_idx]

    b1_dates = meta["race_date"].values
    val_mask = b1_dates >= val_start
    train_mask = ~val_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Training for production save...")
    print(f"  Train: {len(X_train)}R, Val: {len(X_val)}R")

    model, metrics = train_boat1_model(
        X_train, y_train, X_val, y_val,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        extra_params=json.loads(args.params) if args.params else None,
    )

    if "val_auc" in metrics:
        print(f"  Val AUC: {metrics['val_auc']:.4f}")

    # Compute feature means for NaN fallback at prediction time
    feature_means = {c: float(X[c].astype("float64").mean()) for c in BOAT1_FEATURE_COLS}

    model_dir = args.model_dir
    save_boat1_model(model, model_dir)
    save_model_meta(
        model_dir,
        feature_columns=BOAT1_FEATURE_COLS,
        hyperparameters={"n_estimators": args.n_estimators, "learning_rate": args.learning_rate},
        training={"n_train": len(X_train), "n_val": len(X_val),
                  "date_range": f"{dates[0]} ~ {end_date}",
                  "val_auc": metrics.get("val_auc")},
        feature_means=feature_means,
    )
    print(f"\nModel saved to {model_dir}/")


def _suggest_lgb_params(trial):
    """Suggest LightGBM hyperparameters for Optuna trial."""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
    return params, n_estimators, learning_rate


def run_optuna(args, X, y, meta) -> None:
    """Hyperparameter optimization with Optuna."""
    import optuna

    folds = walk_forward_splits(
        X, y, meta,
        n_folds=args.n_folds,
        fold_months=args.fold_months,
        train_start=args.start_date,
    )
    if not folds:
        print("ERROR: No valid folds generated")
        sys.exit(1)

    print(f"Folds: {len(folds)}")
    print(f"Objective: {args.objective}")

    # Pre-load payouts (only needed for ev_roi)
    fold_payouts = []
    if args.objective == "ev_roi":
        print("Pre-loading payouts...")
        for fold in folds:
            test_race_ids = fold["test"]["meta"]["race_id"].unique()
            payouts = load_payouts(DB_PATH, test_race_ids)
            fold_payouts.append(payouts)

    def objective_ev_roi(trial: optuna.Trial) -> float:
        params, n_estimators, learning_rate = _suggest_lgb_params(trial)

        ev_rois = []
        for fold, payouts in zip(folds, fold_payouts):
            model, _ = train_boat1_model(
                fold["train"]["X"], fold["train"]["y"],
                fold["val"]["X"], fold["val"]["y"],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                extra_params=params,
            )
            result = evaluate_boat1(
                model, fold["test"]["X"], fold["test"]["y"], fold["test"]["meta"],
                payouts_cache=payouts,
            )
            # Use EV≥0 ROI as objective (the actual deployment strategy)
            ev_results = result.get("ev_analysis", [])
            ev0 = [e for e in ev_results if e["ev_threshold"] == 0]
            ev_rois.append(ev0[0]["roi"] if ev0 else 0.0)

        mean_roi = float(np.mean(ev_rois))
        trial.set_user_attr("roi_std", float(np.std(ev_rois)))
        trial.set_user_attr("rois", [round(r, 4) for r in ev_rois])
        return mean_roi

    def objective_upset_fbeta(trial: optuna.Trial) -> float:
        params, n_estimators, learning_rate = _suggest_lgb_params(trial)
        threshold = trial.suggest_float("upset_threshold", 0.30, 0.55)

        beta = args.beta
        beta_sq = beta ** 2
        fold_fbetas = []
        fold_recalls = []
        fold_precisions = []
        fold_n_bets = []

        for fold in folds:
            model, _ = train_boat1_model(
                fold["train"]["X"], fold["train"]["y"],
                fold["val"]["X"], fold["val"]["y"],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                extra_params=params,
            )

            y_prob = model.predict_proba(fold["test"]["X"])[:, 1]
            y_true = fold["test"]["y"].values

            # Upset detection: predict "boat 1 loses" when prob < threshold
            pred_upset = y_prob < threshold
            actual_upset = y_true == 0

            tp = int((pred_upset & actual_upset).sum())
            fp = int((pred_upset & ~actual_upset).sum())
            fn = int((~pred_upset & actual_upset).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
            else:
                fbeta = 0.0

            fold_fbetas.append(fbeta)
            fold_recalls.append(recall)
            fold_precisions.append(precision)
            fold_n_bets.append(tp + fp)

        mean_fbeta = float(np.mean(fold_fbetas))
        trial.set_user_attr("upset_recall", round(float(np.mean(fold_recalls)), 4))
        trial.set_user_attr("upset_precision", round(float(np.mean(fold_precisions)), 4))
        trial.set_user_attr("n_bets_per_fold", fold_n_bets)
        trial.set_user_attr("fbeta_per_fold", [round(f, 4) for f in fold_fbetas])
        trial.set_user_attr("threshold", round(threshold, 4))
        return mean_fbeta

    objective = objective_upset_fbeta if args.objective == "upset_fbeta" else objective_ev_roi

    study = optuna.create_study(
        direction="maximize",
        study_name=f"boat1-{args.objective}",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 70)
    print(f"Optuna Search Complete — {args.objective}")
    print("=" * 70)

    if args.objective == "upset_fbeta":
        best = study.best_trial
        print(f"Best F{args.beta}: {study.best_value:.4f}")
        print(f"  Recall:      {best.user_attrs['upset_recall']:.4f}")
        print(f"  Precision:   {best.user_attrs['upset_precision']:.4f}")
        print(f"  Threshold:   {best.user_attrs['threshold']:.4f}")
        print(f"  Bets/fold:   {best.user_attrs['n_bets_per_fold']}")
        print(f"  F-beta/fold: {best.user_attrs['fbeta_per_fold']}")
    else:
        print(f"Best ROI: {study.best_value:.1%}")
        print(f"  std={study.best_trial.user_attrs['roi_std']:.1%}")
        print(f"  per fold: {study.best_trial.user_attrs['rois']}")

    print(f"\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    print(f"\nTop 10 trials:")
    for t in trials[:10]:
        if t.value is None:
            continue
        if args.objective == "upset_fbeta":
            thr = t.user_attrs.get("threshold", 0)
            rec = t.user_attrs.get("upset_recall", 0)
            prec = t.user_attrs.get("upset_precision", 0)
            bets = t.user_attrs.get("n_bets_per_fold", [])
            print(f"  #{t.number}: F{args.beta}={t.value:.4f} "
                  f"recall={rec:.3f} prec={prec:.3f} thr={thr:.3f} bets={bets}")
        else:
            print(f"  #{t.number}: ROI={t.value:.1%} ±{t.user_attrs['roi_std']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Train boat 1 win prediction binary classifier"
    )
    parser.add_argument("--mode", choices=["single", "wfcv", "optuna"], default="single")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", default=None, help="Extra LightGBM params as JSON")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trial count")
    parser.add_argument(
        "--objective", choices=["ev_roi", "upset_fbeta"], default="ev_roi",
        help="Optuna objective: ev_roi (tansho ROI) or upset_fbeta (upset detection F-beta)",
    )
    parser.add_argument("--beta", type=float, default=1.5, help="F-beta for upset_fbeta (default: 1.5)")
    parser.add_argument("--save", action="store_true", help="Save model to --model-dir")
    parser.add_argument("--model-dir", default="models/boat1", help="Model output dir")
    args = parser.parse_args()

    print("Boat 1 Binary Classifier")
    print(f"Mode: {args.mode}")
    if args.mode == "optuna":
        print(f"Objective: {args.objective}")
        if args.objective == "upset_fbeta":
            print(f"Beta: {args.beta}")
    else:
        print(f"Trees: {args.n_estimators}, LR: {args.learning_rate}")
    if args.start_date:
        print(f"Train start: {args.start_date}")
    print()

    t0 = time.time()
    print("Building features...")
    df = build_features_df(DB_PATH, start_date=args.start_date)
    print(f"  Raw: {len(df)} entries ({len(df)//6} races)")

    print("Reshaping to boat 1 format...")
    X, y, meta = reshape_to_boat1(df)
    print(f"  Boat1: {len(X)} races, {len(BOAT1_FEATURE_COLS)} features")
    print(f"  Boat 1 win rate: {y.mean():.1%}")
    print()

    if args.save:
        run_save(args, X, y, meta, df)
    elif args.mode == "single":
        run_single(args, X, y, meta)
    elif args.mode == "wfcv":
        run_wfcv(args, X, y, meta)
    elif args.mode == "optuna":
        run_optuna(args, X, y, meta)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
