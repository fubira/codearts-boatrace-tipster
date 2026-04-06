"""Grid search for rolling feature window sizes.

Tests different window sizes and feature combinations.
Builds features once per window size, then tests subsets in-memory.

Usage:
    PYTHONUNBUFFERED=1 uv run --directory ml python -m scripts.grid_rolling
"""

import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

import boatrace_tipster_ml.features as feat
import boatrace_tipster_ml.feature_config as fc
from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.evaluate import evaluate_model, load_payouts
from boatrace_tipster_ml.model import train_model, walk_forward_splits

DB_PATH = DEFAULT_DB_PATH

# Baseline 24 features
BASE_COLS = [
    "boat_number", "racer_weight",
    "national_win_rate", "national_top2_rate", "national_top3_rate",
    "local_win_rate", "local_top3_rate",
    "motor_top3_rate", "exhibition_time",
    "racer_course_win_rate", "racer_course_top2_rate",
    "recent_avg_position",
    "rel_national_win_rate", "rel_exhibition_time",
    "stadium_course_win_rate", "course_number", "rel_exhibition_st",
    "kado_x_exhibition",
    "tourn_exhibition_delta", "tourn_st_delta", "tourn_avg_position",
    "class_x_boat", "weight_x_boat", "wind_speed_x_boat",
]

# All rolling-derived features (superset for build)
ALL_ROLLING_COLS = [
    "rel_rolling_st", "rel_rolling_win_rate", "rel_rolling_avg_pos",
    "rel_rolling_course_win", "rel_rolling_course_st",
    "st_form_delta", "pos_form_delta",
]

# Feature groups to test (added on top of BASE_COLS)
FEATURE_GROUPS = {
    "st": ["rel_rolling_st"],
    "pos": ["rel_rolling_avg_pos"],
    "win": ["rel_rolling_win_rate"],
    "course_win": ["rel_rolling_course_win"],
    "course_st": ["rel_rolling_course_st"],
    "form_delta": ["st_form_delta", "pos_form_delta"],
    "st+course": ["rel_rolling_st", "rel_rolling_course_win"],
    "st+pos": ["rel_rolling_st", "rel_rolling_avg_pos"],
    "all_rel": ["rel_rolling_st", "rel_rolling_win_rate", "rel_rolling_avg_pos"],
}

WINDOWS = [5, 10, 15, 20, 30]


def build_full(window: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build features with all rolling columns included."""
    feat.ROLLING_WINDOW = window
    # Temporarily expand FEATURE_COLS to include all rolling features
    saved = fc.FEATURE_COLS[:]
    fc.FEATURE_COLS = BASE_COLS + ALL_ROLLING_COLS
    try:
        X, y, meta = feat.build_features(DB_PATH)
    finally:
        fc.FEATURE_COLS = saved
    return X, y, meta


def run_wfcv_subset(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    feature_cols: list[str],
    payouts_per_fold: list[dict],
    folds_indices: list[dict],
) -> dict:
    """Run WF-CV on a feature subset using pre-split indices."""
    rois_2r = []
    rois_tan = []
    ndcgs = []
    top1s = []
    hits_2r = []

    X_sub = X[feature_cols]

    for fidx, payouts in zip(folds_indices, payouts_per_fold):
        train_X = X_sub.iloc[fidx["train"]]
        train_y = y.iloc[fidx["train"]]
        train_meta = meta.iloc[fidx["train"]]
        val_X = X_sub.iloc[fidx["val"]]
        val_y = y.iloc[fidx["val"]]
        val_meta = meta.iloc[fidx["val"]]
        test_X = X_sub.iloc[fidx["test"]]
        test_y = y.iloc[fidx["test"]]
        test_meta = meta.iloc[fidx["test"]]

        model, _ = train_model(
            train_X, train_y, train_meta,
            val_X, val_y, val_meta,
            relevance_scheme="top_heavy",
        )
        result = evaluate_model(
            model, test_X, test_y, test_meta,
            payouts_cache=payouts, skip_confidence=True,
        )
        ndcgs.append(result["avgNDCG"])
        top1s.append(result["topNAccuracy"]["1"])
        hits_2r.append(result["multiHitRates"]["2連単"])
        rois_2r.append(
            result.get("payoutROI", {}).get("2連単", {}).get("recoveryRate", 0)
        )
        rois_tan.append(
            result.get("payoutROI", {}).get("単勝", {}).get("recoveryRate", 0)
        )

    return {
        "nDCG": np.mean(ndcgs),
        "top1": np.mean(top1s),
        "hit_2r": np.mean(hits_2r),
        "roi_2r_mean": np.mean(rois_2r),
        "roi_2r_std": np.std(rois_2r),
        "roi_tan_mean": np.mean(rois_tan),
        "roi_tan_std": np.std(rois_tan),
        "roi_2r_folds": [round(r, 4) for r in rois_2r],
    }


def main():
    t0 = time.time()
    results = []

    for window in WINDOWS:
        print(f"\n{'=' * 80}")
        print(f"Building features with window={window} race-days")
        print(f"{'=' * 80}")

        tw = time.time()
        X, y, meta = build_full(window)
        print(f"  Built in {time.time() - tw:.1f}s ({X.shape[1]} features)")

        # Create WF-CV splits (index-based, reusable across feature subsets)
        folds = walk_forward_splits(X, y, meta, n_folds=4, fold_months=2)
        folds_indices = []
        payouts_per_fold = []
        for fold in folds:
            fidx = {
                "train": fold["train"]["X"].index.tolist(),
                "val": fold["val"]["X"].index.tolist(),
                "test": fold["test"]["X"].index.tolist(),
            }
            folds_indices.append(fidx)
            test_ids = fold["test"]["meta"]["race_id"].unique()
            payouts_per_fold.append(load_payouts(DB_PATH, test_ids))

        # Baseline (only for first window, it's window-independent)
        if window == WINDOWS[0]:
            print(f"\n  baseline_24:")
            r = run_wfcv_subset(X, y, meta, BASE_COLS, payouts_per_fold, folds_indices)
            r["label"] = "baseline_24"
            results.append(r)
            _print_result(r)

        # Test each feature group
        for group_name, group_cols in FEATURE_GROUPS.items():
            cols = BASE_COLS + group_cols
            # Verify all columns exist
            missing = [c for c in cols if c not in X.columns]
            if missing:
                print(f"\n  w{window}_{group_name}: SKIP (missing {missing})")
                continue

            label = f"w{window}_{group_name}"
            print(f"\n  {label}:")
            r = run_wfcv_subset(X, y, meta, cols, payouts_per_fold, folds_indices)
            r["label"] = label
            results.append(r)
            _print_result(r)

    # Summary table
    print(f"\n\n{'=' * 110}")
    print("SUMMARY (sorted by 2連単 ROI)")
    print(f"{'=' * 110}")
    print(f"  {'Config':<25s} {'nDCG':>6s} {'Top1':>5s} {'2連単hit':>7s} "
          f"{'2連単ROI':>8s} {'±':>5s} {'単勝ROI':>7s} {'±':>5s} {'folds':>30s}")
    print("  " + "-" * 105)

    sorted_results = sorted(results, key=lambda r: -r["roi_2r_mean"])
    for r in sorted_results:
        folds_str = " ".join(f"{v:.1%}" for v in r["roi_2r_folds"])
        print(f"  {r['label']:<25s} {r['nDCG']:>6.4f} {r['top1']:>4.1%} {r['hit_2r']:>6.1%} "
              f"{r['roi_2r_mean']:>7.1%} {r['roi_2r_std']:>4.1%} "
              f"{r['roi_tan_mean']:>6.1%} {r['roi_tan_std']:>4.1%}  "
              f"[{folds_str}]")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


def _print_result(r: dict) -> None:
    print(f"    2連単ROI={r['roi_2r_mean']:.1%}±{r['roi_2r_std']:.1%} "
          f"nDCG={r['nDCG']:.4f} Top1={r['top1']:.1%} "
          f"hit={r['hit_2r']:.1%} 単勝={r['roi_tan_mean']:.1%}")


if __name__ == "__main__":
    main()
