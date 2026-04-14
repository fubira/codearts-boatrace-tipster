"""Test whether p2_v2 HP produces stable predictions when train data
shifts by 2 days. Diagnoses whether the WF-CV instability comes from:

  (1) fundamental model instability (predictions diverge widely)
  (2) filter edge brittleness (predictions similar but buy decisions flip)
  (3) early stopping pathology (best_iter differs, full-iter models agree)

Trains 4 versions of p2_v2 #294 HP on fold-1 train data:
  A_orig  : cutoff 2025-07-11, early_stopping_rounds=200
  A_today : cutoff 2025-07-13, early_stopping_rounds=200
  B_orig  : cutoff 2025-07-11, early_stopping=None, n_est=1333 (production)
  B_today : cutoff 2025-07-13, early_stopping=None, n_est=1333

Predicts on the OVERLAPPING test window (2025-08-13 to 2025-10-11, 59 days)
and compares per-race predictions across the 4 models.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model
from scripts.tune_p2 import _load_trifecta_odds

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
N_EST_UPPER = 1333
LR = 0.006172936736632081
RELEVANCE = "podium"
GAP23 = 0.13
EV = 0.0
CONC_TH = 0.448  # original tune's conc for #294
GAP12_TH = 0.0   # original tune had no gap12

OVERLAP_START = "2025-08-13"
OVERLAP_END = "2025-10-11"


def _train(df, train_cutoff, val_start, val_end, *, early_stop):
    """Train p2_v2 HP on data < val_end, with last month as val."""
    train_df = df[df["race_date"] < val_end]
    val_mask = train_df["race_date"] >= val_start
    X = train_df[FEATURES].copy()
    y = train_df["finish_position"]
    meta = train_df[["race_id", "racer_id", "race_date", "boat_number"]].copy()
    n_train = int((~val_mask).sum() // FIELD_SIZE)
    n_val = int(val_mask.sum() // FIELD_SIZE)
    print(f"  train_R={n_train} val_R={n_val} val={val_start}..{val_end}",
          file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            X[~val_mask], y[~val_mask], meta[~val_mask],
            X[val_mask], y[val_mask], meta[val_mask],
            n_estimators=N_EST_UPPER, learning_rate=LR,
            relevance_scheme=RELEVANCE, extra_params=dict(HP),
            early_stopping_rounds=200 if early_stop else None,
        )
    best_it = getattr(model, "best_iteration_", None) or N_EST_UPPER
    return model, best_it


def _predict_features(model, df_overlap):
    """Return per-race {race_id, p1_prob, p2_prob, p3_prob, gap12, gap23,
    top3_conc, r1_boat, r2_boat, r3_boat}."""
    n_races = len(df_overlap) // FIELD_SIZE
    X = df_overlap[FEATURES]
    scores = model.predict(X)
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = df_overlap["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = df_overlap["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    rows = []
    for i in range(n_races):
        p1 = probs[i, pred_order[i, 0]]
        p2 = probs[i, pred_order[i, 1]]
        p3 = probs[i, pred_order[i, 2]]
        rows.append({
            "race_id": int(rids[i]),
            "r1": int(top_boats[i, 0]),
            "r2": int(top_boats[i, 1]),
            "r3": int(top_boats[i, 2]),
            "p1": float(p1),
            "p2": float(p2),
            "p3": float(p3),
            "gap12": float(p1 - p2),
            "gap23": float(p2 - p3),
            "conc": float((p2 + p3) / (1 - p1 + 1e-10)),
        })
    return pd.DataFrame(rows).set_index("race_id")


def _passes_filter(row):
    """P2 filter: top1=boat1 AND gap12>=th AND conc>=th AND gap23>=th."""
    if row["r1"] != 1:
        return False
    if row["gap12"] < GAP12_TH:
        return False
    if row["conc"] < CONC_TH:
        return False
    if row["gap23"] < GAP23:
        return False
    return True


def main():
    print("Loading features...", file=sys.stderr)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(DEFAULT_DB_PATH)
    print(f"  {time.time()-t0:.1f}s, rows={len(df)}", file=sys.stderr)

    overlap = df[(df["race_date"] >= OVERLAP_START) & (df["race_date"] <= OVERLAP_END)]
    overlap = overlap.sort_values(["race_id", "boat_number"]).reset_index(drop=True)
    n_overlap = len(overlap) // FIELD_SIZE
    print(f"Overlap test window {OVERLAP_START}..{OVERLAP_END}: {n_overlap} races",
          file=sys.stderr)

    # Train 4 models
    models = {}
    print("\n[A_orig] cutoff=2025-07-11 early_stop=200", file=sys.stderr)
    t1 = time.time()
    models["A_orig"] = _train(df, "2025-07-11", "2025-07-11", "2025-08-11", early_stop=True)
    print(f"  best_it={models['A_orig'][1]} ({time.time()-t1:.0f}s)", file=sys.stderr)

    print("\n[A_today] cutoff=2025-07-13 early_stop=200", file=sys.stderr)
    t1 = time.time()
    models["A_today"] = _train(df, "2025-07-13", "2025-07-13", "2025-08-13", early_stop=True)
    print(f"  best_it={models['A_today'][1]} ({time.time()-t1:.0f}s)", file=sys.stderr)

    print("\n[B_orig] cutoff=2025-07-11 early_stop=None n_est=1333", file=sys.stderr)
    t1 = time.time()
    models["B_orig"] = _train(df, "2025-07-11", "2025-07-11", "2025-08-11", early_stop=False)
    print(f"  iters=1333 ({time.time()-t1:.0f}s)", file=sys.stderr)

    print("\n[B_today] cutoff=2025-07-13 early_stop=None n_est=1333", file=sys.stderr)
    t1 = time.time()
    models["B_today"] = _train(df, "2025-07-13", "2025-07-13", "2025-08-13", early_stop=False)
    print(f"  iters=1333 ({time.time()-t1:.0f}s)", file=sys.stderr)

    # Predict on overlap window with all 4 models
    preds = {name: _predict_features(model, overlap) for name, (model, _) in models.items()}

    print()
    print("=" * 90)
    print("Per-race prediction comparison (overlap test window only)")
    print("=" * 90)

    # Pairwise comparisons
    pairs = [
        ("A_orig", "A_today", "Early-stop, 2-day train shift"),
        ("B_orig", "B_today", "Full 1333 iter, 2-day train shift"),
        ("A_orig", "B_orig", "Original cutoff: early-stop vs full iter"),
        ("A_today", "B_today", "Today cutoff: early-stop vs full iter"),
    ]

    for a, b, label in pairs:
        pa = preds[a]
        pb = preds[b]
        common = pa.index.intersection(pb.index)
        pa = pa.loc[common]
        pb = pb.loc[common]

        # Per-race prob divergence
        p1_diff = (pa["p1"] - pb["p1"]).abs()
        gap23_diff = (pa["gap23"] - pb["gap23"]).abs()
        conc_diff = (pa["conc"] - pb["conc"]).abs()

        # Filter pass: same races bought?
        buy_a = pa.apply(_passes_filter, axis=1)
        buy_b = pb.apply(_passes_filter, axis=1)
        both = (buy_a & buy_b).sum()
        only_a = (buy_a & ~buy_b).sum()
        only_b = (~buy_a & buy_b).sum()
        neither = (~buy_a & ~buy_b).sum()

        # Top-1 boat agreement
        same_top1 = (pa["r1"] == pb["r1"]).sum()

        print(f"\n--- {a} vs {b} ({label}) ---")
        print(f"  races compared: {len(common)}")
        print(f"  same top-1 boat: {same_top1}/{len(common)} ({100*same_top1/len(common):.1f}%)")
        print(f"  |Δp1|       median={p1_diff.median():.4f} mean={p1_diff.mean():.4f} max={p1_diff.max():.4f}")
        print(f"  |Δgap23|    median={gap23_diff.median():.4f} mean={gap23_diff.mean():.4f} max={gap23_diff.max():.4f}")
        print(f"  |Δconc|     median={conc_diff.median():.4f} mean={conc_diff.mean():.4f} max={conc_diff.max():.4f}")
        print(f"  buy decisions:")
        print(f"    both buy:    {both}")
        print(f"    only {a}:    {only_a}")
        print(f"    only {b}:    {only_b}")
        print(f"    both skip:   {neither}")
        if (both + only_a + only_b) > 0:
            agreement = both / (both + only_a + only_b) * 100
            print(f"    Jaccard buy agreement: {agreement:.1f}%")


if __name__ == "__main__":
    main()
