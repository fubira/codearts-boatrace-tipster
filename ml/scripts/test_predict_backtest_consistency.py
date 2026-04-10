"""Consistency test: predict path (runner) vs backtest path must produce same predictions.

Catches bugs where NaN handling, feature column filtering, or data loading
differs between the two paths. The NaN fill mismatch found on 2026-04-10
(b1_prob 65% vs 13%) is the type of bug this test prevents.

Usage:
    uv run --directory ml python -m scripts.test_predict_backtest_consistency
"""

import contextlib
import io
import sys

import numpy as np

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import load_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta

MODEL_DIR = "models/trifecta_v1"
# Use a recent date with known data
TEST_DATE = "2026-04-09"
TOLERANCE = 1e-4  # Allow float precision diff (typical: ~5e-5)


def main():
    print(f"Consistency test: predict vs backtest for {TEST_DATE}")
    print(f"Model: {MODEL_DIR}")
    print()

    # === Path A: Backtest path (build_features_df → reshape → predict) ===
    print("Path A (backtest): build_features_df...", end="", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(DEFAULT_DB_PATH)
    next_day = "2026-04-10"
    df_a = df_full[
        (df_full["race_date"] >= TEST_DATE) & (df_full["race_date"] < next_day)
    ].reset_index(drop=True)
    print(f" {len(df_a) // 6} races")

    b1_model = load_boat1_model(f"{MODEL_DIR}/boat1")
    b1_meta = load_model_meta(f"{MODEL_DIR}/boat1")
    rank_model = load_model(f"{MODEL_DIR}/ranking")
    rank_meta = load_model_meta(f"{MODEL_DIR}/ranking")

    with contextlib.redirect_stdout(io.StringIO()):
        X_b1_a, _, meta_b1_a = reshape_to_boat1(df_a)
    fill_nan_with_means(X_b1_a, b1_meta)
    b1_probs_a = b1_model.predict_proba(X_b1_a)[:, 1]

    X_rank_a, _, meta_rank_a = prepare_feature_matrix(df_a)
    fill_nan_with_means(X_rank_a, rank_meta)
    rank_scores_a = rank_model.predict(X_rank_a)

    # === Path B: Predict path (predict_trifecta without snapshot) ===
    print("Path B (predict): predict_trifecta...", end="", flush=True)
    # Import relative to the scripts package
    import importlib
    mod = importlib.import_module("predict_trifecta")
    predict_trifecta = mod.predict_trifecta

    result = predict_trifecta(
        date=TEST_DATE,
        model_dir=MODEL_DIR,
        db_path=DEFAULT_DB_PATH,
        b1_threshold=0.99,  # Accept all to see all b1_probs
        ev_threshold=-999,  # Accept all
        use_snapshots=False,
    )
    print(f" {result.get('n_races', 0)} races")

    # Build b1_prob map from predict result
    b1_map_b = {}
    for p in result.get("predictions", []):
        b1_map_b[p["race_id"]] = p["b1_prob"]
    for rid_str, info in result.get("skipped", {}).items():
        b1_map_b[int(rid_str)] = info["b1_prob"]

    # === Compare ===
    race_ids_a = meta_b1_a["race_id"].values.astype(int)
    n_compared = 0
    n_b1_mismatch = 0
    max_b1_diff = 0.0
    mismatches = []

    for i, rid in enumerate(race_ids_a):
        rid = int(rid)
        if rid not in b1_map_b:
            continue
        b1_a = b1_probs_a[i]
        b1_b = b1_map_b[rid]
        diff = abs(b1_a - b1_b)
        n_compared += 1
        if diff > max_b1_diff:
            max_b1_diff = diff
        if diff > TOLERANCE:
            n_b1_mismatch += 1
            if diff > 0.01:  # Only report significant mismatches
                mismatches.append((rid, b1_a, b1_b, diff))

    print()
    print(f"=== Results ===")
    print(f"Compared: {n_compared} races")
    print(f"b1_prob max diff: {max_b1_diff:.8f}")
    print(f"b1_prob mismatches (>{TOLERANCE}): {n_b1_mismatch}")

    if mismatches:
        print(f"\nSignificant mismatches (diff > 1%):")
        for rid, a, b, d in sorted(mismatches, key=lambda x: -x[3])[:10]:
            print(f"  race {rid}: backtest={a:.4f} predict={b:.4f} diff={d:.4f}")

    # Ranking scores comparison
    rank_rids_a = meta_rank_a["race_id"].values.astype(int)
    # predict_trifecta doesn't expose raw rank scores, so we verify
    # that the winner_pick is consistent
    pick_map_b = {}
    for p in result.get("predictions", []):
        pick_map_b[p["race_id"]] = p["winner_pick"]

    n_pick_compared = 0
    n_pick_mismatch = 0
    from scipy.special import softmax

    FIELD = 6
    n_races_rank = len(rank_scores_a) // FIELD
    for r in range(n_races_rank):
        idx = r * FIELD
        rid = int(rank_rids_a[idx])
        if rid not in pick_map_b:
            continue
        scores = rank_scores_a[idx : idx + FIELD]
        boats = meta_rank_a["boat_number"].values[idx : idx + FIELD].astype(int)
        best_j = max(
            (j for j in range(FIELD) if boats[j] != 1),
            key=lambda j: scores[j],
        )
        pick_a = int(boats[best_j])
        pick_b = pick_map_b[rid]
        n_pick_compared += 1
        if pick_a != pick_b:
            n_pick_mismatch += 1

    print(f"\nwinner_pick compared: {n_pick_compared}")
    print(f"winner_pick mismatches: {n_pick_mismatch}")

    # Final verdict
    print()
    if n_b1_mismatch == 0 and n_pick_mismatch == 0:
        print("PASS: predict and backtest paths are consistent")
    else:
        print("FAIL: paths diverge — investigate mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()
