"""Consistency test: all prediction paths must produce identical results.

Compares three paths that should give the same b1_prob and winner_pick:
  A) Backtest path: build_features_df → reshape → fill_nan → predict
  B) Predict path (full pipeline): predict_trifecta(snapshot_path=None)
  C) Predict path (snapshot): predict_trifecta(snapshot_path=latest)

The NaN fill mismatch (b1=65% vs 13%) and Int64 type error found on
2026-04-10/11 are the types of bugs this test catches.

Usage:
    cd ml && PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/test_predict_backtest_consistency.py
"""

import contextlib
import glob
import io
import os
import sys

import numpy as np

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import load_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta

MODEL_DIR = "models/trifecta_v1"
TOLERANCE = 1e-4  # Allow float precision diff (typical: ~5e-5)

# Find latest snapshot and derive test date
SNAPSHOT_DIR = os.path.join(os.path.dirname(DEFAULT_DB_PATH), "stats-snapshots")


def find_test_config() -> tuple[str, str | None]:
    """Find the best test date and snapshot path."""
    snaps = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.db")))
    # Filter to standard date-named snapshots (YYYY-MM-DD.db)
    snaps = [s for s in snaps if os.path.basename(s).count("-") == 2 and len(os.path.basename(s)) == 13]
    if snaps:
        snap_path = snaps[-1]
        # Snapshot date is the through-date; test date = next day
        snap_date = os.path.basename(snap_path).replace(".db", "")
        from datetime import datetime, timedelta
        test_date = (datetime.strptime(snap_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        return test_date, snap_path
    return "2026-04-09", None


def get_b1_and_picks(b1_probs, meta_b1, rank_scores, meta_rank, field=6):
    """Extract per-race b1_prob and winner_pick from raw model outputs."""
    from scipy.special import softmax
    result = {}
    race_ids = meta_b1["race_id"].values.astype(int)
    for i, rid in enumerate(race_ids):
        result[int(rid)] = {"b1_prob": b1_probs[i]}

    n_races = len(rank_scores) // field
    for r in range(n_races):
        idx = r * field
        rid = int(meta_rank["race_id"].values[idx])
        scores = rank_scores[idx:idx + field]
        boats = meta_rank["boat_number"].values[idx:idx + field].astype(int)
        best_j = max((j for j in range(field) if boats[j] != 1), key=lambda j: scores[j])
        if rid in result:
            result[rid]["winner_pick"] = int(boats[best_j])
    return result


def run_backtest_path(df_day):
    """Path A: backtest — same code as backtest_trifecta.py run_period."""
    b1_model = load_boat1_model(f"{MODEL_DIR}/boat1")
    b1_meta = load_model_meta(f"{MODEL_DIR}/boat1")
    rank_model = load_model(f"{MODEL_DIR}/ranking")
    rank_meta = load_model_meta(f"{MODEL_DIR}/ranking")

    with contextlib.redirect_stdout(io.StringIO()):
        X_b1, _, meta_b1 = reshape_to_boat1(df_day)
    fill_nan_with_means(X_b1, b1_meta)
    b1_probs = b1_model.predict_proba(X_b1)[:, 1]

    X_rank, _, meta_rank = prepare_feature_matrix(df_day)
    fill_nan_with_means(X_rank, rank_meta)
    rank_scores = rank_model.predict(X_rank)

    return get_b1_and_picks(b1_probs, meta_b1, rank_scores, meta_rank)


def run_predict_path(test_date, snapshot_path=None):
    """Path B/C: predict_trifecta (full pipeline or snapshot)."""
    import importlib
    mod = importlib.import_module("predict_trifecta")

    result = mod.predict_trifecta(
        date=test_date,
        model_dir=MODEL_DIR,
        db_path=DEFAULT_DB_PATH,
        b1_threshold=0.99,
        ev_threshold=-999,
        snapshot_path=snapshot_path,
        use_snapshots=False,
    )

    b1_map = {}
    for p in result.get("predictions", []):
        b1_map[p["race_id"]] = {"b1_prob": p["b1_prob"], "winner_pick": p["winner_pick"]}
    for rid_str, info in result.get("skipped", {}).items():
        entry = {"b1_prob": info["b1_prob"]}
        if info.get("pick"):
            entry["winner_pick"] = info["pick"]
        b1_map[int(rid_str)] = entry

    return b1_map, result.get("n_races", 0)


def compare(name_a, data_a, name_b, data_b):
    """Compare two prediction maps and report differences."""
    common = set(data_a.keys()) & set(data_b.keys())
    if not common:
        print(f"  WARNING: no common races between {name_a} and {name_b}")
        return False

    b1_diffs = []
    pick_mismatches = []
    for rid in common:
        a, b = data_a[rid], data_b[rid]
        diff = abs(a["b1_prob"] - b["b1_prob"])
        if diff > TOLERANCE:
            b1_diffs.append((rid, a["b1_prob"], b["b1_prob"], diff))
        pa = a.get("winner_pick")
        pb = b.get("winner_pick")
        if pa and pb and pa != pb:
            pick_mismatches.append((rid, pa, pb))

    ok = True
    print(f"  {name_a} vs {name_b}: {len(common)} races")

    if b1_diffs:
        print(f"    b1_prob mismatches: {len(b1_diffs)}")
        for rid, a, b, d in sorted(b1_diffs, key=lambda x: -x[3])[:5]:
            print(f"      race {rid}: {name_a}={a:.4f} {name_b}={b:.4f} diff={d:.4f}")
        ok = False
    else:
        max_diff = max(
            (abs(data_a[r]["b1_prob"] - data_b[r]["b1_prob"]) for r in common),
            default=0,
        )
        print(f"    b1_prob OK (max diff: {max_diff:.8f})")

    if pick_mismatches:
        print(f"    winner_pick mismatches: {len(pick_mismatches)}")
        ok = False
    else:
        print(f"    winner_pick OK")

    return ok


def main():
    test_date, snapshot_path = find_test_config()
    from datetime import datetime, timedelta
    next_day = (datetime.strptime(test_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Test date: {test_date}")
    print(f"Snapshot:  {snapshot_path or 'none'}")
    print(f"Model:     {MODEL_DIR}")
    print()

    # Load full features once
    print("Loading features...", end="", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(DEFAULT_DB_PATH)
    df_day = df_full[
        (df_full["race_date"] >= test_date) & (df_full["race_date"] < next_day)
    ].reset_index(drop=True)
    n_races = len(df_day) // 6
    print(f" {n_races} races")

    if n_races == 0:
        print("ERROR: no races for test date")
        sys.exit(1)

    # Path A: backtest
    print("Path A (backtest)...", end="", flush=True)
    data_a = run_backtest_path(df_day)
    print(f" {len(data_a)} races")

    # Path B: predict (full pipeline)
    print("Path B (predict full)...", end="", flush=True)
    data_b, n_b = run_predict_path(test_date)
    print(f" {n_b} races")

    # Path C: predict (snapshot) — only if snapshot available
    data_c = None
    if snapshot_path:
        print("Path C (predict snapshot)...", end="", flush=True)
        data_c, n_c = run_predict_path(test_date, snapshot_path=snapshot_path)
        print(f" {n_c} races")

    # Compare all pairs
    print()
    print("=== Comparisons ===")
    all_ok = True

    # A vs B must match exactly (same data source, same code path)
    all_ok &= compare("backtest", data_a, "predict_full", data_b)

    if data_c is not None:
        # A vs C: snapshot uses previous-day stats, so tourn_ features differ.
        # This is by design. We check for catastrophic divergence only.
        SNAP_TOLERANCE = 0.15  # snapshot may differ due to tourn_ features
        print()
        common = set(data_a.keys()) & set(data_c.keys())
        big_diffs = []
        for rid in common:
            diff = abs(data_a[rid]["b1_prob"] - data_c[rid]["b1_prob"])
            if diff > SNAP_TOLERANCE:
                big_diffs.append((rid, data_a[rid]["b1_prob"], data_c[rid]["b1_prob"], diff))

        if big_diffs:
            print(f"  backtest vs predict_snap: {len(big_diffs)} races with diff > {SNAP_TOLERANCE:.0%}")
            for rid, a, c, d in sorted(big_diffs, key=lambda x: -x[3])[:5]:
                print(f"    race {rid}: backtest={a:.4f} snap={c:.4f} diff={d:.4f}")
            all_ok = False
        else:
            max_diff = max(
                (abs(data_a[r]["b1_prob"] - data_c[r]["b1_prob"]) for r in common),
                default=0,
            )
            print(f"  backtest vs predict_snap: OK (max diff: {max_diff:.4f}, within {SNAP_TOLERANCE:.0%} tolerance)")

    print()
    if all_ok:
        paths = "A/B/C" if data_c else "A/B"
        print(f"PASS: all paths ({paths}) are consistent")
    else:
        print("FAIL: paths diverge — investigate mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()
