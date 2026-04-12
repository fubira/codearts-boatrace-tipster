"""Compare full pipeline vs snapshot pipeline for P2 strategy.

Builds features both ways for a given date, runs the ranking model, and reports
per-race differences in rank_score, top3_conc, gap23 — exposing divergence
between runner (snapshot path) and offline analysis (full path).

Usage:
    cd ml && PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/compare_full_vs_snapshot.py \
        --date 2026-04-12 --snapshot ../data/stats-snapshots/2026-04-11.db
"""

import argparse
import contextlib
import io
import sys
from datetime import datetime, timedelta

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
from boatrace_tipster_ml.registry import get_active_model_dir
from boatrace_tipster_ml.snapshot_features import build_features_from_snapshot

FIELD_SIZE = 6
MODEL_DIR = f"{get_active_model_dir()}/ranking"


def score_df(df, model, meta):
    feature_cols = meta["feature_columns"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ERROR missing features: {missing}", file=sys.stderr)
        sys.exit(1)
    X = df[feature_cols].copy()
    fill_nan_with_means(X, meta)
    scores = model.predict(X)

    n = len(X) // FIELD_SIZE
    scores2d = scores.reshape(n, FIELD_SIZE)
    boats2d = df["boat_number"].values.reshape(n, FIELD_SIZE)
    rids = df["race_id"].values.reshape(n, FIELD_SIZE)[:, 0]

    exp_s = np.exp(scores2d - scores2d.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    order = np.argsort(-scores2d, axis=1)

    out = {}
    for i in range(n):
        po = order[i]
        p1 = float(probs[i, po[0]])
        p2 = float(probs[i, po[1]])
        p3 = float(probs[i, po[2]])
        out[int(rids[i])] = {
            "top1_boat": int(boats2d[i, po[0]]),
            "rank2_boat": int(boats2d[i, po[1]]),
            "rank3_boat": int(boats2d[i, po[2]]),
            "p1": p1, "p2": p2, "p3": p3,
            "top3_conc": (p2 + p3) / (1 - p1 + 1e-10),
            "gap23": p2 - p3,
            "scores": scores2d[i].tolist(),
            "boats": boats2d[i].tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--focus-races", default="",
                    help="comma-separated race_ids to print details for")
    args = ap.parse_args()

    next_day = (datetime.strptime(args.date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    focus = {int(x) for x in args.focus_races.split(",") if x.strip()}

    print(f"Date:     {args.date}")
    print(f"Snapshot: {args.snapshot}")
    print(f"DB:       {args.db_path}")
    print()

    model = load_model(MODEL_DIR)
    meta = load_model_meta(MODEL_DIR)

    print("Building full pipeline...", end="", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path, start_date=args.date, end_date=next_day)
    print(f" {len(df_full) // FIELD_SIZE} races")

    print("Building snapshot pipeline...", end="", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df_snap = build_features_from_snapshot(args.db_path, args.snapshot, args.date)
    print(f" {len(df_snap) // FIELD_SIZE} races")
    print()

    full_out = score_df(df_full, model, meta)
    snap_out = score_df(df_snap, model, meta)

    common = sorted(set(full_out) & set(snap_out))
    only_full = set(full_out) - set(snap_out)
    only_snap = set(snap_out) - set(full_out)

    print(f"Common races: {len(common)}")
    if only_full:
        print(f"  Only in full: {sorted(only_full)[:10]}")
    if only_snap:
        print(f"  Only in snap: {sorted(only_snap)[:10]}")
    print()

    big_score_diff = []
    top1_mismatch = []
    rank23_mismatch = []
    conc_diffs = []
    gap_diffs = []

    for rid in common:
        a, b = full_out[rid], snap_out[rid]
        score_diff = max(abs(sa - sb) for sa, sb in zip(a["scores"], b["scores"]))
        if score_diff > 0.01:
            big_score_diff.append((rid, score_diff))
        if a["top1_boat"] != b["top1_boat"]:
            top1_mismatch.append(rid)
        if (a["rank2_boat"], a["rank3_boat"]) != (b["rank2_boat"], b["rank3_boat"]):
            rank23_mismatch.append(rid)
        conc_diffs.append(abs(a["top3_conc"] - b["top3_conc"]))
        gap_diffs.append(abs(a["gap23"] - b["gap23"]))

    print(f"Max score diff:    {max((d for _, d in big_score_diff), default=0):.4f}")
    print(f"Races score>0.01:  {len(big_score_diff)} / {len(common)}")
    print(f"Top1 mismatches:   {len(top1_mismatch)}")
    print(f"Rank2/3 mismatch:  {len(rank23_mismatch)}")
    print(f"Mean conc diff:    {np.mean(conc_diffs):.4f}  max: {max(conc_diffs):.4f}")
    print(f"Mean gap23 diff:   {np.mean(gap_diffs):.4f}  max: {max(gap_diffs):.4f}")
    print()

    if top1_mismatch[:10]:
        print(f"Top1 mismatch races (first 10): {top1_mismatch[:10]}")
        print()

    if focus:
        print("=== Focus races ===")
        for rid in sorted(focus):
            print(f"\nRace {rid}")
            for label, src in [("FULL", full_out), ("SNAP", snap_out)]:
                if rid not in src:
                    print(f"  {label}: NOT PRESENT")
                    continue
                d = src[rid]
                print(
                    f"  {label}: top1={d['top1_boat']} rank=[{d['rank2_boat']},{d['rank3_boat']}] "
                    f"p1={d['p1']:.4f} p2={d['p2']:.4f} p3={d['p3']:.4f} "
                    f"conc={d['top3_conc']:.4f} gap23={d['gap23']:.4f}"
                )
                print(f"    boats={d['boats']} scores={[round(s,3) for s in d['scores']]}")


if __name__ == "__main__":
    main()
