"""Analyze a single day's P2 predictions vs filter thresholds.

Classifies each race into buckets (BOUGHT / not_b1 / gap12_low / conc_low /
gap23_low / ev_low) and computes hit rate per bucket. Also lists the closest
borderline skips (races that barely missed the filter cutoff) and shows what
would happen if we relaxed each threshold by a small margin.

Use case: daily review of whether today's picks were reasonable and whether
any near-misses would have hit if thresholds were slightly looser.

Usage:
    cd ml && uv run python -m scripts.analyze_decisions              # latest day in DB
    cd ml && uv run python -m scripts.analyze_decisions --date 2026-04-14
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from scripts.tune_p2 import FEATURES, _load_trifecta_odds, _trifecta_prob

FIELD_SIZE = 6


def _latest_race_date(db_path: str) -> str:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT MAX(race_date) FROM db.races").fetchone()
    finally:
        conn.close()
    return str(row[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Race date YYYY-MM-DD (default: latest date in DB)")
    ap.add_argument("--model-dir", default="models/p2_v2")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    if args.date is None:
        args.date = _latest_race_date(args.db_path)
        print(f"Using latest race date: {args.date}", file=sys.stderr)

    with open(f"{args.model_dir}/ranking/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{args.model_dir}/ranking/model_meta.json") as f:
        meta = json.load(f)
    strategy = meta["strategy"]
    feature_means = meta["feature_means"]
    gap23_th = strategy["gap23_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap12_th = strategy["gap12_min_threshold"]
    ev_th = strategy["ev_threshold"]

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    df = df_full[df_full["race_date"] == args.date]
    df = df.sort_values(["race_id", "boat_number"]).reset_index(drop=True)

    odds_map = _load_trifecta_odds(args.db_path)

    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = X[c].fillna(feature_means.get(c, 0.0))
    scores = model.predict(X)
    n_races = len(df) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = df["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    finish_2d = df["finish_position"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    decisions = []
    for i in range(n_races):
        rid = int(rids[i])
        r1, r2, r3 = top_boats[i, 0], top_boats[i, 1], top_boats[i, 2]
        p1 = model_probs[i, pred_order[i, 0]]
        p2 = model_probs[i, pred_order[i, 1]]
        p3 = model_probs[i, pred_order[i, 2]]
        gap12 = p1 - p2
        conc = (p2 + p3) / (1 - p1 + 1e-10)
        gap23 = p2 - p3

        # Actual result
        actual_order = np.argsort(finish_2d[i])
        a1, a2, a3 = (boats_2d[i, actual_order[0]],
                      boats_2d[i, actual_order[1]],
                      boats_2d[i, actual_order[2]])
        hit_combo = f"{int(a1)}-{int(a2)}-{int(a3)}"

        # P2 tickets EV
        i1, i2, i3 = pred_order[i, 0], pred_order[i, 1], pred_order[i, 2]
        tickets = []
        max_ev = -999
        for combo, prob_fn in [
            (f"{r1}-{r2}-{r3}", lambda: _trifecta_prob(model_probs[i], i1, i2, i3)),
            (f"{r1}-{r3}-{r2}", lambda: _trifecta_prob(model_probs[i], i1, i3, i2)),
        ]:
            odds = odds_map.get((rid, combo))
            if not odds or odds <= 0:
                continue
            mp = prob_fn()
            ev = mp / (1 / odds) * 0.75 - 1
            tickets.append({"combo": combo, "odds": odds, "ev": ev, "hit": combo == hit_combo})
            if ev > max_ev:
                max_ev = ev

        # Classify decision
        if r1 != 1:
            reason = "not_b1"
            margin = None
        elif gap12 < gap12_th:
            reason = "gap12_low"
            margin = gap12_th - gap12  # positive = margin from threshold
        elif conc < conc_th:
            reason = "conc_low"
            margin = conc_th - conc
        elif gap23 < gap23_th:
            reason = "gap23_low"
            margin = gap23_th - gap23
        elif max_ev < ev_th:
            reason = "ev_low"
            margin = ev_th - max_ev
        else:
            reason = "BOUGHT"
            margin = 0.0

        # Would any ticket have hit?
        any_hit = any(t["hit"] for t in tickets)
        hit_ticket = next((t for t in tickets if t["hit"]), None)

        decisions.append({
            "rid": rid, "r1": int(r1), "reason": reason, "margin": margin,
            "p1": float(p1), "gap12": float(gap12), "conc": float(conc),
            "gap23": float(gap23), "max_ev": float(max_ev),
            "tickets": tickets, "any_hit": any_hit, "hit_ticket": hit_ticket,
            "hit_combo": hit_combo,
        })

    # Summary by reason
    print(f"\n=== p2_v2 decisions for {args.date} (total {n_races} races) ===\n")
    from collections import defaultdict
    buckets = defaultdict(list)
    for d in decisions:
        buckets[d["reason"]].append(d)

    print(f"{'bucket':<15} {'count':>5} {'hit':>4} {'hit%':>6} {'notes':<40}")
    print("-" * 75)
    for reason in ["BOUGHT", "not_b1", "gap12_low", "conc_low", "gap23_low", "ev_low"]:
        items = buckets.get(reason, [])
        if not items:
            continue
        hits = sum(1 for d in items if d["any_hit"])
        pct = 100 * hits / len(items) if items else 0
        print(f"{reason:<15} {len(items):>5} {hits:>4} {pct:>5.1f}%")

    # Borderline analysis: how close did BOUGHT skip over near-miss races
    print(f"\n=== Borderline analysis (top 10 closest to BUY threshold) ===\n")
    skipped = [d for d in decisions if d["reason"] != "BOUGHT" and d["reason"] != "not_b1"
               and d["margin"] is not None]
    skipped.sort(key=lambda d: d["margin"])

    print(f"{'rid':>8} {'reason':<12} {'margin':>8} {'gap12':>7} {'conc':>6} {'gap23':>7} {'max_ev':>7} {'hit':>4}")
    for d in skipped[:20]:
        hit_mark = "HIT" if d["any_hit"] else "-"
        print(f"  {d['rid']:>6} {d['reason']:<12} {d['margin']:>+8.3f} "
              f"{d['gap12']:>7.3f} {d['conc']:>6.3f} {d['gap23']:>7.3f} {d['max_ev']:>+7.3f} {hit_mark:>4}")

    # Bought races detail
    print(f"\n=== Bought races detail ===\n")
    for d in buckets.get("BOUGHT", []):
        print(f"  rid={d['rid']} gap12={d['gap12']:.3f} conc={d['conc']:.3f} "
              f"gap23={d['gap23']:.3f} ev={d['max_ev']:+.3f} → {d['hit_combo']} "
              f"{'HIT' if d['any_hit'] else 'MISS'}")
        for t in d["tickets"]:
            mark = "★" if t["hit"] else " "
            print(f"    {mark}{t['combo']} @ {t['odds']:.1f} (EV {t['ev']:+.1%})")

    # "If we lowered the bar" analysis.
    # NOTE: Counts races that PASSED this specific filter at the relaxed
    # threshold. Subsequent filters (next in chain: conc → gap23 → ev) may
    # still reject them. This is an UPPER BOUND for "extra races if threshold
    # relaxed", not the actual incremental buy count. To get the actual count,
    # run the full filter chain with the relaxed threshold.
    print(f"\n=== What if we relaxed each threshold by small margin (upper bound) ===\n")
    for delta_name, delta_val in [("gap12 -0.005", 0.005), ("gap12 -0.01", 0.01),
                                    ("conc -0.02", 0.02), ("conc -0.05", 0.05),
                                    ("gap23 -0.01", 0.01), ("gap23 -0.02", 0.02)]:
        reason_filter = delta_name.split()[0] + "_low"
        if reason_filter == "gap12_low":
            extra = [d for d in decisions if d["reason"] == "gap12_low" and d["margin"] <= delta_val]
        elif reason_filter == "conc_low":
            extra = [d for d in decisions if d["reason"] == "conc_low" and d["margin"] <= delta_val]
        elif reason_filter == "gap23_low":
            extra = [d for d in decisions if d["reason"] == "gap23_low" and d["margin"] <= delta_val]
        else:
            continue
        hits = sum(1 for d in extra if d["any_hit"])
        print(f"  {delta_name}: +{len(extra)} races ({hits} hits)")


if __name__ == "__main__":
    main()
