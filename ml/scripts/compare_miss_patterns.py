"""Compare miss patterns between recent days and historical baseline.

Question: are recent days' losses within the "normal" distribution of
historical losses, or has something shifted? Runs confirmed-odds-based
predictions for both periods and compares:

  - hit rate
  - miss breakdown (boat1 not 1st vs ordering wrong)
  - filter value averages (p1, gap12, conc, gap23)
  - boat 1 actual finish position distribution

Use case: weekly health check. If recent miss pattern is statistically
indistinguishable from baseline, recent losses are just bad luck. If the
pattern shifts, either the model is drifting or something in the data
pipeline has changed.

Defaults to "recent = last 7 days, baseline = 30 days before that" ending
at the latest race date in the DB.

Usage:
    # Default: recent = last 7 days, baseline = 30 days before
    cd ml && uv run python -m scripts.compare_miss_patterns

    # Custom windows
    cd ml && uv run python -m scripts.compare_miss_patterns \\
        --recent-days 4 --baseline-days 27
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from scripts.tune_p2 import FEATURES, _load_trifecta_odds, _trifecta_prob

FIELD_SIZE = 6


def analyze_period(df_full, odds_map, model, feature_means, strategy,
                   from_date, to_date):
    gap23_th = strategy["gap23_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap12_th = strategy["gap12_min_threshold"]
    ev_th = strategy["ev_threshold"]

    df = df_full[(df_full["race_date"] >= from_date) & (df_full["race_date"] <= to_date)]
    df = df.sort_values(["race_id", "boat_number"]).reset_index(drop=True)

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

    bought = []
    for i in range(n_races):
        rid = int(rids[i])
        r1, r2, r3 = top_boats[i, 0], top_boats[i, 1], top_boats[i, 2]
        p1 = model_probs[i, pred_order[i, 0]]
        p2 = model_probs[i, pred_order[i, 1]]
        p3 = model_probs[i, pred_order[i, 2]]
        gap12 = p1 - p2
        conc = (p2 + p3) / (1 - p1 + 1e-10)
        gap23 = p2 - p3

        if r1 != 1 or gap12 < gap12_th or conc < conc_th or gap23 < gap23_th:
            continue

        i1, i2, i3 = pred_order[i, 0], pred_order[i, 1], pred_order[i, 2]
        tickets = []
        for combo, prob_fn in [
            (f"{r1}-{r2}-{r3}", lambda: _trifecta_prob(model_probs[i], i1, i2, i3)),
            (f"{r1}-{r3}-{r2}", lambda: _trifecta_prob(model_probs[i], i1, i3, i2)),
        ]:
            odds = odds_map.get((rid, combo))
            if not odds or odds <= 0:
                continue
            mp = prob_fn()
            ev = mp / (1 / odds) * 0.75 - 1
            if ev >= ev_th:
                tickets.append({"combo": combo, "odds": odds, "ev": ev})
        if not tickets:
            continue

        # Actual result
        actual_order = np.argsort(finish_2d[i])
        a1 = int(boats_2d[i, actual_order[0]])
        a2 = int(boats_2d[i, actual_order[1]])
        a3 = int(boats_2d[i, actual_order[2]])
        hit_combo = f"{a1}-{a2}-{a3}"
        any_hit = any(t["combo"] == hit_combo for t in tickets)

        # Boat 1 actual finish position (1-6)
        boat1_idx = np.where(boats_2d[i] == 1)[0][0]
        boat1_finish = int(finish_2d[i, boat1_idx])  # 1=1st, 2=2nd, etc. NaN=DSQ

        # Classify miss reason
        if any_hit:
            miss_reason = "HIT"
        elif boat1_finish != 1:
            miss_reason = f"boat1_{boat1_finish}th" if boat1_finish else "boat1_DSQ"
        else:
            # Boat 1 won but 2/3 order wrong
            miss_reason = "2_3_order_wrong"

        bought.append({
            "rid": rid, "p1": float(p1), "gap12": float(gap12), "conc": float(conc),
            "gap23": float(gap23),
            "r1": int(r1), "r2": int(r2), "r3": int(r3),
            "actual": (a1, a2, a3),
            "boat1_finish": boat1_finish,
            "any_hit": any_hit, "miss_reason": miss_reason,
            "tickets": tickets,
        })
    return bought


def print_summary(label, bought):
    hits = [b for b in bought if b["any_hit"]]
    misses = [b for b in bought if not b["any_hit"]]
    hit_pct = 100 * len(hits) / len(bought) if bought else 0
    print(f"\n=== {label} ===")
    print(f"Bought: {len(bought)} races, Hit: {len(hits)} ({hit_pct:.1f}%)")
    print()

    # Miss reason breakdown
    print("Miss breakdown:")
    reasons = Counter(b["miss_reason"] for b in bought)
    for reason, count in reasons.most_common():
        pct = 100 * count / len(bought) if bought else 0
        bar = "#" * int(pct / 3)
        print(f"  {reason:<22}: {count:>4} ({pct:>4.1f}%)  {bar}")

    # Filter value distributions (means)
    if bought:
        print()
        print("Filter value averages:")
        for key in ["p1", "gap12", "conc", "gap23"]:
            vals = [b[key] for b in bought]
            print(f"  {key:<7}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f}")

    # For misses only: boat 1 finish distribution
    if misses:
        print()
        print("Miss: boat 1 actual finish position:")
        finish_dist = Counter(m["boat1_finish"] for m in misses)
        for pos in sorted(finish_dist.keys()):
            count = finish_dist[pos]
            pct = 100 * count / len(misses)
            pos_label = f"{pos}位" if pos else "失格"
            print(f"  {pos_label}: {count:>4} ({pct:>4.1f}%)")


def _latest_race_date(db_path: str) -> pd.Timestamp:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT MAX(race_date) FROM db.races").fetchone()
    finally:
        conn.close()
    return pd.Timestamp(str(row[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recent-days", type=int, default=7,
                    help="Recent window size in days (default: 7)")
    ap.add_argument("--baseline-days", type=int, default=30,
                    help="Baseline window size in days, ending where recent starts (default: 30)")
    ap.add_argument("--to", dest="to_date", default=None,
                    help="End date YYYY-MM-DD (default: latest race in DB)")
    ap.add_argument("--recent-from", default=None,
                    help="Override recent window start (YYYY-MM-DD)")
    ap.add_argument("--recent-to", default=None,
                    help="Override recent window end (YYYY-MM-DD)")
    ap.add_argument("--baseline-from", default=None,
                    help="Override baseline window start (YYYY-MM-DD)")
    ap.add_argument("--baseline-to", default=None,
                    help="Override baseline window end (YYYY-MM-DD)")
    ap.add_argument("--model-dir", default="models/p2_v2")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    # Resolve date windows. Explicit --recent-from / --recent-to etc.
    # override the default "last N days ending at latest DB date" rule.
    end = pd.Timestamp(args.to_date) if args.to_date else _latest_race_date(args.db_path)
    if args.recent_to is None:
        args.recent_to = str(end.date())
    if args.recent_from is None:
        args.recent_from = str((end - pd.Timedelta(days=args.recent_days - 1)).date())
    if args.baseline_to is None:
        args.baseline_to = str((pd.Timestamp(args.recent_from) - pd.Timedelta(days=1)).date())
    if args.baseline_from is None:
        args.baseline_from = str((pd.Timestamp(args.baseline_to) - pd.Timedelta(days=args.baseline_days - 1)).date())

    print(f"Baseline: {args.baseline_from} ~ {args.baseline_to}")
    print(f"Recent:   {args.recent_from} ~ {args.recent_to}")

    with open(f"{args.model_dir}/ranking/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{args.model_dir}/ranking/model_meta.json") as f:
        meta = json.load(f)
    strategy = meta["strategy"]
    feature_means = meta["feature_means"]

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    odds_map = _load_trifecta_odds(args.db_path)

    recent = analyze_period(df_full, odds_map, model, feature_means, strategy,
                            args.recent_from, args.recent_to)
    baseline = analyze_period(df_full, odds_map, model, feature_means, strategy,
                              args.baseline_from, args.baseline_to)

    print_summary(f"BASELINE {args.baseline_from} ~ {args.baseline_to}", baseline)
    print_summary(f"RECENT {args.recent_from} ~ {args.recent_to}", recent)

    # Direct comparison
    print()
    print("=" * 60)
    print("Comparison table")
    print("=" * 60)
    print(f"{'metric':<30} {'baseline':>12} {'recent':>12}")

    def fmt(x, is_pct=False):
        return f"{x:.3f}" if not is_pct else f"{x*100:.1f}%"

    def period_stats(bought):
        if not bought:
            return {}
        hits = sum(1 for b in bought if b["any_hit"])
        n = len(bought)
        misses = [b for b in bought if not b["any_hit"]]
        m_boat1_not_1st = sum(1 for m in misses if m["boat1_finish"] != 1)
        m_ordering = sum(1 for m in misses if m["boat1_finish"] == 1)
        return {
            "n": n,
            "hits": hits,
            "hit_pct": hits / n,
            "misses": len(misses),
            "boat1_not_1st_pct": m_boat1_not_1st / len(misses) if misses else 0,
            "ordering_wrong_pct": m_ordering / len(misses) if misses else 0,
            "p1_mean": np.mean([b["p1"] for b in bought]),
            "gap12_mean": np.mean([b["gap12"] for b in bought]),
            "conc_mean": np.mean([b["conc"] for b in bought]),
            "gap23_mean": np.mean([b["gap23"] for b in bought]),
        }

    bs = period_stats(baseline)
    rs = period_stats(recent)

    for key, label, pct in [
        ("n", "bought races", False),
        ("hits", "hits", False),
        ("hit_pct", "hit rate", True),
        ("boat1_not_1st_pct", "miss: boat1 NOT 1st (%)", True),
        ("ordering_wrong_pct", "miss: ordering wrong (%)", True),
        ("p1_mean", "avg p1 (model top prob)", False),
        ("gap12_mean", "avg gap12", False),
        ("conc_mean", "avg conc", False),
        ("gap23_mean", "avg gap23", False),
    ]:
        bv = bs.get(key, 0)
        rv = rs.get(key, 0)
        if pct:
            print(f"{label:<30} {bv*100:>11.1f}% {rv*100:>11.1f}%")
        elif key in ("n", "hits"):
            print(f"{label:<30} {int(bv):>12} {int(rv):>12}")
        else:
            print(f"{label:<30} {bv:>12.3f} {rv:>12.3f}")


if __name__ == "__main__":
    main()
