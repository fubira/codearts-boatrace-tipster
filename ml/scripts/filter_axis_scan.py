"""Filter axis scanner — stratify baseline P2 results by a new signal axis.

Usage:
    cd ml && uv run python scripts/filter_axis_scan.py \
        --from 2026-01-01 --to 2026-04-13 --axis gap34

Axes:
  gap34:   p3 - p4 (rank-3 vs rank-4 separation)
  p1:      top-1 softmax confidence (absolute)
  entropy: score entropy across 6 boats (race difficulty)

Loads the active model + its strategy filters (including gap12_min_threshold),
enumerates races passing all existing filters + at least one EV+ ticket, then
stratifies by the chosen axis to find low-confidence cliffs similar to the
gap12 discovery.
"""

import argparse
import contextlib
import io
import sys

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
from boatrace_tipster_ml.registry import get_active_model_dir

FIELD_SIZE = 6
UNIT_COST = 100


def _trifecta_prob(probs_6, i1, i2, i3):
    p1, p2, p3 = probs_6[i1], probs_6[i2], probs_6[i3]
    if p1 >= 1.0 or (p1 + p2) >= 1.0:
        return 0.0
    return p1 * (p2 / (1 - p1)) * (p3 / (1 - p1 - p2))


def _load_confirmed_odds(db_path, from_date, to_date):
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT o.race_id, o.combination, o.odds
        FROM db.race_odds o
        JOIN db.races r ON r.id = o.race_id
        WHERE o.bet_type = '3連単'
          AND r.race_date >= ? AND r.race_date < ?
          AND o.odds > 0
        """,
        [from_date, to_date],
    ).fetchall()
    conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def _axis_value(axis: str, probs, po):
    if axis == "gap34":
        return float(probs[po[2]]) - float(probs[po[3]])
    if axis == "p1":
        return float(probs[po[0]])
    if axis == "entropy":
        p = probs
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))
    raise ValueError(f"Unknown axis: {axis}")


SWEEP_POINTS = {
    "gap34":   [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10],
    "p1":      [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    "entropy": [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60],
}

# For gap34/p1 the cliff is at the LOW end (skip races below the threshold).
# For entropy the cliff is at the HIGH end (skip races above the threshold).
SWEEP_DIRECTION = {"gap34": "ge", "p1": "ge", "entropy": "le"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--axis", choices=list(SWEEP_POINTS.keys()), default="gap34")
    parser.add_argument("--model-dir", default=get_active_model_dir())
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    print("Loading features...", file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    rank_dir = f"{args.model_dir}/ranking"
    rank_model = load_model(rank_dir)
    rank_meta = load_model_meta(rank_dir)
    feature_cols = rank_meta["feature_columns"]
    strategy = rank_meta["strategy"]
    gap12_th = strategy.get("gap12_min_threshold", 0.0)
    conc_th = strategy.get("top3_conc_threshold", 0.0)
    gap23_th = strategy.get("gap23_threshold", 0.0)
    ev_th = strategy.get("ev_threshold", 0.0)

    print(
        f"Strategy: gap12≥{gap12_th} conc≥{conc_th} gap23≥{gap23_th} ev≥{ev_th} | scanning axis={args.axis}",
        file=sys.stderr,
    )

    odds = _load_confirmed_odds(args.db_path, args.from_date, args.to_date)

    test_df = df[(df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)]
    X = test_df[feature_cols].copy()
    fill_nan_with_means(X, rank_meta)
    meta = test_df[["race_id", "boat_number", "finish_position"]].copy()
    scores = rank_model.predict(X)

    n_races = len(X) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = meta["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    y_2d = meta["finish_position"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    actual_order = np.argsort(y_2d, axis=1)
    actual_boats = np.take_along_axis(boats_2d, actual_order, axis=1).astype(int)

    records = []
    for i in range(n_races):
        if top_boats[i, 0] != 1:
            continue
        po = pred_order[i]
        probs = model_probs[i]
        p1, p2, p3 = float(probs[po[0]]), float(probs[po[1]]), float(probs[po[2]])

        if (p1 - p2) < gap12_th:
            continue
        if (p2 + p3) / (1 - p1 + 1e-10) < conc_th:
            continue
        if (p2 - p3) < gap23_th:
            continue

        axis_val = _axis_value(args.axis, probs, po)

        rid = int(rids[i])
        r2, r3 = int(top_boats[i, 1]), int(top_boats[i, 2])
        i1, i2, i3 = po[0], po[1], po[2]
        a1, a2, a3 = int(actual_boats[i, 0]), int(actual_boats[i, 1]), int(actual_boats[i, 2])
        hit_combo = f"{a1}-{a2}-{a3}"

        tickets = []
        for combo, ia, ib, ic in [
            (f"1-{r2}-{r3}", i1, i2, i3),
            (f"1-{r3}-{r2}", i1, i3, i2),
        ]:
            mkt = odds.get((rid, combo))
            if not mkt or mkt <= 0:
                continue
            mp = _trifecta_prob(probs, ia, ib, ic)
            ev = mp / (1.0 / mkt) * 0.75 - 1
            if ev >= ev_th:
                tickets.append({"combo": combo, "odds": mkt})

        if not tickets:
            continue

        payout = 0.0
        won = False
        for t in tickets:
            if t["combo"] == hit_combo:
                payout = t["odds"] * UNIT_COST
                won = True
                break

        records.append({
            "axis": axis_val,
            "n_tickets": len(tickets),
            "cost": len(tickets) * UNIT_COST,
            "payout": payout,
            "won": won,
        })

    n = len(records)
    if n == 0:
        print("No records after filters.")
        return

    total_cost = sum(r["cost"] for r in records)
    total_payout = sum(r["payout"] for r in records)
    total_wins = sum(1 for r in records if r["won"])
    total_tickets = sum(r["n_tickets"] for r in records)

    print(f"\n=== Baseline candidate races: {n} ===")
    print(
        f"Total: {n}R / {total_tickets}T / {total_wins}W "
        f"hit={100*total_wins/n:.1f}% ROI={100*total_payout/total_cost:.0f}% "
        f"P/L={total_payout-total_cost:+,.0f}"
    )

    vals = np.array([r["axis"] for r in records])
    print(
        f"\n{args.axis} 分布: min={vals.min():.4f} p10={np.percentile(vals,10):.4f} "
        f"p25={np.percentile(vals,25):.4f} median={np.median(vals):.4f} "
        f"p75={np.percentile(vals,75):.4f} p90={np.percentile(vals,90):.4f} max={vals.max():.4f}"
    )

    quantiles = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    bins = [-np.inf] + list(quantiles) + [np.inf]
    print(f"\n=== {args.axis} 5分位層別 ===")
    print(
        f"{'範囲':<24} {'レース':>6} {'勝':>4} {'hit%':>6} {'平均配当':>10} "
        f"{'ROI%':>6} {'P/L':>10}"
    )
    print("-" * 80)
    for i in range(5):
        lo, hi = bins[i], bins[i + 1]
        bin_recs = [r for r in records if lo <= r["axis"] < hi]
        if not bin_recs:
            continue
        bn = len(bin_recs)
        bw = sum(1 for r in bin_recs if r["won"])
        bc = sum(r["cost"] for r in bin_recs)
        bp = sum(r["payout"] for r in bin_recs)
        avg_payout = (bp / bw) if bw > 0 else 0
        label = f"[{lo:.4f},{hi:.4f})" if np.isfinite(lo) and np.isfinite(hi) else (
            f"<{hi:.4f}" if not np.isfinite(lo) else f"≥{lo:.4f}"
        )
        print(
            f"{label:<24} {bn:>6} {bw:>4} {100*bw/bn:>5.1f}% "
            f"{avg_payout:>10,.0f} {100*bp/bc if bc>0 else 0:>5.0f}% {bp-bc:>+10,.0f}"
        )

    direction = SWEEP_DIRECTION[args.axis]
    op_label = "≥" if direction == "ge" else "≤"
    print(f"\n=== {args.axis} フィルタスイープ（{args.axis} {op_label} th を残す）===")
    print(
        f"{args.axis+op_label:>10} {'レース':>6} {'勝':>4} {'hit%':>6} "
        f"{'ROI%':>6} {'P/L':>10}"
    )
    print("-" * 60)
    for th in SWEEP_POINTS[args.axis]:
        if direction == "ge":
            kept = [r for r in records if r["axis"] >= th]
        else:
            kept = [r for r in records if r["axis"] <= th]
        if not kept:
            continue
        kn = len(kept)
        kw = sum(1 for r in kept if r["won"])
        kc = sum(r["cost"] for r in kept)
        kp = sum(r["payout"] for r in kept)
        print(
            f"{th:>+9.4f} {kn:>6} {kw:>4} {100*kw/kn:>5.1f}% "
            f"{100*kp/kc if kc>0 else 0:>5.0f}% {kp-kc:>+10,.0f}"
        )


if __name__ == "__main__":
    main()
