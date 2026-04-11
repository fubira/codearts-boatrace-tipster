"""Daily P2 purchase summary over a date range.

Usage:
    cd ml && uv run python scripts/daily_p2_summary.py --from 2026-03-11 --to 2026-04-11
"""

import argparse
import contextlib
import io
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
FIELD_SIZE = 6


def _trifecta_prob(probs_6, i1, i2, i3):
    p1 = probs_6[i1]
    p2 = probs_6[i2]
    p3 = probs_6[i3]
    if p1 >= 1.0 or (p1 + p2) >= 1.0:
        return 0.0
    return p1 * (p2 / (1 - p1)) * (p3 / (1 - p1 - p2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--model-dir", default="models/p2_v1")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    print("Loading features...", file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    # Load model
    rank_dir = f"{args.model_dir}/ranking"
    rank_model = load_model(rank_dir)
    rank_meta = load_model_meta(rank_dir)
    strategy = rank_meta.get("strategy", {})
    gap23_th = strategy.get("gap23_threshold", 0.13)
    conc_th = strategy.get("top3_conc_threshold", 0.65)
    ev_th = strategy.get("ev_threshold", 0.0)
    feature_cols = rank_meta.get("feature_columns", [])

    print(f"Strategy: gap23>={gap23_th} conc>={conc_th} ev>={ev_th}", file=sys.stderr)

    # Load odds
    conn = get_connection(args.db_path)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '3連単'"
    ).fetchall()
    conn.close()
    trifecta_odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    # Filter date range
    test_df = df[(df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)]
    if len(test_df) == 0:
        print("No data in range")
        return

    X = test_df[feature_cols].copy()
    fill_nan_with_means(X, rank_meta)
    meta = test_df[["race_id", "boat_number", "race_date", "stadium_id", "race_number", "finish_position"]].copy()

    scores = rank_model.predict(X)
    n_races = len(X) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    dates = meta["race_date"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    y_2d = meta["finish_position"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    actual_order = np.argsort(y_2d, axis=1)
    actual_boats = np.take_along_axis(boats_2d, actual_order, axis=1).astype(int)

    # Per-day summary
    daily = defaultdict(lambda: {"total": 0, "bought": 0, "tickets": 0, "wins": 0, "cost": 0, "payout": 0})

    for i in range(n_races):
        rid = int(race_ids[i])
        date = str(dates[i])
        d = daily[date]
        d["total"] += 1

        if top_boats[i, 0] != 1:
            continue
        po = pred_order[i]
        probs = model_probs[i]
        p1, p2, p3 = float(probs[po[0]]), float(probs[po[1]]), float(probs[po[2]])
        if (p2 + p3) / (1 - p1 + 1e-10) < conc_th:
            continue
        if (p2 - p3) < gap23_th:
            continue

        r2, r3 = int(top_boats[i, 1]), int(top_boats[i, 2])
        a1, a2, a3 = int(actual_boats[i, 0]), int(actual_boats[i, 1]), int(actual_boats[i, 2])
        hit_combo = f"{a1}-{a2}-{a3}"

        tickets = []
        for combo, i_a, i_b, i_c in [
            (f"1-{r2}-{r3}", po[0], po[1], po[2]),
            (f"1-{r3}-{r2}", po[0], po[2], po[1]),
        ]:
            mkt_odds = trifecta_odds.get((rid, combo))
            if not mkt_odds or mkt_odds <= 0:
                continue
            mp = _trifecta_prob(probs, i_a, i_b, i_c)
            ev = mp / (1.0 / mkt_odds) * 0.75 - 1
            if ev >= ev_th:
                tickets.append({"combo": combo, "odds": mkt_odds})

        if not tickets:
            continue

        d["bought"] += 1
        d["tickets"] += len(tickets)
        d["cost"] += len(tickets) * 100

        for t in tickets:
            if t["combo"] == hit_combo:
                d["wins"] += 1
                d["payout"] += t["odds"] * 100
                break

    # Print
    print(f"\n{'日付':<12} {'全R':>4} {'購入':>4} {'券数':>4} {'的中':>4} {'ROI':>6} {'P/L':>10}")
    print("-" * 55)

    cum_cost = cum_payout = 0
    total_bought = total_wins = total_tickets = 0
    n_days = 0

    for date in sorted(daily.keys()):
        d = daily[date]
        n_days += 1
        total_bought += d["bought"]
        total_wins += d["wins"]
        total_tickets += d["tickets"]
        cum_cost += d["cost"]
        cum_payout += d["payout"]
        roi = d["payout"] / d["cost"] if d["cost"] > 0 else 0
        pl = d["payout"] - d["cost"]
        marker = "+" if pl > 0 else ("-" if pl < 0 else " ")
        bought_str = f"{d['bought']}R" if d["bought"] > 0 else "-"
        print(f"{date:<12} {d['total']:>4} {bought_str:>4} {d['tickets']:>4} {d['wins']:>4} "
              f"{roi:>5.0%} {pl:>+9,.0f} {marker}")

    print("-" * 55)
    total_roi = cum_payout / cum_cost if cum_cost > 0 else 0
    print(f"{'合計':<12} {'':>4} {total_bought:>4} {total_tickets:>4} {total_wins:>4} "
          f"{total_roi:>5.0%} {cum_payout - cum_cost:>+9,.0f}")
    print(f"\n{n_days}日間, {total_bought}R ({total_bought/n_days:.1f}R/日), "
          f"{total_wins}W ({total_wins/total_bought*100:.1f}%), ROI {total_roi:.0%}")


if __name__ == "__main__":
    main()
