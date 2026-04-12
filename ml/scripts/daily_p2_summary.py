"""Daily P2 purchase summary — verified with T-5 snapshot odds.

Simulates exactly what the runner would see: T-5 timing odds only.
Also compares with confirmed odds to quantify drift impact.

Usage:
    cd ml && uv run python scripts/daily_p2_summary.py --from 2026-03-11 --to 2026-04-11
"""

import argparse
import contextlib
import io
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta

FIELD_SIZE = 6


def _trifecta_prob(probs_6, i1, i2, i3):
    p1, p2, p3 = probs_6[i1], probs_6[i2], probs_6[i3]
    if p1 >= 1.0 or (p1 + p2) >= 1.0:
        return 0.0
    return p1 * (p2 / (1 - p1)) * (p3 / (1 - p1 - p2))


def _load_odds_by_timing(db_path, from_date, to_date, timing):
    """Load trifecta odds for a specific snapshot timing."""
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT s.race_id, s.combination, s.odds
        FROM db.race_odds_snapshots s
        JOIN db.races r ON r.id = s.race_id
        WHERE s.bet_type = '3連単' AND s.timing = ?
          AND r.race_date >= ? AND r.race_date < ?
          AND s.odds > 0
        """,
        [timing, from_date, to_date],
    ).fetchall()
    conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


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


def evaluate_day(races_in_day, trifecta_odds, ev_threshold):
    """Evaluate P2 for a day's races with given odds. Returns (bought, tickets, wins, cost, payout)."""
    bought = 0
    total_tickets = 0
    wins = 0
    cost = 0.0
    payout = 0.0

    for rd in races_in_day:
        rid = rd["rid"]
        probs = rd["probs"]
        po = rd["po"]
        top_boats = rd["top_boats"]
        actual = rd["actual"]

        r2, r3 = int(top_boats[1]), int(top_boats[2])
        hit_combo = f"{actual[0]}-{actual[1]}-{actual[2]}"

        tickets = []
        for combo, ia, ib, ic in [
            (f"1-{r2}-{r3}", po[0], po[1], po[2]),
            (f"1-{r3}-{r2}", po[0], po[2], po[1]),
        ]:
            mkt_odds = trifecta_odds.get((rid, combo))
            if not mkt_odds or mkt_odds <= 0:
                continue
            mp = _trifecta_prob(probs, ia, ib, ic)
            ev = mp / (1.0 / mkt_odds) * 0.75 - 1
            if ev >= ev_threshold:
                tickets.append({"combo": combo, "odds": mkt_odds})

        if not tickets:
            continue

        bought += 1
        total_tickets += len(tickets)
        cost += len(tickets) * 100

        for t in tickets:
            if t["combo"] == hit_combo:
                wins += 1
                payout += t["odds"] * 100
                break

    return bought, total_tickets, wins, cost, payout


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

    rank_model = load_model(f"{args.model_dir}/ranking")
    rank_meta = load_model_meta(f"{args.model_dir}/ranking")
    feature_cols = rank_meta["feature_columns"]
    strategy = rank_meta["strategy"]
    gap23_th = strategy["gap23_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    ev_th = strategy["ev_threshold"]

    print(f"Strategy: gap23>={gap23_th} conc>={conc_th} ev>={ev_th}", file=sys.stderr)

    # Load T-5 snapshot odds AND confirmed odds
    print("Loading odds...", file=sys.stderr, flush=True)
    t5_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-5")
    confirmed_odds = _load_confirmed_odds(args.db_path, args.from_date, args.to_date)

    # Check T-5 coverage
    t5_rids = set(rid for rid, _ in t5_odds.keys())
    conf_rids = set(rid for rid, _ in confirmed_odds.keys())
    print(f"T-5 coverage: {len(t5_rids)} races, confirmed: {len(conf_rids)} races", file=sys.stderr)

    # Filter date range
    test_df = df[(df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)]
    if len(test_df) == 0:
        print("No data in range")
        return

    X = test_df[feature_cols].copy()
    fill_nan_with_means(X, rank_meta)
    meta = test_df[["race_id", "boat_number", "race_date", "finish_position"]].copy()
    scores = rank_model.predict(X)

    n_races = len(X) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = meta["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    dates = meta["race_date"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    y_2d = meta["finish_position"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    actual_order = np.argsort(y_2d, axis=1)
    actual_boats = np.take_along_axis(boats_2d, actual_order, axis=1).astype(int)

    # Pre-filter races by model (B1 top + conc + gap23)
    daily_candidates = defaultdict(list)
    for i in range(n_races):
        if top_boats[i, 0] != 1:
            continue
        po = pred_order[i]
        probs = model_probs[i]
        p1, p2, p3 = float(probs[po[0]]), float(probs[po[1]]), float(probs[po[2]])
        if (p2 + p3) / (1 - p1 + 1e-10) < conc_th:
            continue
        if (p2 - p3) < gap23_th:
            continue

        date = str(dates[i])
        daily_candidates[date].append({
            "rid": int(rids[i]),
            "probs": probs,
            "po": po,
            "top_boats": top_boats[i],
            "actual": (int(actual_boats[i, 0]), int(actual_boats[i, 1]), int(actual_boats[i, 2])),
        })

    # Evaluate with both odds sources
    print(f"\n{'日付':<12} {'候補':>4} {'T5購入':>6} {'T5券':>4} {'T5的中':>6} {'T5 ROI':>7} {'T5 P/L':>9} | {'確定購入':>8} {'確定ROI':>8}")
    print("-" * 90)

    t5_tot = {"b": 0, "t": 0, "w": 0, "c": 0.0, "p": 0.0}
    cf_tot = {"b": 0, "t": 0, "w": 0, "c": 0.0, "p": 0.0}
    zero_days = []
    n_days = 0

    for date in sorted(daily_candidates.keys()):
        cands = daily_candidates[date]
        n_days += 1

        t5_b, t5_t, t5_w, t5_c, t5_p = evaluate_day(cands, t5_odds, ev_th)
        cf_b, cf_t, cf_w, cf_c, cf_p = evaluate_day(cands, confirmed_odds, ev_th)

        t5_tot["b"] += t5_b; t5_tot["t"] += t5_t; t5_tot["w"] += t5_w
        t5_tot["c"] += t5_c; t5_tot["p"] += t5_p
        cf_tot["b"] += cf_b; cf_tot["t"] += cf_t; cf_tot["w"] += cf_w
        cf_tot["c"] += cf_c; cf_tot["p"] += cf_p

        t5_roi = t5_p / t5_c if t5_c > 0 else 0
        t5_pl = t5_p - t5_c
        cf_roi = cf_p / cf_c if cf_c > 0 else 0
        m = "+" if t5_pl > 0 else ("-" if t5_pl < 0 else " ")

        t5_buy_str = f"{t5_b}R" if t5_b > 0 else " -"
        cf_buy_str = f"{cf_b}R" if cf_b > 0 else " -"

        print(f"{date:<12} {len(cands):>4} {t5_buy_str:>6} {t5_t:>4} {t5_w:>6} {t5_roi:>6.0%} {t5_pl:>+9,.0f} {m} | {cf_buy_str:>8} {cf_roi:>7.0%}")

        if t5_b == 0:
            zero_days.append(date)

    # Also count days with NO candidates at all (not in daily_candidates)
    all_dates = sorted(set(str(d) for d in dates))
    no_cand_days = [d for d in all_dates if d not in daily_candidates]

    print("-" * 90)
    t5_roi = t5_tot["p"] / t5_tot["c"] if t5_tot["c"] > 0 else 0
    cf_roi = cf_tot["p"] / cf_tot["c"] if cf_tot["c"] > 0 else 0
    print(f"{'合計':<12} {'':>4} {t5_tot['b']:>6} {t5_tot['t']:>4} {t5_tot['w']:>6} "
          f"{t5_roi:>6.0%} {t5_tot['p']-t5_tot['c']:>+9,.0f}   | {cf_tot['b']:>8} {cf_roi:>7.0%}")

    cand_days = len(daily_candidates)
    total_days = len(all_dates)
    print(f"\n期間: {total_days}日")
    print(f"候補あり: {cand_days}日 / T-5購入あり: {cand_days - len(zero_days)}日 / 候補なし: {len(no_cand_days)}日")
    print(f"T-5: {t5_tot['b']}R ({t5_tot['b']/(cand_days - len(zero_days)) if (cand_days - len(zero_days)) > 0 else 0:.1f}R/購入日), "
          f"{t5_tot['w']}W ({t5_tot['w']/t5_tot['b']*100 if t5_tot['b']>0 else 0:.1f}%), ROI {t5_roi:.0%}")
    print(f"確定: {cf_tot['b']}R, {cf_tot['w']}W ({cf_tot['w']/cf_tot['b']*100 if cf_tot['b']>0 else 0:.1f}%), ROI {cf_roi:.0%}")

    if zero_days:
        print(f"\nT-5 購入 0R の日 ({len(zero_days)}日): {', '.join(zero_days)}")
    if no_cand_days:
        print(f"候補なし（conc+gap23 不通過）の日 ({len(no_cand_days)}日): {', '.join(no_cand_days[:10])}{'...' if len(no_cand_days)>10 else ''}")

    # T-5 coverage check for zero days
    if zero_days:
        print(f"\n--- T-5 購入 0R の日の詳細 ---")
        for date in zero_days:
            cands = daily_candidates[date]
            n_has_t5 = 0
            for rd in cands:
                rid = rd["rid"]
                r2, r3 = int(rd["top_boats"][1]), int(rd["top_boats"][2])
                has = any(t5_odds.get((rid, c)) for c in [f"1-{r2}-{r3}", f"1-{r3}-{r2}"])
                if has:
                    n_has_t5 += 1
            print(f"  {date}: {len(cands)}候補, T-5 odds あり={n_has_t5}, "
                  f"T-5なし={len(cands)-n_has_t5}")


if __name__ == "__main__":
    main()
