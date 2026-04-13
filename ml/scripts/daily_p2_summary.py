"""Daily P2 purchase summary — practice-first numbers.

Practice number per race uses fallback chain matching runner behavior:
  1. T-5 buy + T-1 drift check (if both snapshots exist) → closest to real runner
  2. T-5 buy only (if T-1 missing)
  3. Confirmed odds (if neither snapshot exists; backtest fallback pre-snapshot era)

Also shows confirmed-odds number as a reference column.

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
from boatrace_tipster_ml.registry import get_active_model_dir

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


def _select_p2_tickets(probs, po, odds_map, rid, r2, r3, ev_threshold):
    """Select P2 tickets passing EV threshold using given odds map.

    Returns list of {combo, odds, model_prob} and a reason if no tickets
    could be selected ("no_odds" if odds missing for both, "ev_low" if
    both failed EV, None if at least one passed)."""
    tickets = []
    odds_hits = 0
    for combo, ia, ib, ic in [
        (f"1-{r2}-{r3}", po[0], po[1], po[2]),
        (f"1-{r3}-{r2}", po[0], po[2], po[1]),
    ]:
        mkt_odds = odds_map.get((rid, combo))
        if not mkt_odds or mkt_odds <= 0:
            continue
        odds_hits += 1
        mp = _trifecta_prob(probs, ia, ib, ic)
        ev = mp / (1.0 / mkt_odds) * 0.75 - 1
        if ev >= ev_threshold:
            tickets.append({"combo": combo, "odds": mkt_odds, "model_prob": mp})
    if not tickets:
        return [], ("no_odds" if odds_hits == 0 else "ev_low")
    return tickets, None


def _apply_t1_drift(tickets, t1_odds, rid, ev_threshold):
    """Re-check tickets against T-1 odds (mirrors runner.ts T-1 drift)."""
    surviving = []
    for t in tickets:
        new_odds = t1_odds.get((rid, t["combo"]))
        if not new_odds or new_odds <= 0:
            continue
        new_ev = t["model_prob"] / (1.0 / new_odds) * 0.75 - 1
        if new_ev >= ev_threshold:
            surviving.append({"combo": t["combo"], "odds": new_odds, "model_prob": t["model_prob"]})
    return surviving


def evaluate_day_practice(races_in_day, t5_odds, t1_odds, confirmed_odds, ev_threshold):
    """Per-race practice evaluation with fallback chain T-5+T-1 → T-5 → confirmed.

    Returns aggregate (bought, tickets, wins, cost, payout) and source counters.
    """
    bought = 0
    total_tickets = 0
    wins = 0
    cost = 0.0
    payout = 0.0
    src_counts = {"T-1": 0, "T-5": 0, "conf": 0, "none": 0}

    for rd in races_in_day:
        rid = rd["rid"]
        probs = rd["probs"]
        po = rd["po"]
        top_boats = rd["top_boats"]
        actual = rd["actual"]

        r2, r3 = int(top_boats[1]), int(top_boats[2])
        hit_combo = f"{actual[0]}-{actual[1]}-{actual[2]}"

        # Decide source via fallback chain
        has_t5 = any(t5_odds.get((rid, c)) for c in [f"1-{r2}-{r3}", f"1-{r3}-{r2}"])
        has_t1 = any(t1_odds.get((rid, c)) for c in [f"1-{r2}-{r3}", f"1-{r3}-{r2}"])

        if has_t5:
            tickets, _ = _select_p2_tickets(probs, po, t5_odds, rid, r2, r3, ev_threshold)
            if has_t1 and tickets:
                # Real runner flow: T-5 buy + T-1 drift re-check
                tickets = _apply_t1_drift(tickets, t1_odds, rid, ev_threshold)
                src = "T-1"
            else:
                src = "T-5"
        else:
            tickets, _ = _select_p2_tickets(probs, po, confirmed_odds, rid, r2, r3, ev_threshold)
            src = "conf"

        if not tickets:
            src_counts["none"] += 1
            continue

        src_counts[src] += 1
        bought += 1
        total_tickets += len(tickets)
        cost += len(tickets) * 100

        for t in tickets:
            if t["combo"] == hit_combo:
                wins += 1
                payout += t["odds"] * 100
                break

    return bought, total_tickets, wins, cost, payout, src_counts


def evaluate_day(races_in_day, trifecta_odds, ev_threshold):
    """Backtest-only evaluation with a single odds source (reference column)."""
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

        tickets, _ = _select_p2_tickets(probs, po, trifecta_odds, rid, r2, r3, ev_threshold)
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
    parser.add_argument("--model-dir", default=get_active_model_dir())
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
    gap12_th = strategy.get("gap12_min_threshold", 0.0)

    print(
        f"Strategy: gap12>={gap12_th} gap23>={gap23_th} conc>={conc_th} ev>={ev_th}",
        file=sys.stderr,
    )

    # Load T-5, T-1 snapshot odds AND confirmed odds
    print("Loading odds...", file=sys.stderr, flush=True)
    t5_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-5")
    t1_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-1")
    confirmed_odds = _load_confirmed_odds(args.db_path, args.from_date, args.to_date)

    # Coverage
    t5_rids = set(rid for rid, _ in t5_odds.keys())
    t1_rids = set(rid for rid, _ in t1_odds.keys())
    conf_rids = set(rid for rid, _ in confirmed_odds.keys())
    print(
        f"Coverage: T-5 {len(t5_rids)} races, T-1 {len(t1_rids)} races, confirmed {len(conf_rids)} races",
        file=sys.stderr,
    )

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

    # Pre-filter races by model (B1 top + gap12 + conc + gap23)
    daily_candidates = defaultdict(list)
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

        date = str(dates[i])
        daily_candidates[date].append({
            "rid": int(rids[i]),
            "probs": probs,
            "po": po,
            "top_boats": top_boats[i],
            "actual": (int(actual_boats[i, 0]), int(actual_boats[i, 1]), int(actual_boats[i, 2])),
        })

    # Per-day practice-first table (T-1 drift → T-5 → confirmed fallback) + confirmed reference
    print(
        f"\n{'日付':<12} {'候補':>4} {'src':>8} {'購入':>5} {'券':>4} {'的中':>5} "
        f"{'ROI':>6} {'P/L':>9} | {'確定R':>6} {'確ROI':>6} {'確P/L':>9}"
    )
    print("-" * 95)

    pr_tot = {"b": 0, "t": 0, "w": 0, "c": 0.0, "p": 0.0}
    cf_tot = {"b": 0, "t": 0, "w": 0, "c": 0.0, "p": 0.0}
    src_tot = {"T-1": 0, "T-5": 0, "conf": 0, "none": 0}

    for date in sorted(daily_candidates.keys()):
        cands = daily_candidates[date]

        pr_b, pr_t, pr_w, pr_c, pr_p, src = evaluate_day_practice(
            cands, t5_odds, t1_odds, confirmed_odds, ev_th
        )
        cf_b, cf_t, cf_w, cf_c, cf_p = evaluate_day(cands, confirmed_odds, ev_th)

        pr_tot["b"] += pr_b; pr_tot["t"] += pr_t; pr_tot["w"] += pr_w
        pr_tot["c"] += pr_c; pr_tot["p"] += pr_p
        cf_tot["b"] += cf_b; cf_tot["t"] += cf_t; cf_tot["w"] += cf_w
        cf_tot["c"] += cf_c; cf_tot["p"] += cf_p
        for k in src_tot:
            src_tot[k] += src[k]

        pr_roi = pr_p / pr_c if pr_c > 0 else 0
        pr_pl = pr_p - pr_c
        cf_roi = cf_p / cf_c if cf_c > 0 else 0
        cf_pl = cf_p - cf_c

        # Day-level source label: dominant non-none
        bought_by_src = {k: v for k, v in src.items() if k != "none" and v > 0}
        if not bought_by_src:
            src_label = "-"
        elif len(bought_by_src) == 1:
            src_label = next(iter(bought_by_src))
        else:
            src_label = "mix"

        pr_buy_str = f"{pr_b}R" if pr_b > 0 else " -"
        cf_buy_str = f"{cf_b}R" if cf_b > 0 else " -"

        print(
            f"{date:<12} {len(cands):>4} {src_label:>8} {pr_buy_str:>5} {pr_t:>4} {pr_w:>5} "
            f"{pr_roi:>5.0%} {pr_pl:>+9,.0f} | {cf_buy_str:>6} {cf_roi:>5.0%} {cf_pl:>+9,.0f}"
        )

    # Count days with NO candidates at all
    all_dates = sorted(set(str(d) for d in dates))
    no_cand_days = [d for d in all_dates if d not in daily_candidates]

    print("-" * 95)
    pr_roi = pr_tot["p"] / pr_tot["c"] if pr_tot["c"] > 0 else 0
    cf_roi = cf_tot["p"] / cf_tot["c"] if cf_tot["c"] > 0 else 0
    print(
        f"{'合計':<12} {'':>4} {'':>8} {pr_tot['b']:>4}R {pr_tot['t']:>4} {pr_tot['w']:>5} "
        f"{pr_roi:>5.0%} {pr_tot['p']-pr_tot['c']:>+9,.0f} | {cf_tot['b']:>5}R {cf_roi:>5.0%} "
        f"{cf_tot['p']-cf_tot['c']:>+9,.0f}"
    )

    total_days = len(all_dates)
    print(f"\n期間: {total_days}日 / 候補あり: {len(daily_candidates)}日 / 候補なし: {len(no_cand_days)}日")
    print(
        f"実践 (T-1 drift→T-5→確定 fallback): {pr_tot['b']}R, "
        f"{pr_tot['w']}W ({pr_tot['w']/pr_tot['b']*100 if pr_tot['b']>0 else 0:.1f}%), "
        f"ROI {pr_roi:.0%}, P/L {pr_tot['p']-pr_tot['c']:+,.0f}"
    )
    print(
        f"  内訳: T-1={src_tot['T-1']}R, T-5={src_tot['T-5']}R, 確定={src_tot['conf']}R "
        f"(EV不通過/オッズ欠落 {src_tot['none']}R)"
    )
    print(
        f"確定参考: {cf_tot['b']}R, {cf_tot['w']}W "
        f"({cf_tot['w']/cf_tot['b']*100 if cf_tot['b']>0 else 0:.1f}%), ROI {cf_roi:.0%}, "
        f"P/L {cf_tot['p']-cf_tot['c']:+,.0f}"
    )
    if no_cand_days:
        head = ', '.join(no_cand_days[:10])
        tail = '...' if len(no_cand_days) > 10 else ''
        print(f"候補なし（conc+gap23 不通過）の日 ({len(no_cand_days)}日): {head}{tail}")


if __name__ == "__main__":
    main()
