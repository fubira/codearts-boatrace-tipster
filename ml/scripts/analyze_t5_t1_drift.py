"""T-5 / T-1 drift detailed analysis for a P2 model.

Runs predictions over a date range and for each race that passed the
gap12/conc/gap23 filter chain, records:

  - T-5 tickets (ev ≥ threshold against T-5 odds)
  - T-1 surviving tickets (after re-checking ev against T-1 odds)
  - confirmed-odds tickets (ev against final odds, reference)
  - hit/miss for each ticket using actual 3連単 result

Then prints aggregate stats:

  - T-5 path: total tickets bought, hits, ROI, P/L
  - T-1 drift path: survived tickets after drift re-check, hits, ROI, P/L
  - Confirmed ref: tickets that would be bought on confirmed odds
  - Drop rate: (T-5 - T-1_survived) / T-5
  - Per-day breakdown

Requires snapshot data (race_odds_snapshots table) for the specified
period. Use 2026-04-07 onwards per current DB coverage.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from scripts._p2_decision import (
    compute_race_decisions,
    load_model_and_strategy,
)


def _load_odds_by_timing(db_path: str, from_date: str, to_date: str, timing: str) -> dict[tuple[int, str], float]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT s.race_id, s.combination, s.odds
            FROM db.race_odds_snapshots s
            JOIN db.races r ON s.race_id = r.id
            WHERE s.timing = ? AND s.bet_type = '3連単'
              AND r.race_date BETWEEN ? AND ?
            """,
            (timing, from_date, to_date),
        ).fetchall()
    finally:
        conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def _load_confirmed_odds(db_path: str, from_date: str, to_date: str) -> dict[tuple[int, str], float]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT o.race_id, o.combination, o.odds
            FROM db.race_odds o
            JOIN db.races r ON o.race_id = r.id
            WHERE o.bet_type = '3連単'
              AND r.race_date BETWEEN ? AND ?
            """,
            (from_date, to_date),
        ).fetchall()
    finally:
        conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def _evaluate_tickets(d, odds_map, ev_threshold):
    """For a RaceDecision, re-evaluate each ticket against a different
    odds source (T-1 / confirmed). Returns list of (combo, odds, ev, hit)
    for tickets that still pass ev_threshold."""
    tickets = []
    for t in d.tickets:
        o = odds_map.get((d.rid, t.combo))
        if not o or o <= 0:
            continue
        new_ev = t.model_prob / (1.0 / o) * 0.75 - 1
        if new_ev >= ev_threshold:
            tickets.append((t.combo, o, new_ev, t.hit))
    return tickets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--model-dir", default="models/p2_v3")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    print(f"Model: {args.model_dir}", file=sys.stderr)
    print(f"Period: {args.from_date} ~ {args.to_date}", file=sys.stderr)

    model, strategy, feature_means = load_model_and_strategy(args.model_dir)
    gap23_th = strategy["gap23_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap12_th = strategy["gap12_min_threshold"]
    ev_th = strategy["ev_threshold"]

    print(f"Filters: gap12≥{gap12_th} conc≥{conc_th} gap23≥{gap23_th} ev≥{ev_th}", file=sys.stderr)

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    df = df_full[(df_full["race_date"] >= args.from_date) & (df_full["race_date"] <= args.to_date)]

    t5_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-5")
    t1_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-1")
    conf_odds = _load_confirmed_odds(args.db_path, args.from_date, args.to_date)

    t5_rids = set(rid for rid, _ in t5_odds.keys())
    t1_rids = set(rid for rid, _ in t1_odds.keys())
    print(f"Coverage: T-5 {len(t5_rids)} races, T-1 {len(t1_rids)} races", file=sys.stderr)

    # compute decisions using T-5 odds as the "initial" map.
    # Then we'll re-evaluate each ticket at T-1 and confirmed.
    decisions = compute_race_decisions(df, model, feature_means, t5_odds)

    # Per-path aggregates
    t5_stats = {"races": 0, "tickets": 0, "wins": 0, "cost": 0, "payout": 0}
    t1_stats = {"races": 0, "tickets": 0, "wins": 0, "cost": 0, "payout": 0}
    conf_stats = {"races": 0, "tickets": 0, "wins": 0, "cost": 0, "payout": 0}
    drop_stats = {"dropped_tickets": 0, "dropped_races": 0}

    # Per-day for breakdown table
    per_day: dict[str, dict] = {}
    df_dates = df.sort_values(["race_date", "race_id", "boat_number"]).reset_index(drop=True)
    date_by_rid = df_dates.drop_duplicates("race_id").set_index("race_id")["race_date"].astype(str).to_dict()

    for d in decisions:
        if d.rid not in t5_rids:
            continue
        # Apply filter chain — these are rechecked per RaceDecision
        if d.r1 != 1:
            continue
        if d.gap12 < gap12_th:
            continue
        if d.conc < conc_th:
            continue
        if d.gap23 < gap23_th:
            continue

        # T-5 path: tickets from d.tickets (already built against T-5 odds)
        t5_tickets = [t for t in d.tickets if t.ev >= ev_th]
        # T-1 path: re-evaluate each T-5 ticket at T-1 odds
        t1_tickets = _evaluate_tickets(d, t1_odds, ev_th)
        # Confirmed: re-evaluate at confirmed odds
        conf_tickets_list = _evaluate_tickets(d, conf_odds, ev_th)

        date = date_by_rid.get(d.rid, "?")
        day = per_day.setdefault(date, {
            "t5_tickets": 0, "t5_wins": 0, "t5_cost": 0, "t5_payout": 0,
            "t1_tickets": 0, "t1_wins": 0, "t1_cost": 0, "t1_payout": 0,
            "dropped_tickets": 0,
        })

        # T-5 aggregation
        if t5_tickets:
            t5_stats["races"] += 1
            for t in t5_tickets:
                t5_stats["tickets"] += 1
                t5_stats["cost"] += 100
                day["t5_tickets"] += 1
                day["t5_cost"] += 100
                if t.hit:
                    t5_stats["wins"] += 1
                    t5_stats["payout"] += t.odds * 100
                    day["t5_wins"] += 1
                    day["t5_payout"] += t.odds * 100

        # T-1 aggregation (surviving tickets only)
        if t1_tickets:
            t1_stats["races"] += 1
            for combo, o, ev, is_hit in t1_tickets:
                t1_stats["tickets"] += 1
                t1_stats["cost"] += 100
                day["t1_tickets"] += 1
                day["t1_cost"] += 100
                if is_hit:
                    t1_stats["wins"] += 1
                    t1_stats["payout"] += o * 100
                    day["t1_wins"] += 1
                    day["t1_payout"] += o * 100

        # Drop counting (tickets in T-5 but not in T-1)
        t5_combos = set(t.combo for t in t5_tickets)
        t1_combos = set(c for c, _, _, _ in t1_tickets)
        dropped = t5_combos - t1_combos
        drop_stats["dropped_tickets"] += len(dropped)
        day["dropped_tickets"] += len(dropped)
        if t5_tickets and not t1_tickets:
            drop_stats["dropped_races"] += 1

        # Confirmed aggregation
        if conf_tickets_list:
            conf_stats["races"] += 1
            for combo, o, ev, is_hit in conf_tickets_list:
                conf_stats["tickets"] += 1
                conf_stats["cost"] += 100
                if is_hit:
                    conf_stats["wins"] += 1
                    conf_stats["payout"] += o * 100

    # Print aggregate comparison
    def _fmt(s):
        hit = 100 * s["wins"] / s["tickets"] if s["tickets"] else 0
        roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
        pl = s["payout"] - s["cost"]
        return (f"{s['races']:>5}R {s['tickets']:>5}T {s['wins']:>3}W "
                f"{hit:>5.1f}%  {roi:>5.0f}%  {int(pl):>+9,}")

    print()
    print(f"=== T-5 / T-1 / Confirmed 比較 ({args.model_dir.split('/')[-1]}) ===")
    print(f"period: {args.from_date} ~ {args.to_date}")
    print()
    print(f"{'path':<18} {'races':>6} {'tkts':>6} {'wins':>5} {'hit%':>6} {'ROI%':>6} {'P/L':>11}")
    print("-" * 68)
    print(f"{'T-5 (early buy)':<18} {_fmt(t5_stats)}")
    print(f"{'T-1 (drift)':<18} {_fmt(t1_stats)}")
    print(f"{'Confirmed (ref)':<18} {_fmt(conf_stats)}")

    # Drop rate
    print()
    t5_tkt = t5_stats["tickets"]
    dropped = drop_stats["dropped_tickets"]
    drop_rate = 100 * dropped / t5_tkt if t5_tkt else 0
    print(f"T-1 drift drop: {dropped} tickets out of {t5_tkt} ({drop_rate:.1f}%)")
    print(f"  all-dropped races: {drop_stats['dropped_races']} "
          f"(races where T-5 had tickets but T-1 dropped all)")

    # Per-day
    print()
    print("=== Per-day breakdown ===")
    print(f"{'date':<11} {'T-5_tkts':>8} {'T-5_W':>6} {'T-5_PL':>9} "
          f"{'T-1_tkts':>8} {'T-1_W':>6} {'T-1_PL':>9} {'drop':>5}")
    print("-" * 72)
    for date in sorted(per_day.keys()):
        d = per_day[date]
        t5_pl = d["t5_payout"] - d["t5_cost"]
        t1_pl = d["t1_payout"] - d["t1_cost"]
        print(f"{date:<11} {d['t5_tickets']:>8} {d['t5_wins']:>6} {int(t5_pl):>+9,} "
              f"{d['t1_tickets']:>8} {d['t1_wins']:>6} {int(t1_pl):>+9,} "
              f"{d['dropped_tickets']:>5}")


if __name__ == "__main__":
    main()
