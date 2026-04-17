"""Sweep EV threshold on T-5 / T-1 drift / confirmed paths.

Validates whether relaxing the EV threshold actually improves dry-run
performance (T-1 drift path = current runner behavior). Complements
threshold_sweep.py which only uses confirmed odds.

Period is constrained by race_odds_snapshots coverage (2026-04-07 onward).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.registry import get_active_model_dir
from scripts._p2_decision import compute_race_decisions, load_model_and_strategy
from scripts.analyze_t5_t1_drift import (
    _evaluate_tickets,
    _load_confirmed_odds,
    _load_odds_by_timing,
)


def _aggregate(decisions, odds_map, ev_threshold, gap12_th, conc_th, gap23_th,
               rids_with_odds):
    """Return (races, tickets, wins, cost, payout) for tickets where
    re-evaluated EV at this odds source clears ev_threshold."""
    races = tickets = wins = cost = 0
    payout = 0.0
    for d in decisions:
        if d.rid not in rids_with_odds:
            continue
        if d.r1 != 1:
            continue
        if d.gap12 < gap12_th:
            continue
        if d.conc < conc_th:
            continue
        if d.gap23 < gap23_th:
            continue
        tks = _evaluate_tickets(d, odds_map, ev_threshold)
        if not tks:
            continue
        races += 1
        for combo, odds, ev, hit in tks:
            tickets += 1
            cost += 100
            if hit:
                wins += 1
                payout += odds * 100
    return races, tickets, wins, cost, payout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="from_date", default="2026-04-07")
    ap.add_argument("--to", dest="to_date", default="2026-04-17")
    ap.add_argument("--model-dir", default=get_active_model_dir())
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument(
        "--ev-levels",
        default="-0.25,-0.22,-0.20,-0.15,-0.10,-0.06,-0.04,-0.02,0.00",
    )
    args = ap.parse_args()
    ev_levels = [float(x) for x in args.ev_levels.split(",")]

    print(f"Model: {args.model_dir}", file=sys.stderr)
    print(f"Period: {args.from_date} ~ {args.to_date}", file=sys.stderr)

    model, strategy, feature_means = load_model_and_strategy(args.model_dir)
    gap12_th = strategy["gap12_min_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap23_th = strategy["gap23_threshold"]

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    df = df_full[
        (df_full["race_date"] >= args.from_date)
        & (df_full["race_date"] <= args.to_date)
    ]

    t5_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-5")
    t1_odds = _load_odds_by_timing(args.db_path, args.from_date, args.to_date, "T-1")
    conf_odds = _load_confirmed_odds(args.db_path, args.from_date, args.to_date)

    t5_rids = {rid for rid, _ in t5_odds.keys()}
    t1_rids = {rid for rid, _ in t1_odds.keys()}
    conf_rids = {rid for rid, _ in conf_odds.keys()}
    print(f"Coverage: T-5 {len(t5_rids)} / T-1 {len(t1_rids)} / conf {len(conf_rids)} races",
          file=sys.stderr)

    # Compute decisions using T-5 as initial odds map.
    decisions = compute_race_decisions(df, model, feature_means, t5_odds)

    def _print_header(title):
        print(f"\n=== {title} ===")
        print(f"{'EV':>6} {'races':>6} {'tkts':>5} {'wins':>5} "
              f"{'hit%':>6} {'ROI':>6} {'P/L':>9}")

    def _print_row(ev, races, tkts, wins, cost, payout):
        hit = (wins / tkts * 100) if tkts else 0.0
        roi = (payout / cost * 100) if cost else 0.0
        pl = payout - cost
        print(f"{ev:+6.2f} {races:6d} {tkts:5d} {wins:5d} "
              f"{hit:5.1f}% {roi:5.0f}% {pl:+9.0f}")

    # T-5 path
    _print_header("T-5 path (early buy based on T-5 odds)")
    for ev in ev_levels:
        res = _aggregate(decisions, t5_odds, ev, gap12_th, conc_th, gap23_th, t5_rids)
        _print_row(ev, *res)

    # T-1 drift path (current runner behavior)
    _print_header("T-1 drift path (current runner: re-check EV at T-1)")
    for ev in ev_levels:
        res = _aggregate(decisions, t1_odds, ev, gap12_th, conc_th, gap23_th, t1_rids)
        _print_row(ev, *res)

    # Confirmed (reference)
    _print_header("Confirmed odds (reference)")
    for ev in ev_levels:
        res = _aggregate(decisions, conf_odds, ev, gap12_th, conc_th, gap23_th, conf_rids)
        _print_row(ev, *res)


if __name__ == "__main__":
    main()
