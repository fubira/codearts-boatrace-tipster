"""EV threshold sensitivity analysis for a P2 model.

Evaluates how a P2 model's P/L, ROI, and rank2 composition respond to
different EV thresholds across three odds sources (T-5 / T-1 drift /
confirmed). Used for:

1. Tuning the EV baseline threshold for a new model (the optimum differs
   per model; p2_v3 best is EV=-0.25 but future models may diverge).
2. Validating the chosen threshold does not degenerate into outer-2nd
   bias (rank2 ∈ {4, 5, 6} hits less than rank2 ∈ {2, 3}).
3. Understanding what concrete bets a threshold relaxation adds
   (per-race diff between two EV levels).

Period is constrained by race_odds_snapshots coverage (2026-04-07+ has
T-5 / T-1 snapshots; confirmed odds cover the full history).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import Counter, defaultdict

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.registry import get_active_model_dir
from scripts._p2_decision import compute_race_decisions, load_model_and_strategy
from scripts.analyze_t5_t1_drift import (
    _evaluate_tickets,
    _load_confirmed_odds,
    _load_odds_by_timing,
)

PATH_LABELS = {
    "t5": "T-5 (early buy)",
    "t1": "T-1 drift (runner)",
    "confirmed": "confirmed (reference)",
}
DEFAULT_EV_LEVELS = "-0.25,-0.22,-0.20,-0.15,-0.10,-0.06,-0.04,-0.02,0.00"


def _load_race_info(db_path, from_date, to_date):
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT r.id, r.race_date, s.name, r.race_number
            FROM db.races r JOIN db.stadiums s ON r.stadium_id = s.id
            WHERE r.race_date BETWEEN ? AND ?
            """,
            (from_date, to_date),
        ).fetchall()
    finally:
        conn.close()
    return {int(r[0]): (str(r[1]), r[2], int(r[3])) for r in rows}


def _filter_passed(decisions, odds_rids, gap12_th, conc_th, gap23_th):
    return [
        d for d in decisions
        if d.rid in odds_rids
        and d.r1 == 1
        and d.gap12 >= gap12_th
        and d.conc >= conc_th
        and d.gap23 >= gap23_th
    ]


def _aggregate(passed, odds_map, ev_threshold):
    races = tickets = wins = cost = 0
    payout = 0.0
    for d in passed:
        tks = _evaluate_tickets(d, odds_map, ev_threshold)
        if not tks:
            continue
        races += 1
        for _combo, odds, _ev, hit in tks:
            tickets += 1
            cost += 100
            if hit:
                wins += 1
                payout += odds * 100
    return races, tickets, wins, cost, payout


def _rank2_counts(passed, odds_map, ev_threshold):
    counts: Counter[int] = Counter()
    hits: Counter[int] = Counter()
    for d in passed:
        for combo, _odds, _ev, hit in _evaluate_tickets(d, odds_map, ev_threshold):
            r2 = int(combo.split("-")[1])
            counts[r2] += 1
            if hit:
                hits[r2] += 1
    return counts, hits


def _print_sweep(path_key, passed, odds_map, ev_levels, rank2):
    print(f"\n=== EV sweep: {PATH_LABELS[path_key]} ===")
    header = (f"{'EV':>6} {'races':>6} {'tkts':>5} {'wins':>5} "
              f"{'hit%':>6} {'ROI':>6} {'P/L':>9}")
    if rank2:
        header += " | " + " ".join(f"{f'r2={b}':>5}" for b in range(2, 7))
        header += " | " + " ".join(f"{f'h{b}':>4}" for b in range(2, 7))
    print(header)
    for ev in ev_levels:
        races, tkts, wins, cost, payout = _aggregate(passed, odds_map, ev)
        hit = (wins / tkts * 100) if tkts else 0.0
        roi = (payout / cost * 100) if cost else 0.0
        pl = payout - cost
        row = (f"{ev:+6.2f} {races:6d} {tkts:5d} {wins:5d} "
               f"{hit:5.1f}% {roi:5.0f}% {pl:+9.0f}")
        if rank2:
            c, h = _rank2_counts(passed, odds_map, ev)
            row += " | " + " ".join(f"{c.get(b, 0):5d}" for b in range(2, 7))
            row += " | " + " ".join(f"{h.get(b, 0):4d}" for b in range(2, 7))
        print(row)


def _print_diff(passed, odds_map, ev_a, ev_b, race_info, path_label):
    # Always show "bets that the more-relaxed threshold adds over the
    # stricter one" regardless of BASE,NEW order: swap internally so the
    # listing semantics ("ADDED by lower EV") are stable.
    ev_base, ev_new = (ev_b, ev_a) if ev_a < ev_b else (ev_a, ev_b)
    print(f"\n=== bets ADDED by EV={ev_new:+.2f} over EV={ev_base:+.2f} "
          f"({path_label}) ===")
    added_by_day: dict[str, list] = defaultdict(list)
    added_hits = 0
    added_pl = 0.0
    for d in passed:
        base_combos = {t[0] for t in _evaluate_tickets(d, odds_map, ev_base)}
        new_tks = _evaluate_tickets(d, odds_map, ev_new)
        added = [t for t in new_tks if t[0] not in base_combos]
        if not added:
            continue
        date, stadium, race_no = race_info.get(d.rid, ("?", "?", 0))
        for combo, odds, ev, hit in added:
            pl = odds * 100 - 100 if hit else -100
            added_by_day[date].append(
                (d.rid, stadium, race_no, combo, odds, ev, hit, pl, d.hit_combo)
            )
            if hit:
                added_hits += 1
            added_pl += pl

    print(f"{'日付':>10} {'場':>4} {'R':>3} {'買目':>7} {'odds':>7} "
          f"{'EV':>7} {'hit':>5} {'P/L':>7} {'実績':>7}")
    for date in sorted(added_by_day.keys()):
        for row in sorted(added_by_day[date], key=lambda r: (r[1], r[2], r[3])):
            _rid, stadium, race_no, combo, odds, ev, hit, pl, actual = row
            hit_str = "WIN" if hit else "-"
            print(f"{date:>10} {stadium:>4} {race_no:>3} {combo:>7} "
                  f"{odds:7.1f} {ev:+7.2f} {hit_str:>5} "
                  f"{pl:+7.0f} {actual:>7}")
    total = sum(len(v) for v in added_by_day.values())
    print(f"\nTotal added tickets: {total} / hits: {added_hits} / "
          f"added P/L: {added_pl:+.0f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--model-dir", default=get_active_model_dir())
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument(
        "--paths", default="t5,t1,confirmed",
        help="comma-separated subset of t5,t1,confirmed (default: all)",
    )
    ap.add_argument("--ev-levels", default=DEFAULT_EV_LEVELS)
    ap.add_argument(
        "--rank2-breakdown", action="store_true",
        help="append rank2 ∈ {2..6} ticket counts and hits per sweep row",
    )
    ap.add_argument(
        "--diff-ev", default=None,
        help="BASE,NEW (e.g. 0.00,-0.25). Lists bets the more-relaxed "
             "threshold adds over the stricter one.",
    )
    ap.add_argument(
        "--diff-path", default="t1", choices=list(PATH_LABELS.keys()),
        help="path used for --diff-ev listing (default: t1)",
    )
    args = ap.parse_args()

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    for p in paths:
        if p not in PATH_LABELS:
            ap.error(f"unknown path: {p}")
    ev_levels = [float(x) for x in args.ev_levels.split(",")]

    diff_ev = None
    if args.diff_ev:
        try:
            vals = [float(x) for x in args.diff_ev.split(",")]
        except ValueError:
            ap.error("--diff-ev must be 'BASE,NEW' (two floats)")
        if len(vals) != 2:
            ap.error("--diff-ev must have exactly 2 values")
        diff_ev = (vals[0], vals[1])

    print(f"Model: {args.model_dir}", file=sys.stderr)
    print(f"Period: {args.from_date} ~ {args.to_date}", file=sys.stderr)

    model, strategy, feature_means = load_model_and_strategy(args.model_dir)
    gap12_th = strategy["gap12_min_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap23_th = strategy["gap23_threshold"]
    print(f"Filters: gap12≥{gap12_th} conc≥{conc_th} gap23≥{gap23_th}",
          file=sys.stderr)

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    df = df_full[
        (df_full["race_date"] >= args.from_date)
        & (df_full["race_date"] <= args.to_date)
    ]

    # T-5 is always loaded so compute_race_decisions enumerates every P2
    # ticket once (ev re-evaluation against other odds sources reuses the
    # stored model_prob). Other paths are loaded only when referenced.
    needed = set(paths) | ({args.diff_path} if diff_ev else set()) | {"t5"}
    odds_by_path: dict[str, dict] = {}
    for p in needed:
        if p == "confirmed":
            odds_by_path[p] = _load_confirmed_odds(
                args.db_path, args.from_date, args.to_date
            )
        else:
            timing = "T-5" if p == "t5" else "T-1"
            odds_by_path[p] = _load_odds_by_timing(
                args.db_path, args.from_date, args.to_date, timing
            )

    coverage = {k: {rid for rid, _ in v.keys()} for k, v in odds_by_path.items()}
    cov_str = " / ".join(f"{k}:{len(coverage[k])}" for k in sorted(needed))
    print(f"Coverage: {cov_str} races", file=sys.stderr)

    decisions = compute_race_decisions(df, model, feature_means, odds_by_path["t5"])

    for p in paths:
        passed = _filter_passed(
            decisions, coverage[p], gap12_th, conc_th, gap23_th,
        )
        print(f"-- {PATH_LABELS[p]}: {len(passed)} races pass filters --",
              file=sys.stderr)
        _print_sweep(p, passed, odds_by_path[p], ev_levels, args.rank2_breakdown)

    if diff_ev:
        dp = args.diff_path
        race_info = _load_race_info(args.db_path, args.from_date, args.to_date)
        passed = _filter_passed(
            decisions, coverage[dp], gap12_th, conc_th, gap23_th,
        )
        _print_diff(
            passed, odds_by_path[dp], diff_ev[0], diff_ev[1],
            race_info, PATH_LABELS[dp],
        )


if __name__ == "__main__":
    main()
