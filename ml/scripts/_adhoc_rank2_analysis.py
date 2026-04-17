"""One-off: rank2 distribution of bought tickets by EV threshold.

Validates the hypothesis that EV>=0 filter biases toward outer-boat-2nd
tickets (rank2 ∈ {4,5,6}) which hit less often than inner-boat-2nd tickets.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import Counter, defaultdict

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.registry import get_active_model_dir
from scripts._p2_decision import compute_race_decisions, load_model_and_strategy
from scripts.tune_p2 import _load_trifecta_odds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--model-dir", default=get_active_model_dir())
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    ap.add_argument("--ev-levels", default="-0.10,-0.06,-0.04,-0.02,0.00,0.02,0.04")
    args = ap.parse_args()

    ev_levels = [float(x) for x in args.ev_levels.split(",")]

    model, strategy, feature_means = load_model_and_strategy(args.model_dir)
    gap12_th = strategy["gap12_min_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap23_th = strategy["gap23_threshold"]

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    df = df_full[
        (df_full["race_date"] >= args.date_from)
        & (df_full["race_date"] <= args.date_to)
    ]
    print(f"Period: {args.date_from} ~ {args.date_to}", file=sys.stderr)
    print(f"Races loaded: {len(df) // 6}", file=sys.stderr)

    odds_map = _load_trifecta_odds(args.db_path)
    decisions = compute_race_decisions(df, model, feature_means, odds_map)

    # Passed the pre-EV filters (1号艇 top1 / gap12 / conc / gap23)
    passed = [
        d for d in decisions
        if d.r1 == 1
        and d.gap12 >= gap12_th
        and d.conc >= conc_th
        and d.gap23 >= gap23_th
    ]
    print(f"Pre-EV filter passed: {len(passed)} races", file=sys.stderr)

    # For each EV level, collect all tickets that would be bought
    print(f"\n=== rank2 distribution of bought TICKETS by EV level ===")
    print(f"{'EV':>6} {'tkts':>5} {'hits':>5} {'hit%':>6} | "
          + " ".join(f"r2={b}" for b in range(2, 7))
          + " | " + " ".join(f"hit{b}" for b in range(2, 7)))
    for ev_th in ev_levels:
        buys = []  # (rank2_boat, hit)
        for d in passed:
            for t in d.tickets:
                if t.ev < ev_th:
                    continue
                # rank2 in combo string (e.g. "1-4-3" → rank2_boat=4)
                r2_boat = int(t.combo.split("-")[1])
                buys.append((r2_boat, t.hit))

        total = len(buys)
        hits = sum(1 for _, h in buys if h)
        hit_rate = (hits / total * 100) if total else 0.0

        r2_counts = Counter(b for b, _ in buys)
        r2_hits = defaultdict(int)
        for b, h in buys:
            if h:
                r2_hits[b] += 1

        dist = " ".join(f"{r2_counts.get(b, 0):5d}" for b in range(2, 7))
        hit_by = " ".join(f"{r2_hits.get(b, 0):4d}" for b in range(2, 7))
        print(f"{ev_th:+6.2f} {total:5d} {hits:5d} {hit_rate:5.1f}% | {dist} | {hit_by}")

    # Hit rate by rank2 position (ignoring EV filter) for passed races
    print(f"\n=== Hit rate by rank2 boat (all passed races, unfiltered) ===")
    all_tkts = [(int(t.combo.split("-")[1]), t.hit)
                for d in passed for t in d.tickets]
    print(f"{'r2':>3} {'tkts':>5} {'hits':>5} {'hit%':>6}")
    for r2 in range(2, 7):
        tkts = [h for b, h in all_tkts if b == r2]
        n = len(tkts)
        w = sum(tkts)
        rate = (w / n * 100) if n else 0.0
        print(f"{r2:3d} {n:5d} {w:5d} {rate:5.1f}%")


if __name__ == "__main__":
    main()
