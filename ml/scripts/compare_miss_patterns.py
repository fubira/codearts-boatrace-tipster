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
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.registry import get_active_model_dir
from scripts._p2_decision import (
    RaceDecision,
    compute_race_decisions,
    latest_race_date,
    load_model_and_strategy,
)
from scripts.tune_p2 import _load_trifecta_odds


@dataclass
class BoughtRace:
    d: RaceDecision
    any_hit: bool
    miss_reason: str  # HIT / boat1_DSQ / boat1_<N>th / 2_3_order_wrong


def analyze_period(df_full, odds_map, model, feature_means, strategy,
                   from_date, to_date) -> list[BoughtRace]:
    """Apply P2 filter chain to a date window and return bought races."""
    gap23_th = strategy["gap23_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap12_th = strategy["gap12_min_threshold"]
    ev_th = strategy["ev_threshold"]

    df = df_full[(df_full["race_date"] >= from_date) & (df_full["race_date"] <= to_date)]
    decisions = compute_race_decisions(df, model, feature_means, odds_map)

    bought: list[BoughtRace] = []
    for d in decisions:
        if d.r1 != 1 or d.gap12 < gap12_th or d.conc < conc_th or d.gap23 < gap23_th:
            continue
        # EV-pass tickets only — a race with both tickets below EV threshold
        # is skipped entirely (no purchase).
        kept = [t for t in d.tickets if t.ev >= ev_th]
        if not kept:
            continue

        any_hit = any(t.hit for t in kept)

        if any_hit:
            miss_reason = "HIT"
        elif d.boat1_finish == 0:
            miss_reason = "boat1_DSQ"
        elif d.boat1_finish != 1:
            miss_reason = f"boat1_{d.boat1_finish}th"
        else:
            # Boat 1 won but 2/3 ordering wrong
            miss_reason = "2_3_order_wrong"

        bought.append(BoughtRace(d=d, any_hit=any_hit, miss_reason=miss_reason))
    return bought


def print_summary(label: str, bought: list[BoughtRace]):
    hits = [b for b in bought if b.any_hit]
    misses = [b for b in bought if not b.any_hit]
    hit_pct = 100 * len(hits) / len(bought) if bought else 0
    print(f"\n=== {label} ===")
    print(f"Bought: {len(bought)} races, Hit: {len(hits)} ({hit_pct:.1f}%)")
    print()

    # Miss reason breakdown
    print("Miss breakdown:")
    reasons = Counter(b.miss_reason for b in bought)
    for reason, count in reasons.most_common():
        pct = 100 * count / len(bought) if bought else 0
        bar = "#" * int(pct / 3)
        print(f"  {reason:<22}: {count:>4} ({pct:>4.1f}%)  {bar}")

    # Filter value distributions (means)
    if bought:
        print()
        print("Filter value averages:")
        for key in ["p1", "gap12", "conc", "gap23"]:
            vals = [getattr(b.d, key) for b in bought]
            print(f"  {key:<7}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f}")

    # For misses only: boat 1 finish distribution
    if misses:
        print()
        print("Miss: boat 1 actual finish position:")
        finish_dist = Counter(m.d.boat1_finish for m in misses)
        for pos in sorted(finish_dist.keys()):
            count = finish_dist[pos]
            pct = 100 * count / len(misses)
            pos_label = f"{pos}位" if pos else "失格"
            print(f"  {pos_label}: {count:>4} ({pct:>4.1f}%)")


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
    ap.add_argument("--model-dir", default=get_active_model_dir())
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    # Resolve date windows. Explicit --recent-from / --recent-to etc.
    # override the default "last N days ending at latest DB date" rule.
    end = pd.Timestamp(args.to_date) if args.to_date else pd.Timestamp(latest_race_date(args.db_path))
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

    model, strategy, feature_means = load_model_and_strategy(args.model_dir)

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

    def period_stats(bought: list[BoughtRace]) -> dict:
        if not bought:
            return {}
        hits = sum(1 for b in bought if b.any_hit)
        n = len(bought)
        misses = [b for b in bought if not b.any_hit]
        m_boat1_not_1st = sum(1 for m in misses if m.d.boat1_finish != 1)
        m_ordering = sum(1 for m in misses if m.d.boat1_finish == 1)
        return {
            "n": n,
            "hits": hits,
            "hit_pct": hits / n,
            "misses": len(misses),
            "boat1_not_1st_pct": m_boat1_not_1st / len(misses) if misses else 0,
            "ordering_wrong_pct": m_ordering / len(misses) if misses else 0,
            "p1_mean": np.mean([b.d.p1 for b in bought]),
            "gap12_mean": np.mean([b.d.gap12 for b in bought]),
            "conc_mean": np.mean([b.d.conc for b in bought]),
            "gap23_mean": np.mean([b.d.gap23 for b in bought]),
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
