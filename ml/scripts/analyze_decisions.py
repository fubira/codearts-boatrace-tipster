"""Analyze a single day's P2 predictions vs filter thresholds.

Classifies each race into buckets (BOUGHT / not_b1 / gap12_low / conc_low /
gap23_low / ev_low) and computes hit rate per bucket. Also lists the closest
borderline skips (races that barely missed the filter cutoff) and shows what
would happen if we relaxed each threshold by a small margin.

Use case: daily review of whether today's picks were reasonable and whether
any near-misses would have hit if thresholds were slightly looser.

Usage:
    cd ml && uv run python -m scripts.analyze_decisions              # latest day in DB
    cd ml && uv run python -m scripts.analyze_decisions --date 2026-04-14
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import defaultdict
from dataclasses import dataclass

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from scripts._p2_decision import (
    RaceDecision,
    compute_race_decisions,
    latest_race_date,
    load_model_and_strategy,
)
from scripts.tune_p2 import _load_trifecta_odds


@dataclass
class ClassifiedRace:
    d: RaceDecision
    reason: str
    margin: float | None
    any_hit: bool


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Race date YYYY-MM-DD (default: latest date in DB)")
    ap.add_argument("--model-dir", default="models/p2_v2")
    ap.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    if args.date is None:
        args.date = latest_race_date(args.db_path)
        print(f"Using latest race date: {args.date}", file=sys.stderr)

    model, strategy, feature_means = load_model_and_strategy(args.model_dir)
    gap23_th = strategy["gap23_threshold"]
    conc_th = strategy["top3_conc_threshold"]
    gap12_th = strategy["gap12_min_threshold"]
    ev_th = strategy["ev_threshold"]

    with contextlib.redirect_stdout(io.StringIO()):
        df_full = build_features_df(args.db_path)
    df = df_full[df_full["race_date"] == args.date]

    odds_map = _load_trifecta_odds(args.db_path)
    decisions = compute_race_decisions(df, model, feature_means, odds_map)
    n_races = len(decisions)

    # Classify each race by the first filter that rejects it. Margin is
    # the positive distance from the threshold — used for borderline
    # ranking. analyze_decisions uses unfiltered `any_hit` (no EV gate),
    # so a race can be marked `ev_low` yet still show HIT in the output.
    classified: list[ClassifiedRace] = []
    for d in decisions:
        any_hit = any(t.hit for t in d.tickets)
        if d.r1 != 1:
            reason, margin = "not_b1", None
        elif d.gap12 < gap12_th:
            reason, margin = "gap12_low", gap12_th - d.gap12
        elif d.conc < conc_th:
            reason, margin = "conc_low", conc_th - d.conc
        elif d.gap23 < gap23_th:
            reason, margin = "gap23_low", gap23_th - d.gap23
        elif d.max_ev < ev_th:
            reason, margin = "ev_low", ev_th - d.max_ev
        else:
            reason, margin = "BOUGHT", 0.0
        classified.append(ClassifiedRace(d=d, reason=reason, margin=margin, any_hit=any_hit))

    # Summary by reason
    print(f"\n=== p2_v2 decisions for {args.date} (total {n_races} races) ===\n")
    buckets: dict[str, list[ClassifiedRace]] = defaultdict(list)
    for c in classified:
        buckets[c.reason].append(c)

    print(f"{'bucket':<15} {'count':>5} {'hit':>4} {'hit%':>6} {'notes':<40}")
    print("-" * 75)
    for reason in ["BOUGHT", "not_b1", "gap12_low", "conc_low", "gap23_low", "ev_low"]:
        items = buckets.get(reason, [])
        if not items:
            continue
        hits = sum(1 for c in items if c.any_hit)
        pct = 100 * hits / len(items) if items else 0
        print(f"{reason:<15} {len(items):>5} {hits:>4} {pct:>5.1f}%")

    # Borderline analysis: how close did BOUGHT skip over near-miss races
    print(f"\n=== Borderline analysis (top 10 closest to BUY threshold) ===\n")
    skipped = [c for c in classified
               if c.reason not in ("BOUGHT", "not_b1") and c.margin is not None]
    skipped.sort(key=lambda c: c.margin)

    print(f"{'rid':>8} {'reason':<12} {'margin':>8} {'gap12':>7} {'conc':>6} {'gap23':>7} {'max_ev':>7} {'hit':>4}")
    for c in skipped[:20]:
        hit_mark = "HIT" if c.any_hit else "-"
        d = c.d
        print(f"  {d.rid:>6} {c.reason:<12} {c.margin:>+8.3f} "
              f"{d.gap12:>7.3f} {d.conc:>6.3f} {d.gap23:>7.3f} {d.max_ev:>+7.3f} {hit_mark:>4}")

    # Bought races detail
    print(f"\n=== Bought races detail ===\n")
    for c in buckets.get("BOUGHT", []):
        d = c.d
        print(f"  rid={d.rid} gap12={d.gap12:.3f} conc={d.conc:.3f} "
              f"gap23={d.gap23:.3f} ev={d.max_ev:+.3f} → {d.hit_combo} "
              f"{'HIT' if c.any_hit else 'MISS'}")
        for t in d.tickets:
            mark = "★" if t.hit else " "
            print(f"    {mark}{t.combo} @ {t.odds:.1f} (EV {t.ev:+.1%})")

    # "If we lowered the bar" analysis.
    # NOTE: Counts races that PASSED this specific filter at the relaxed
    # threshold. Subsequent filters (next in chain: conc → gap23 → ev) may
    # still reject them. This is an UPPER BOUND for "extra races if threshold
    # relaxed", not the actual incremental buy count. To get the actual count,
    # run the full filter chain with the relaxed threshold.
    print(f"\n=== What if we relaxed each threshold by small margin (upper bound) ===\n")
    for delta_name, delta_val in [("gap12 -0.005", 0.005), ("gap12 -0.01", 0.01),
                                    ("conc -0.02", 0.02), ("conc -0.05", 0.05),
                                    ("gap23 -0.01", 0.01), ("gap23 -0.02", 0.02)]:
        reason_filter = delta_name.split()[0] + "_low"
        if reason_filter not in ("gap12_low", "conc_low", "gap23_low"):
            continue
        extra = [c for c in classified
                 if c.reason == reason_filter and c.margin <= delta_val]
        hits = sum(1 for c in extra if c.any_hit)
        print(f"  {delta_name}: +{len(extra)} races ({hits} hits)")


if __name__ == "__main__":
    main()
