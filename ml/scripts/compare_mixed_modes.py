"""Compare all-flow vs mixed modes with different exacta ratios.

Runs WF-CV once, then extracts params for each mode and runs MC.

Usage:
    uv run --directory ml python -m scripts.compare_mixed_modes
    uv run --directory ml python -m scripts.compare_mixed_modes --n-sims 10000
"""

import argparse
import math
import sys
import time

from scripts.simulate_monte_carlo import (
    _extract_params,
    collect_all_candidates,
    run_projection,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b1-threshold", type=float, default=0.42)
    parser.add_argument("--ev-threshold", type=float, default=0.30)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--bankroll", type=float, default=70000)
    parser.add_argument("--unit-divisor", type=int, default=800)
    parser.add_argument("--min-unit", type=int, default=100)
    parser.add_argument("--bet-cap", type=int, default=2000)
    args = parser.parse_args()

    periods = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365}

    print("Collecting all candidates from WF-CV...", file=sys.stderr)
    t0 = time.time()
    all_results = collect_all_candidates(args.n_folds)
    print(f"Done in {time.time() - t0:.1f}s ({len(all_results)} candidates)\n",
          file=sys.stderr)

    modes = [
        ("3連単のみ (20pt)", {"all_flow": True, "mixed": False, "exacta_ratio": 0}),
        ("混合 ratio=0.5 (22.5pt)", {"all_flow": True, "mixed": True, "exacta_ratio": 0.5}),
        ("混合 ratio=1.0 (25pt)", {"all_flow": True, "mixed": True, "exacta_ratio": 1.0}),
        ("混合 ratio=2.0 (30pt)", {"all_flow": True, "mixed": True, "exacta_ratio": 2.0}),
        ("混合 ratio=3.0 (35pt)", {"all_flow": True, "mixed": True, "exacta_ratio": 3.0}),
    ]

    b1_thr = args.b1_threshold
    ev_thr = args.ev_threshold

    for label, mode_kwargs in modes:
        params = _extract_params(
            all_results, b1_thr, ev_thr, **mode_kwargs,
        )
        n = sum(1 for r in all_results
                if r["b1_prob"] < b1_thr and r["ev"] >= ev_thr)
        print(f"\n{'='*70}")
        print(f"[{label}] b1<{b1_thr} ev>+{ev_thr:.0%} — {n} bets, "
              f"{params['bets_per_day']:.2f}/day, hit={params['hit_rate']:.1%}")
        print(f"  payout: median={math.exp(params['payout_mu']):.0f}x, "
              f"tickets={params['tickets_per_bet']:.1f}")
        print(f"{'='*70}")
        run_projection(
            n_sims=args.n_sims, periods=periods, params=params,
            bankroll=args.bankroll, unit_divisor=args.unit_divisor,
            min_unit=args.min_unit, max_unit=args.bet_cap, seed=args.seed,
        )


if __name__ == "__main__":
    main()
