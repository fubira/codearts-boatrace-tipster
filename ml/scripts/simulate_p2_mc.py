"""Monte Carlo projection for the P2 strategy from empirical OOS numbers.

Pulls parameters (hit_rate / bets_per_day / tickets_per_bet / payout
distribution) from analyze_model's evaluation of the active model on the
given OOS period, then runs the generic simulate_once engine. Never use an
analysis window that includes training data: use --from 2026-01-01 or later
for p2_v2 (end_date = 2026-01-01).

Usage (module invocation — no PYTHONPATH hacks):
    cd ml && uv run python -m scripts.simulate_p2_mc --from 2026-01-01 --to "$(date +%F)"
    cd ml && uv run python -m scripts.simulate_p2_mc --from 2026-01-01 --to "$(date +%F)" \
        --bankroll 70000 --unit-divisor 150 --bet-cap 30000 --n-sims 10000
"""

import argparse
import contextlib
import io
import math
import sys

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from boatrace_tipster_ml.registry import get_active_model_dir
from scripts.analyze_model import evaluate_period
from scripts.simulate_monte_carlo import simulate_once

FIELD_SIZE = 6


def extract_p2_params(purchases: list, total_days: int) -> dict:
    """Compute hit_rate / bets_per_day / tickets_per_bet / payout lognormal.

    Payout multiplier uses actual odds (median/mean → lognormal params).
    """
    n_bets = len(purchases)
    hit_odds = [p.payout / 100 for p in purchases if p.won]
    n_wins = len(hit_odds)

    hit_rate = n_wins / n_bets if n_bets else 0
    bets_per_day = n_bets / total_days if total_days else 0

    # Avg tickets per bet (P2 uses 1-2 tickets adaptive)
    tickets_per_bet = sum(len(p.tickets) for p in purchases) / n_bets if n_bets else 0

    # Payout lognormal from empirical odds. Fit via log-space moments.
    if hit_odds:
        log_odds = np.log(hit_odds)
        mu = float(np.mean(log_odds))
        sigma = float(np.std(log_odds, ddof=1)) if len(hit_odds) > 1 else 0.5
    else:
        mu, sigma = 2.0, 0.5  # safe defaults

    return {
        "hit_rate": hit_rate,
        "bets_per_day": bets_per_day,
        "tickets_per_bet": tickets_per_bet,
        "payout_mu": mu,
        "payout_sigma": sigma,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "total_days": total_days,
    }


def run_mc(
    params: dict,
    n_sims: int,
    seed: int,
    initial_bankroll: float,
    unit_divisor: int,
    min_unit: int,
    max_unit: int,
    periods: dict[str, int],
) -> None:
    rng = np.random.default_rng(seed)

    median_mult = math.exp(params["payout_mu"])
    mean_mult = math.exp(params["payout_mu"] + params["payout_sigma"] ** 2 / 2)

    print(f"=== P2 Monte Carlo ({n_sims:,} sims, seed={seed}) ===")
    print(f"Bankroll: ¥{initial_bankroll:,.0f}, Unit: BR/{unit_divisor} "
          f"(¥{min_unit:,}~¥{max_unit:,})")
    print(f"Hit: {params['hit_rate']:.1%}, Bets/day: {params['bets_per_day']:.2f}, "
          f"Tickets/bet: {params['tickets_per_bet']:.2f}")
    print(f"Payout (odds×unit): lognormal(mu={params['payout_mu']:.3f}, "
          f"sigma={params['payout_sigma']:.3f}) → median {median_mult:.1f}x, "
          f"mean {mean_mult:.1f}x")
    print(f"Source: {params['n_bets']}R / {params['total_days']}d "
          f"({params['n_wins']} wins)")
    print()

    header = f"{'Period':>8} {'BR med':>10} {'P/L med':>12} {'P25':>12} {'P75':>12} {'DD med':>7} {'Bust':>6}"
    print(header)
    print("-" * len(header))

    for label, n_days in periods.items():
        results = [
            simulate_once(
                n_days, rng,
                hit_rate=params["hit_rate"],
                bets_per_day=params["bets_per_day"],
                tickets_per_bet=params["tickets_per_bet"],
                payout_mu=params["payout_mu"],
                payout_sigma=params["payout_sigma"],
                initial_bankroll=initial_bankroll,
                unit_divisor=unit_divisor,
                min_unit=min_unit,
                max_unit=max_unit,
            )
            for _ in range(n_sims)
        ]

        final_brs = np.array([r["final_bankroll"] for r in results])
        profits = np.array([r["profit"] for r in results])
        dd_pcts = np.array([r["max_dd_pct"] for r in results])
        busts = sum(1 for r in results if r["bust"])

        print(
            f"{label:>8} "
            f"¥{np.median(final_brs):>9,.0f} "
            f"{np.median(profits):>+11,.0f} "
            f"{np.percentile(profits, 25):>+11,.0f} "
            f"{np.percentile(profits, 75):>+11,.0f} "
            f"{np.median(dd_pcts):>6.1%} "
            f"{busts / n_sims:>6.1%}"
        )


def count_active_days(db_path: str, from_date: str, to_date: str) -> int:
    """Count distinct race_dates in the period (races actually held)."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT COUNT(DISTINCT race_date) FROM db.races "
        "WHERE race_date >= ? AND race_date < ?",
        [from_date, to_date],
    ).fetchone()
    conn.close()
    return int(row[0]) if row else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--bankroll", type=float, default=70_000)
    parser.add_argument("--unit-divisor", type=int, default=150)
    parser.add_argument("--min-unit", type=int, default=100)
    parser.add_argument("--bet-cap", type=int, default=30_000,
                        dest="bet_cap")
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=str, default="30,90,180,365",
                        help="Comma-separated day counts")
    args = parser.parse_args()

    model_dir = args.model_dir or get_active_model_dir()
    print(f"Model: {model_dir}/ranking", file=sys.stderr)
    model = load_model(f"{model_dir}/ranking")
    meta = load_model_meta(f"{model_dir}/ranking")

    print("Loading features...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    print("Loading odds...", file=sys.stderr)
    conn = get_connection(args.db_path)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type='3連単'"
    ).fetchall()
    conn.close()
    odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    purchases, _ = evaluate_period(
        model, meta, df, odds, args.from_date, args.to_date
    )
    active_days = count_active_days(args.db_path, args.from_date, args.to_date)
    params = extract_p2_params(purchases, active_days)

    periods = {}
    for d in args.days.split(","):
        d = int(d.strip())
        periods[f"{d}d"] = d

    run_mc(
        params,
        n_sims=args.n_sims,
        seed=args.seed,
        initial_bankroll=args.bankroll,
        unit_divisor=args.unit_divisor,
        min_unit=args.min_unit,
        max_unit=args.bet_cap,
        periods=periods,
    )


if __name__ == "__main__":
    main()
