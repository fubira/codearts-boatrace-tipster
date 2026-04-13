"""Monte Carlo projection for the P2 strategy from empirical OOS numbers.

Pulls parameters (hit_rate / bets_per_day / tickets_per_bet / payout
distribution) from analyze_model's evaluation of the active model on the
given OOS period, then runs a P2-specific simulate engine.

The engine differs from the legacy simulate_monte_carlo (X-allflow, 20
tickets/bet) in two ways:
- `tickets_per_bet` is fractional (~1.3 for P2 adaptive 1-2 tickets). We
  sample per-bet ticket count via Bernoulli on the fractional part instead
  of rounding to int, so cost scaling is accurate
- Payout scales with `unit` (not `n_tickets * unit`) because only one of
  the 1-2 P2 tickets can hit a given outcome

Never use an analysis window that includes training data: use
`--from 2026-01-01` or later for p2_v2 (end_date = 2026-01-01).

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

FIELD_SIZE = 6


def simulate_p2_once(
    n_days: int,
    rng: np.random.Generator,
    *,
    hit_rate: float,
    bets_per_day: float,
    tickets_per_bet: float,
    payout_mu: float,
    payout_sigma: float,
    initial_bankroll: float,
    unit_divisor: int,
    min_unit: int,
    max_unit: int,
) -> dict:
    """Simulate one P2 strategy trajectory over n_days.

    Differs from simulate_monte_carlo.simulate_once by using Bernoulli
    sampling for fractional tickets_per_bet. For example, with 1.31
    tickets/bet, each bet buys 2 tickets with P=0.31 and 1 ticket with
    P=0.69. This preserves the true average cost (1.31 * unit) instead
    of rounding to 1 (which would undercount cost by ~31%).

    Payout when hit scales with `unit` regardless of ticket count because
    only one of the 1-2 P2 tickets can hit a given race outcome.
    """
    bankroll = float(initial_bankroll)
    peak = bankroll
    max_dd_yen = 0.0
    max_dd_pct = 0.0
    consec_loss = 0
    max_consec_loss = 0
    total_wagered = 0.0
    total_payout = 0.0
    win_days = 0
    all_loss_days = 0
    total_bets = 0
    active_days = 0  # days with at least one bet placed

    # Bernoulli params for stochastic ticket count
    base_tickets = int(math.floor(tickets_per_bet))
    extra_prob = tickets_per_bet - base_tickets  # P(one extra ticket)

    for _day in range(n_days):
        n_bets = rng.poisson(bets_per_day)
        if n_bets == 0:
            continue
        active_days += 1
        total_bets += n_bets

        raw_unit = bankroll / unit_divisor
        unit = max(min_unit, min(max_unit, int(raw_unit / 100) * 100))

        day_cost = 0.0
        day_payout = 0.0
        day_wins = 0

        for _ in range(n_bets):
            n_tickets = base_tickets + (1 if rng.random() < extra_prob else 0)
            if n_tickets == 0:
                continue
            cost = n_tickets * unit
            day_cost += cost

            if rng.random() < hit_rate:
                odds_mult = rng.lognormal(payout_mu, payout_sigma)
                # Only one ticket can hit per race outcome; payout scales
                # with `unit`, not with `n_tickets * unit`.
                day_payout += odds_mult * unit
                day_wins += 1

        bankroll += day_payout - day_cost
        total_wagered += day_cost
        total_payout += day_payout

        if day_payout > day_cost:
            win_days += 1
            consec_loss = 0
        else:
            consec_loss += 1
            max_consec_loss = max(max_consec_loss, consec_loss)
            if day_wins == 0:
                all_loss_days += 1

        if bankroll > peak:
            peak = bankroll
        dd_yen = peak - bankroll
        dd_pct = dd_yen / peak if peak > 0 else 0
        max_dd_yen = max(max_dd_yen, dd_yen)
        max_dd_pct = max(max_dd_pct, dd_pct)

        # Bust: can't afford max tickets at min unit
        min_cost = min_unit * math.ceil(tickets_per_bet)
        if bankroll < min_cost:
            return {
                "final_bankroll": bankroll,
                "profit": bankroll - initial_bankroll,
                "max_dd_yen": max_dd_yen,
                "max_dd_pct": max_dd_pct,
                "max_consec_loss": max_consec_loss,
                "win_day_rate": win_days / n_days,
                "all_loss_day_rate": all_loss_days / n_days,
                "avg_bets_per_day": total_bets / n_days,
                "total_bets": total_bets,
                "bust": True,
            }

    return {
        "final_bankroll": bankroll,
        "profit": bankroll - initial_bankroll,
        "max_dd_yen": max_dd_yen,
        "max_dd_pct": max_dd_pct,
        "max_consec_loss": max_consec_loss,
        "win_day_rate": win_days / n_days,
        "all_loss_day_rate": all_loss_days / n_days,
        "avg_bets_per_day": total_bets / n_days,
        "total_bets": total_bets,
        "bust": False,
    }


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

    # Table 1: P/L distribution (percentiles + mean)
    print("## P/L distribution")
    header1 = (
        f"{'Period':>8} "
        f"{'P5':>11} {'P25':>11} {'P50':>11} {'P75':>11} {'P95':>11} "
        f"{'mean':>11}"
    )
    print(header1)
    print("-" * len(header1))

    all_results: dict[str, list[dict]] = {}
    for label, n_days in periods.items():
        results = [
            simulate_p2_once(
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
        all_results[label] = results
        profits = np.array([r["profit"] for r in results])

        print(
            f"{label:>8} "
            f"{np.percentile(profits, 5):>+10,.0f} "
            f"{np.percentile(profits, 25):>+10,.0f} "
            f"{np.percentile(profits, 50):>+10,.0f} "
            f"{np.percentile(profits, 75):>+10,.0f} "
            f"{np.percentile(profits, 95):>+10,.0f} "
            f"{np.mean(profits):>+10,.0f}"
        )

    print()

    # Table 2: Downside metrics
    print("## Downside")
    header2 = (
        f"{'Period':>8} {'Loss%':>7} {'≤0%':>7} {'Worst':>13} "
        f"{'DD med':>7} {'DD P95':>7} {'DD max':>7} {'Bust':>6}"
    )
    print(header2)
    print("-" * len(header2))
    for label, results in all_results.items():
        profits = np.array([r["profit"] for r in results])
        dd_pcts = np.array([r["max_dd_pct"] for r in results])
        busts = sum(1 for r in results if r["bust"])
        loss_rate = float(np.mean(profits < 0))
        breakeven_or_loss = float(np.mean(profits <= 0))
        print(
            f"{label:>8} "
            f"{loss_rate:>6.1%} "
            f"{breakeven_or_loss:>6.1%} "
            f"{profits.min():>+12,.0f} "
            f"{np.median(dd_pcts):>6.1%} "
            f"{np.percentile(dd_pcts, 95):>6.1%} "
            f"{dd_pcts.max():>6.1%} "
            f"{busts / n_sims:>6.1%}"
        )

    print()

    # Table 3: Upside / growth
    print("## Growth (final bankroll)")
    header3 = f"{'Period':>8} {'BR P5':>12} {'BR P50':>12} {'BR P95':>12} {'Best':>14}"
    print(header3)
    print("-" * len(header3))
    for label, results in all_results.items():
        final_brs = np.array([r["final_bankroll"] for r in results])
        print(
            f"{label:>8} "
            f"¥{np.percentile(final_brs, 5):>10,.0f} "
            f"¥{np.percentile(final_brs, 50):>10,.0f} "
            f"¥{np.percentile(final_brs, 95):>10,.0f} "
            f"¥{final_brs.max():>12,.0f}"
        )

    print()

    # Table 4: Activity (bets / win days)
    print("## Activity")
    header4 = (
        f"{'Period':>8} {'bets/day':>10} {'total bets':>12} "
        f"{'勝ち日':>8} {'全敗日':>8} {'max 連敗':>10}"
    )
    print(header4)
    print("-" * len(header4))
    for label, results in all_results.items():
        avg_bpd = np.mean([r["avg_bets_per_day"] for r in results])
        total_b = np.mean([r["total_bets"] for r in results])
        win_day = np.mean([r["win_day_rate"] for r in results])
        loss_day = np.mean([r["all_loss_day_rate"] for r in results])
        max_streak = np.mean([r["max_consec_loss"] for r in results])
        print(
            f"{label:>8} "
            f"{avg_bpd:>9.2f} "
            f"{total_b:>11,.0f} "
            f"{win_day:>7.1%} "
            f"{loss_day:>7.1%} "
            f"{max_streak:>9.1f}"
        )

    print()
    print(
        f"Note: profits are profit-from-initial (not total turnover). "
        f"initial bankroll = ¥{initial_bankroll:,.0f}"
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
    if not purchases:
        print(
            f"ERROR: No purchases found in {args.from_date} ~ {args.to_date}. "
            f"Check strategy filters and period.",
            file=sys.stderr,
        )
        sys.exit(1)

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
