"""Monte Carlo projection of trifecta strategy with dynamic bankroll management.

Simulates revenue projection using empirical per-bet statistics from WF-CV.
Unit sizing: bankroll / divisor, rounded down to ¥100, clamped to [MIN, MAX].

Usage:
    # Default parameters (from WF-CV backtest, EV>=33%)
    uv run --directory ml python -m scripts.simulate_monte_carlo

    # Custom bankroll and periods
    uv run --directory ml python -m scripts.simulate_monte_carlo --bankroll 100000 --bet-cap 3000

    # Quick run with fewer simulations
    uv run --directory ml python -m scripts.simulate_monte_carlo --n-sims 1000

    # Collect fresh empirical parameters from backtest
    uv run --directory ml python -m scripts.simulate_monte_carlo --from-backtest --ev-threshold 0.33
"""

import argparse
import contextlib
import io
import sys
import time
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Default empirical parameters (WF-CV 4fold OOS, EV>=33%, 2025-08~2026-04)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "hit_rate": 0.164,
    "bets_per_day": 1.76,
    "tickets_per_bet": 11.1,
    # Lognormal payout distribution (in ticket units)
    # Fitted to empirical: median=116, P90=407
    "payout_mu": 4.75,     # ln(116)
    "payout_sigma": 1.05,
}


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------


def simulate_once(
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
    n_tickets = round(tickets_per_bet)

    for _day in range(n_days):
        n_bets = rng.poisson(bets_per_day)
        if n_bets == 0:
            continue

        # Dynamic unit sizing
        raw_unit = bankroll / unit_divisor
        unit = max(min_unit, min(max_unit, int(raw_unit / 100) * 100))

        day_cost = 0.0
        day_payout = 0.0
        day_wins = 0

        for _ in range(n_bets):
            cost = n_tickets * unit
            day_cost += cost

            if rng.random() < hit_rate:
                payout_tickets = rng.lognormal(payout_mu, payout_sigma)
                day_payout += payout_tickets * unit
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

        # Bust: can't afford a single bet
        if bankroll < min_unit * tickets_per_bet:
            return {
                "final_bankroll": bankroll,
                "profit": bankroll - initial_bankroll,
                "max_dd_yen": max_dd_yen,
                "max_dd_pct": max_dd_pct,
                "max_consec_loss": max_consec_loss,
                "win_day_rate": win_days / n_days,
                "all_loss_day_rate": all_loss_days / n_days,
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
        "bust": False,
    }


def run_projection(
    n_sims: int,
    periods: dict[str, int],
    params: dict,
    bankroll: float,
    unit_divisor: int,
    min_unit: int,
    max_unit: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    print(f"Monte Carlo Projection ({n_sims:,} sims, seed={seed})")
    print(f"Bankroll: ¥{bankroll:,.0f}, Unit: BR/{unit_divisor} (¥{min_unit}~¥{max_unit:,})")
    print(f"Hit: {params['hit_rate']:.1%}, Bets/day: {params['bets_per_day']:.2f}, "
          f"Tickets/bet: {params['tickets_per_bet']:.1f}")
    print(f"Payout: lognormal(mu={params['payout_mu']:.2f}, sigma={params['payout_sigma']:.2f}) "
          f"→ median {np.exp(params['payout_mu']):.0f}x tickets")
    print()

    for label, n_days in periods.items():
        results = [
            simulate_once(
                n_days, rng,
                hit_rate=params["hit_rate"],
                bets_per_day=params["bets_per_day"],
                tickets_per_bet=params["tickets_per_bet"],
                payout_mu=params["payout_mu"],
                payout_sigma=params["payout_sigma"],
                initial_bankroll=bankroll,
                unit_divisor=unit_divisor,
                min_unit=min_unit,
                max_unit=max_unit,
            )
            for _ in range(n_sims)
        ]

        profits = np.array([r["profit"] for r in results])
        finals = np.array([r["final_bankroll"] for r in results])
        max_dds = np.array([r["max_dd_yen"] for r in results])
        max_dd_pcts = np.array([r["max_dd_pct"] for r in results])
        consec = np.array([r["max_consec_loss"] for r in results])
        win_rates = np.array([r["win_day_rate"] for r in results])
        all_loss_rates = np.array([r["all_loss_day_rate"] for r in results])
        bust_rate = np.mean([r["bust"] for r in results])

        print(f"=== {label} ({n_days}日) ===")
        print(f"  収益     中央値: ¥{np.median(profits):+,.0f}  "
              f"平均: ¥{np.mean(profits):+,.0f}")
        print(f"  収益分布 P10: ¥{np.percentile(profits, 10):+,.0f}  "
              f"P25: ¥{np.percentile(profits, 25):+,.0f}  "
              f"P75: ¥{np.percentile(profits, 75):+,.0f}  "
              f"P90: ¥{np.percentile(profits, 90):+,.0f}")
        print(f"  最終BR   中央値: ¥{np.median(finals):,.0f}  "
              f"P10: ¥{np.percentile(finals, 10):,.0f}  "
              f"P90: ¥{np.percentile(finals, 90):,.0f}")
        print(f"  MaxDD    中央値: ¥{np.median(max_dds):,.0f}  "
              f"P90: ¥{np.percentile(max_dds, 90):,.0f}  "
              f"P99: ¥{np.percentile(max_dds, 99):,.0f}")
        print(f"  MaxDD%   中央値: {np.median(max_dd_pcts):.0%}  "
              f"P90: {np.percentile(max_dd_pcts, 90):.0%}")
        print(f"  連敗日数 中央値: {np.median(consec):.0f}  "
              f"P90: {np.percentile(consec, 90):.0f}  "
              f"最大: {np.max(consec)}")
        print(f"  勝ち日率 中央値: {np.median(win_rates):.0%}  "
              f"全敗日率: {np.median(all_loss_rates):.0%}")
        print(f"  破産確率: {bust_rate:.1%}  "
              f"黒字確率: {np.mean(profits > 0):.0%}")
        print()


# ---------------------------------------------------------------------------
# Collect empirical parameters from backtest
# ---------------------------------------------------------------------------


def collect_from_backtest(
    ev_threshold: float,
    b1_threshold: float,
    n_folds: int,
) -> dict:
    """Run WF-CV backtest and extract empirical distribution parameters."""
    from boatrace_tipster_ml.db import DEFAULT_DB_PATH
    from boatrace_tipster_ml.feature_config import prepare_feature_matrix
    from boatrace_tipster_ml.model import walk_forward_splits
    from scripts.backtest_trifecta import evaluate_period, load_data, train_models

    print("Collecting empirical parameters from backtest...", file=sys.stderr)
    t0 = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        df, trifecta_odds, tri_win_prob, finish_map, race_date_map = load_data(
            DEFAULT_DB_PATH
        )

    X_rank, y_rank, meta_rank = prepare_feature_matrix(df)
    folds = walk_forward_splits(X_rank, y_rank, meta_rank, n_folds=n_folds, fold_months=2)

    all_results = []
    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]
        train_fold = df[df["race_date"] < test_from]
        test_fold = df[(df["race_date"] >= test_from) & (df["race_date"] < test_to)]
        dates = sorted(train_fold["race_date"].unique())
        val_start = dates[max(0, len(dates) - 60)]

        print(f"  Fold {i+1}/{len(folds)}: {test_from} ~ {test_to}", file=sys.stderr)
        with contextlib.redirect_stdout(io.StringIO()):
            b1_model, rank_model = train_models(
                train_fold[train_fold["race_date"] < val_start],
                train_fold[train_fold["race_date"] >= val_start],
            )

        results = evaluate_period(
            b1_model, rank_model, test_fold,
            trifecta_odds, tri_win_prob, finish_map, race_date_map,
            b1_threshold=b1_threshold, ev_threshold=ev_threshold,
        )
        all_results.extend(results)

    # Extract parameters
    n_bets = len(all_results)
    n_wins = sum(1 for r in all_results if r["won"])
    hit_rate = n_wins / n_bets if n_bets > 0 else 0
    avg_tickets = np.mean([r["tickets"] for r in all_results])

    payouts_when_hit = [r["hit_odds"] for r in all_results if r["won"]]

    # Fit lognormal to payout distribution
    if payouts_when_hit:
        log_payouts = np.log(payouts_when_hit)
        payout_mu = float(np.mean(log_payouts))
        payout_sigma = float(np.std(log_payouts))
    else:
        payout_mu = DEFAULTS["payout_mu"]
        payout_sigma = DEFAULTS["payout_sigma"]

    # Days with bets
    daily = defaultdict(int)
    for r in all_results:
        daily[r["date"]] += 1
    bets_per_day = n_bets / len(daily) if daily else 0

    print(f"  {n_bets} bets, {n_wins} wins ({hit_rate:.1%}), "
          f"{bets_per_day:.2f} bets/day, "
          f"payout median={np.exp(payout_mu):.0f}x in {time.time()-t0:.0f}s",
          file=sys.stderr)

    return {
        "hit_rate": hit_rate,
        "bets_per_day": bets_per_day,
        "tickets_per_bet": avg_tickets,
        "payout_mu": payout_mu,
        "payout_sigma": payout_sigma,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo projection of trifecta strategy"
    )
    parser.add_argument("--bankroll", type=float, default=70_000,
                        help="Initial bankroll (default: 70000)")
    parser.add_argument("--unit-divisor", type=int, default=800,
                        help="Unit = bankroll / divisor (default: 800)")
    parser.add_argument("--min-unit", type=int, default=100,
                        help="Minimum unit (default: 100)")
    parser.add_argument("--bet-cap", type=int, default=2000,
                        help="Maximum unit / betCap (default: 2000)")
    parser.add_argument("--n-sims", type=int, default=10_000,
                        help="Number of simulations (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--days", type=str, default=None,
                        help="Comma-separated day counts (default: 30,90,180,365)")
    # Backtest-based parameter collection
    parser.add_argument("--from-backtest", action="store_true",
                        help="Collect empirical params from WF-CV backtest")
    parser.add_argument("--ev-threshold", type=float, default=0.33,
                        help="EV threshold for backtest (default: 0.33)")
    parser.add_argument("--b1-threshold", type=float, default=0.40,
                        help="B1 threshold for backtest (default: 0.40)")
    parser.add_argument("--n-folds", type=int, default=4,
                        help="WF-CV folds (default: 4)")
    # Manual parameter override
    parser.add_argument("--hit-rate", type=float, default=None)
    parser.add_argument("--bets-per-day", type=float, default=None)
    args = parser.parse_args()

    # Determine parameters
    if args.from_backtest:
        params = collect_from_backtest(
            ev_threshold=args.ev_threshold,
            b1_threshold=args.b1_threshold,
            n_folds=args.n_folds,
        )
    else:
        params = dict(DEFAULTS)

    # Manual overrides
    if args.hit_rate is not None:
        params["hit_rate"] = args.hit_rate
    if args.bets_per_day is not None:
        params["bets_per_day"] = args.bets_per_day

    # Periods
    if args.days:
        day_list = [int(d) for d in args.days.split(",")]
        periods = {f"{d}日": d for d in day_list}
    else:
        periods = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365}

    run_projection(
        n_sims=args.n_sims,
        periods=periods,
        params=params,
        bankroll=args.bankroll,
        unit_divisor=args.unit_divisor,
        min_unit=args.min_unit,
        max_unit=args.bet_cap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
