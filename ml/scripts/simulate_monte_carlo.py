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


def _extract_params(
    all_results: list[dict],
    b1_threshold: float,
    ev_threshold: float,
    all_flow: bool = False,
    mixed: bool = False,
    exacta_ratio: float = 1.0,
) -> dict:
    """Extract MC parameters from backtest results by filtering with thresholds.

    Modes:
        default: X-noB1-noB1 (12 tickets)
        all_flow: X-全流し (20 tickets)
        mixed: 3連単 X-全 + 2連単 X-全 (20 + 5*ratio tickets)
    """
    filtered = [r for r in all_results
                if r["b1_prob"] < b1_threshold and r["ev"] >= ev_threshold]

    n_bets = len(filtered)
    if n_bets == 0:
        return {"hit_rate": 0, "bets_per_day": 0, "tickets_per_bet": 0,
                "payout_mu": DEFAULTS["payout_mu"], "payout_sigma": DEFAULTS["payout_sigma"]}

    if mixed:
        # 3連単(20pt) + 2連単(5pt × ratio) combined
        n_wins = sum(1 for r in filtered if r.get("pick_1st"))
        avg_tickets = 20.0 + 5.0 * exacta_ratio
        payouts_when_hit = []
        for r in filtered:
            if not r.get("pick_1st"):
                continue
            tri = r.get("allflow_odds", 0)
            exa = r.get("exacta_hit_odds", 0) * exacta_ratio
            combined = tri + exa
            if combined > 0:
                payouts_when_hit.append(combined)
    elif all_flow:
        # X-全流し: hit when pick_1st is correct, 20 tickets
        n_wins = sum(1 for r in filtered if r.get("pick_1st"))
        avg_tickets = 20.0
        payouts_when_hit = [r["allflow_odds"] for r in filtered
                           if r.get("pick_1st") and r.get("allflow_odds", 0) > 0]
    else:
        # X-noB1-noB1: original 12 tickets
        n_wins = sum(1 for r in filtered if r["won"])
        avg_tickets = np.mean([r["tickets"] for r in filtered])
        payouts_when_hit = [r["hit_odds"] for r in filtered if r["won"]]

    hit_rate = n_wins / n_bets

    if payouts_when_hit:
        log_payouts = np.log(payouts_when_hit)
        payout_mu = float(np.mean(log_payouts))
        payout_sigma = float(np.std(log_payouts))
    else:
        payout_mu = DEFAULTS["payout_mu"]
        payout_sigma = DEFAULTS["payout_sigma"]

    daily = defaultdict(int)
    for r in filtered:
        daily[r["date"]] += 1
    bets_per_day = n_bets / len(daily) if daily else 0

    return {
        "hit_rate": hit_rate,
        "bets_per_day": bets_per_day,
        "tickets_per_bet": avg_tickets,
        "payout_mu": payout_mu,
        "payout_sigma": payout_sigma,
    }


def collect_all_candidates(model_dir: str = "models/trifecta_v1") -> list[dict]:
    """Load saved production models and collect all candidate bets on OOS period."""
    from collections import defaultdict

    from boatrace_tipster_ml.boat1_features import reshape_to_boat1
    from boatrace_tipster_ml.boat1_model import load_boat1_model
    from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
    from boatrace_tipster_ml.feature_config import prepare_feature_matrix
    from boatrace_tipster_ml.features import build_features_df
    from boatrace_tipster_ml.model import load_model, load_model_meta

    print("Collecting all candidates with saved models...", file=sys.stderr)
    t0 = time.time()

    # Determine OOS period from model_meta training date range
    rank_meta = load_model_meta(f"{model_dir}/ranking")
    date_range = rank_meta.get("training", {}).get("date_range", "") if rank_meta else ""
    # date_range format: "2024-01-01 ~ 2025-12-31"
    oos_start = date_range.split("~")[-1].strip() if "~" in date_range else "2026-01-01"
    # Day after training end
    from datetime import datetime, timedelta
    oos_start = (datetime.strptime(oos_start, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(DEFAULT_DB_PATH, start_date=oos_start)

    conn = get_connection(DEFAULT_DB_PATH)
    rows = conn.execute("SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '3連単'").fetchall()
    trifecta_odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}
    tri_win_prob: dict[tuple[int, int], float] = defaultdict(float)
    for r in rows:
        rid, combo, odds = int(r[0]), r[1], float(r[2])
        if odds > 0:
            tri_win_prob[(rid, int(combo.split("-")[0]))] += 0.75 / odds

    exacta_rows = conn.execute("SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '2連単'").fetchall()
    exacta_odds = {(int(r[0]), r[1]): float(r[2]) for r in exacta_rows}
    conn.close()

    finish_map: dict[tuple[int, int], int] = {}
    race_date_map: dict[int, str] = {}
    for _, row in df[["race_id", "boat_number", "finish_position", "race_date"]].drop_duplicates().iterrows():
        if pd.notna(row["finish_position"]):
            finish_map[(int(row["race_id"]), int(row["boat_number"]))] = int(row["finish_position"])
        race_date_map[int(row["race_id"])] = str(row["race_date"])

    # Load saved models
    b1_model = load_boat1_model(f"{model_dir}/boat1")
    rank_model = load_model(f"{model_dir}/ranking")

    with contextlib.redirect_stdout(io.StringIO()):
        X_b1, _, meta_b1 = reshape_to_boat1(df)
    b1_probs = b1_model.predict_proba(X_b1)[:, 1]

    X_rank, _, meta_rank = prepare_feature_matrix(df)
    rank_scores = rank_model.predict(X_rank)

    # Evaluate with no filtering (b1=1.0, ev=-999) to collect all candidates
    n_races = len(rank_scores) // 6
    scores_2d = rank_scores.reshape(n_races, 6)
    boats_2d = meta_rank["boat_number"].values.reshape(n_races, 6)
    race_ids = meta_rank["race_id"].values.reshape(n_races, 6)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    b1_map = {rid: i for i, rid in enumerate(meta_b1["race_id"].values)}

    all_results = []
    for ri in range(n_races):
        rid = int(race_ids[ri])
        bi = b1_map.get(rid)
        if bi is None:
            continue
        b1p = float(b1_probs[bi])

        wp = int(top_boats[ri, 0])
        if wp == 1:
            wp = int(top_boats[ri, 1])

        bidx = np.where(boats_2d[ri] == wp)[0]
        if len(bidx) == 0:
            continue
        wprob = float(rank_probs[ri, bidx[0]])

        mkt_prob = tri_win_prob.get((rid, wp), 0)
        if mkt_prob <= 0:
            continue
        ev = wprob / mkt_prob * 0.75 - 1

        pick_1st = finish_map.get((rid, wp)) == 1
        allflow_odds = 0.0
        exacta_hit_odds = 0.0
        a2 = a3 = None
        if pick_1st:
            for b in range(1, 7):
                fp = finish_map.get((rid, b))
                if fp == 2: a2 = b
                if fp == 3: a3 = b
            if a2 and a3:
                hc = f"{wp}-{a2}-{a3}"
                ho = trifecta_odds.get((rid, hc))
                if ho and ho > 0:
                    allflow_odds = ho
                ec = f"{wp}-{a2}"
                eo = exacta_odds.get((rid, ec))
                if eo and eo > 0:
                    exacta_hit_odds = eo

        all_results.append({
            "race_id": rid,
            "date": race_date_map.get(rid, ""),
            "winner_pick": wp,
            "b1_prob": round(b1p, 3),
            "winner_prob": round(wprob, 3),
            "ev": round(ev, 3),
            "pick_1st": pick_1st,
            "allflow_odds": round(allflow_odds, 1),
            "exacta_hit_odds": round(exacta_hit_odds, 1),
        })

    print(f"  {len(all_results)} candidates from OOS ({oos_start}~) in {time.time()-t0:.0f}s",
          file=sys.stderr)
    return all_results


def collect_from_backtest(
    ev_threshold: float,
    b1_threshold: float,
    all_flow: bool = False,
    mixed: bool = False,
    exacta_ratio: float = 1.0,
) -> dict:
    """Collect empirical distribution parameters from saved model OOS evaluation."""
    all_results = collect_all_candidates()
    params = _extract_params(
        all_results, b1_threshold, ev_threshold,
        all_flow=all_flow, mixed=mixed, exacta_ratio=exacta_ratio,
    )

    print(f"  b1<{b1_threshold} ev>+{ev_threshold:.0%}: "
          f"{sum(1 for r in all_results if r['b1_prob'] < b1_threshold and r['ev'] >= ev_threshold)} bets, "
          f"{params['bets_per_day']:.2f}/day, hit={params['hit_rate']:.1%}",
          file=sys.stderr)
    return params


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
    # Compare mode: train once, sweep thresholds
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare multiple threshold sets: 'b1:ev,b1:ev,...'")
    parser.add_argument("--all-flow", action="store_true",
                        help="Use X-全流し (20 tickets) instead of X-noB1-noB1 (12 tickets)")
    parser.add_argument("--mixed", action="store_true",
                        help="Use 3連単 X-全 + 2連単 X-全 combined (25 tickets)")
    parser.add_argument("--exacta-ratio", type=float, default=1.0,
                        help="2連単 unit multiplier relative to 3連単 (default: 1.0)")
    args = parser.parse_args()

    # Periods
    if args.days:
        day_list = [int(d) for d in args.days.split(",")]
        periods = {f"{d}日": d for d in day_list}
    else:
        periods = {"1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180, "1年": 365}

    # Compare mode: single backtest, multiple threshold sweeps
    if args.compare:
        all_results = collect_all_candidates()
        if args.mixed:
            flow_label = f"混合(25点×{args.exacta_ratio})"
        elif args.all_flow:
            flow_label = "全流し(20点)"
        else:
            flow_label = "noB1(12点)"
        pairs = [p.strip().split(":") for p in args.compare.split(",")]
        for b1s, evs in pairs:
            b1_thr, ev_thr = float(b1s), float(evs)
            params = _extract_params(
                all_results, b1_thr, ev_thr,
                all_flow=args.all_flow or args.mixed,
                mixed=args.mixed, exacta_ratio=args.exacta_ratio,
            )
            n = sum(1 for r in all_results
                    if r["b1_prob"] < b1_thr and r["ev"] >= ev_thr)
            print(f"\n{'='*70}")
            print(f"[{flow_label}] b1<{b1_thr} ev>+{ev_thr:.0%} — {n} bets, "
                  f"{params['bets_per_day']:.2f}/day, hit={params['hit_rate']:.1%}")
            print(f"{'='*70}")
            run_projection(
                n_sims=args.n_sims, periods=periods, params=params,
                bankroll=args.bankroll, unit_divisor=args.unit_divisor,
                min_unit=args.min_unit, max_unit=args.bet_cap, seed=args.seed,
            )
        return

    # Single run mode
    if args.from_backtest:
        params = collect_from_backtest(
            ev_threshold=args.ev_threshold,
            b1_threshold=args.b1_threshold,
            all_flow=args.all_flow or args.mixed,
            mixed=args.mixed,
            exacta_ratio=args.exacta_ratio,
        )
    else:
        params = dict(DEFAULTS)

    # Manual overrides
    if args.hit_rate is not None:
        params["hit_rate"] = args.hit_rate
    if args.bets_per_day is not None:
        params["bets_per_day"] = args.bets_per_day

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
