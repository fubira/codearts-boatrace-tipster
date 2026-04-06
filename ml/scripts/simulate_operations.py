"""Simulate operational performance of trifecta X-noB1-noB1 strategy.

Outputs key metrics for operational planning:
- Revenue projections (3mo, 6mo, 12mo)
- Win/loss day rates
- Losing streaks
- Maximum drawdown
- Compound growth simulation

Usage:
    uv run --directory ml python -m scripts.simulate_operations
    uv run --directory ml python -m scripts.simulate_operations --ev 0.10 --ev 0.33
    uv run --directory ml python -m scripts.simulate_operations --b1-threshold 0.48
"""

import argparse
import contextlib
import io
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import load_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model

FIELD_SIZE = 6

# Optuna best LambdaRank params
DEFAULT_RANK_PARAMS = {
    "num_leaves": 108,
    "max_depth": 9,
    "min_child_samples": 42,
    "subsample": 0.822,
    "colsample_bytree": 0.402,
    "reg_alpha": 4.92e-08,
    "reg_lambda": 2.0e-08,
}
DEFAULT_N_ESTIMATORS = 915
DEFAULT_LEARNING_RATE = 0.052
DEFAULT_RELEVANCE = "top_heavy"


def load_and_train(db_path, train_end="2026-01-01", val_days=60, model_dir=None):
    """Load data, train or load models.

    If model_dir is given and contains saved models, skip training.
    If model_dir is given but empty, train and save there.
    If model_dir is None, always train (no save).
    """
    b1_dir = os.path.join(model_dir, "boat1") if model_dir else None
    rank_dir = os.path.join(model_dir, "ranking") if model_dir else None

    print("Loading features...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(db_path)

    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '3連単'"
    ).fetchall()
    trifecta_odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    tri_win_prob: dict[tuple[int, int], float] = defaultdict(float)
    for r in rows:
        rid, combo, odds = int(r[0]), r[1], float(r[2])
        if odds <= 0:
            continue
        tri_win_prob[(rid, int(combo.split("-")[0]))] += 0.75 / odds
    conn.close()

    finish_map: dict[tuple[int, int], int] = {}
    race_date_map: dict[int, str] = {}
    for _, row in (
        df[["race_id", "boat_number", "finish_position", "race_date"]]
        .drop_duplicates()
        .iterrows()
    ):
        if pd.notna(row["finish_position"]):
            finish_map[(int(row["race_id"]), int(row["boat_number"]))] = int(
                row["finish_position"]
            )
        race_date_map[int(row["race_id"])] = str(row["race_date"])

    test_df = df[df["race_date"] >= train_end]

    # Load saved production models (no retraining)
    if not (
        b1_dir
        and os.path.exists(os.path.join(b1_dir, "model.pkl"))
        and rank_dir
        and os.path.exists(os.path.join(rank_dir, "model.pkl"))
    ):
        print(f"ERROR: Saved models not found in {model_dir}/. Run train_ranking.py and train_boat1_binary.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading saved models from {model_dir}/...", file=sys.stderr)
    b1_model = load_boat1_model(b1_dir)
    rank_model = load_model(rank_dir)

    print(f"Test: {test_df['race_id'].nunique()}R ({train_end}~)", file=sys.stderr)

    # Prepare test features and predict
    print("Predicting...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        X_b1_te, _, meta_b1_te = reshape_to_boat1(test_df)
        X_r_te, _, meta_r_te = prepare_feature_matrix(test_df)

    b1_probs = b1_model.predict_proba(X_b1_te)[:, 1]
    rank_scores = rank_model.predict(X_r_te)

    n_races = len(X_r_te) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_r_te["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta_r_te["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    b1_map = {rid: i for i, rid in enumerate(meta_b1_te["race_id"].values)}

    return {
        "b1_probs": b1_probs,
        "rank_probs": rank_probs,
        "boats_2d": boats_2d,
        "race_ids": race_ids,
        "top_boats": top_boats,
        "b1_map": b1_map,
        "n_races": n_races,
        "trifecta_odds": trifecta_odds,
        "tri_win_prob": dict(tri_win_prob),
        "finish_map": finish_map,
        "race_date_map": race_date_map,
    }


def collect_daily_results(data, b1_threshold, ev_threshold):
    """Collect per-day P/L results for a given parameter set."""
    daily = defaultdict(lambda: {"races": 0, "tickets": 0, "wins": 0, "payout": 0.0})

    for ri in range(data["n_races"]):
        rid = int(data["race_ids"][ri])
        bi = data["b1_map"].get(rid)
        if bi is None:
            continue
        if float(data["b1_probs"][bi]) >= b1_threshold:
            continue

        wp = int(data["top_boats"][ri, 0])
        if wp == 1:
            wp = int(data["top_boats"][ri, 1])

        bidx = np.where(data["boats_2d"][ri] == wp)[0]
        if len(bidx) == 0:
            continue
        wprob = float(data["rank_probs"][ri, bidx[0]])

        mkt_prob = data["tri_win_prob"].get((rid, wp), 0)
        if mkt_prob <= 0:
            continue
        ev = wprob / mkt_prob * 0.75 - 1
        if ev < ev_threshold:
            continue

        date = data["race_date_map"].get(rid, "")
        excluded = {wp, 1}
        flow = [int(b) for b in data["boats_2d"][ri] if int(b) not in excluded]
        tkts = []
        for b2 in flow:
            for b3 in flow:
                if b2 != b3:
                    c = f"{wp}-{b2}-{b3}"
                    if (rid, c) in data["trifecta_odds"]:
                        tkts.append(c)
        if not tkts:
            continue

        daily[date]["races"] += 1
        daily[date]["tickets"] += len(tkts)

        a2 = a3 = None
        for b in range(1, 7):
            fp = data["finish_map"].get((rid, b))
            if fp == 2:
                a2 = b
            if fp == 3:
                a3 = b

        if data["finish_map"].get((rid, wp)) == 1 and a2 and a3:
            hc = f"{wp}-{a2}-{a3}"
            if hc in tkts:
                ho = data["trifecta_odds"].get((rid, hc))
                if ho:
                    daily[date]["wins"] += 1
                    daily[date]["payout"] += ho

    return daily


def analyze_operations(daily, label):
    """Analyze and print operational metrics."""
    dates = sorted(daily.keys())
    if not dates:
        print(f"\n=== {label}: No qualifying races ===")
        return

    total_days = len(dates)
    total_r = sum(d["races"] for d in daily.values())
    total_t = sum(d["tickets"] for d in daily.values())
    total_w = sum(d["wins"] for d in daily.values())
    total_p = sum(d["payout"] for d in daily.values())
    roi = total_p / total_t if total_t > 0 else 0

    # Daily P/L series
    daily_pls = []
    for date in dates:
        d = daily[date]
        daily_pls.append(d["payout"] - d["tickets"])

    cum = np.cumsum(daily_pls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd))
    max_dd_day = dates[int(np.argmin(dd))]

    # Win/lose days
    win_days = sum(1 for pl in daily_pls if pl > 0)
    lose_days = sum(1 for pl in daily_pls if pl < 0)
    even_days = sum(1 for pl in daily_pls if pl == 0)
    zero_hit_days = sum(1 for d in daily.values() if d["wins"] == 0)

    # Losing streaks
    streaks = []
    current = 0
    for pl in daily_pls:
        if pl < 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    max_streak = max(streaks) if streaks else 0
    avg_streak = np.mean(streaks) if streaks else 0

    # Monthly breakdown
    monthly = defaultdict(lambda: {"tickets": 0, "payout": 0.0, "races": 0, "wins": 0})
    for date in dates:
        m = date[:7]
        d = daily[date]
        monthly[m]["tickets"] += d["tickets"]
        monthly[m]["payout"] += d["payout"]
        monthly[m]["races"] += d["races"]
        monthly[m]["wins"] += d["wins"]

    # Revenue projections (per ¥100 unit)
    pl_per_day = (total_p - total_t) / total_days
    tkt_per_day = total_t / total_days

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    print(f"\n--- Summary ({dates[0]} ~ {dates[-1]}, {total_days} days) ---")
    print(f"  Races: {total_r} ({total_r/total_days:.1f}/d)")
    print(f"  Tickets: {total_t} ({total_t/total_days:.0f}/d)")
    print(f"  Wins: {total_w} ({total_w/total_days*7:.1f}/week)")
    print(f"  ROI: {roi:.0%}")
    print(f"  P/L: {total_p - total_t:+.0f} (per ¥100 unit)")

    print(f"\n--- Win/Loss Days ---")
    print(f"  Win days:     {win_days:>3}/{total_days} ({win_days/total_days:.0%})")
    print(f"  Lose days:    {lose_days:>3}/{total_days} ({lose_days/total_days:.0%})")
    print(f"  Zero-hit days:{zero_hit_days:>3}/{total_days} ({zero_hit_days/total_days:.0%})")

    print(f"\n--- Losing Streaks ---")
    print(f"  Max streak:  {max_streak} days")
    print(f"  Avg streak:  {avg_streak:.1f} days")
    print(f"  Streak count:{len(streaks)}")

    print(f"\n--- Drawdown (per ¥100 unit) ---")
    print(f"  Max DD: {max_dd:+.0f} (on {max_dd_day})")

    print(f"\n--- Monthly Breakdown ---")
    for m in sorted(monthly.keys()):
        md = monthly[m]
        mr = md["payout"] / md["tickets"] if md["tickets"] > 0 else 0
        mpl = md["payout"] - md["tickets"]
        print(
            f"  {m}: {md['races']:>3}R {md['tickets']:>4}tkt "
            f"{md['wins']:>2}W ROI {mr:>4.0%} P/L {mpl:>+7.0f}"
        )

    print(f"\n--- Revenue Projections ---")
    for unit_size in [100, 500, 1000, 2000]:
        for months, label_m in [(3, "3mo"), (6, "6mo"), (12, "12mo")]:
            proj = pl_per_day * 30 * months * unit_size
            cost = tkt_per_day * 30 * months * unit_size
            print(
                f"  ¥{unit_size:>5}/tkt × {label_m}: "
                f"invest ¥{cost:>10,.0f} → profit ¥{proj:>+10,.0f}"
            )
        print()

    print(f"--- Compound Growth (¥70k start → MAX ¥2,000/tkt) ---")
    bankroll = 70000
    for month in range(1, 13):
        unit = max(100, min(2000, int(bankroll / 800 / 100) * 100))
        m_wagered = tkt_per_day * 30 * unit
        m_profit = pl_per_day * 30 * unit
        bankroll += m_profit
        if bankroll <= 0:
            print(f"  Month {month:>2}: BANKRUPT")
            break
        print(
            f"  Month {month:>2}: ¥{unit:>5}/tkt "
            f"profit ¥{m_profit:>+10,.0f} "
            f"bankroll ¥{bankroll:>12,.0f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Simulate trifecta operational performance"
    )
    parser.add_argument(
        "--ev",
        type=float,
        action="append",
        default=None,
        help="EV thresholds to test (can specify multiple)",
    )
    parser.add_argument("--b1-threshold", type=float, default=0.482)
    parser.add_argument("--train-end", default="2026-01-01")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model directory (e.g. models/trifecta_v1). Saves on first run, loads on subsequent runs.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    ev_thresholds = args.ev if args.ev else [0.10, 0.33]

    t0 = time.time()
    data = load_and_train(
        args.db_path, train_end=args.train_end, model_dir=args.model_dir
    )
    print(f"Setup: {time.time() - t0:.1f}s", file=sys.stderr)

    for ev_thr in ev_thresholds:
        daily = collect_daily_results(data, args.b1_threshold, ev_thr)
        analyze_operations(
            daily,
            f"X-noB1-noB1 | b1<{args.b1_threshold:.0%} EV>={ev_thr:.0%} | "
            f"train ~{args.train_end}",
        )

    print(f"\nTotal: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
