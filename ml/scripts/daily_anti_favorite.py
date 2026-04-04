"""Daily breakdown of anti-favorite strategy performance.

Train models on data before the period, then show per-day results.

Usage:
    uv run --directory ml python -m scripts.daily_anti_favorite \
        --from 2026-03-29 --to 2026-04-04
"""

import argparse
import contextlib
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURE_COLS, prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model

FIELD_SIZE = 6
B1_THRESHOLD = 0.40
EV_THRESHOLD = 0  # EV >= 0%


def _get_tansho_odds_map(db_path: str) -> dict[tuple[int, int], float]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT race_id, CAST(combination AS INTEGER), odds "
            "FROM db.race_odds WHERE bet_type = '単勝'"
        ).fetchall()
        return {(int(r[0]), int(r[1])): float(r[2]) for r in rows}
    finally:
        conn.close()


def _get_winner_map(df: pd.DataFrame) -> dict[int, int]:
    winners = df[df["finish_position"] == 1][["race_id", "boat_number"]]
    return dict(zip(winners["race_id"].values, winners["boat_number"].values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    print(f"Loading features...", file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        df = build_features_df(args.db_path)

    odds_map = _get_tansho_odds_map(args.db_path)
    winner_map = _get_winner_map(df)

    train_df = df[df["race_date"] < args.from_date]
    test_df = df[(df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)]

    if len(test_df) == 0:
        print(f"No race data in {args.from_date} ~ {args.to_date}", file=sys.stderr)
        sys.exit(1)

    # Val split for early stopping (last 60 days of training)
    dates = sorted(train_df["race_date"].unique())
    val_start = dates[max(0, len(dates) - 60)]
    train_early = train_df[train_df["race_date"] < val_start]
    train_late = train_df[train_df["race_date"] >= val_start]

    # Train boat1 binary model
    print("Training boat1 binary model...", file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        X_b1_train, y_b1_train, _ = reshape_to_boat1(train_early)
        X_b1_val, y_b1_val, _ = reshape_to_boat1(train_late)
        X_b1_test, _, meta_b1_test = reshape_to_boat1(test_df)
        b1_model, b1_metrics = train_boat1_model(X_b1_train, y_b1_train, X_b1_val, y_b1_val)
        print(f"Boat1 Val AUC: {b1_metrics.get('val_auc', 'N/A')}", file=sys.stderr)

    # Train LambdaRank model
    print("Training LambdaRank model...", file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        X_rank_train, y_rank_train, meta_rank_train = prepare_feature_matrix(train_early)
        X_rank_val, y_rank_val, meta_rank_val = prepare_feature_matrix(train_late)
        X_rank_test, _, meta_rank_test = prepare_feature_matrix(test_df)
        rank_model, _ = train_model(
            X_rank_train, y_rank_train, meta_rank_train,
            X_rank_val, y_rank_val, meta_rank_val,
            early_stopping_rounds=50,
        )

    # Predict
    b1_probs = b1_model.predict_proba(X_b1_test)[:, 1]
    rank_scores = rank_model.predict(X_rank_test)

    n_races = len(X_rank_test) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_rank_test["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids_2d = meta_rank_test["race_id"].values.reshape(n_races, FIELD_SIZE)
    race_ids = race_ids_2d[:, 0]

    # Softmax probs
    exp_scores = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    b1_race_ids = meta_b1_test["race_id"].values
    b1_race_to_idx = {rid: i for i, rid in enumerate(b1_race_ids)}

    # Get race_id -> race_date mapping from test_df
    race_date_map = dict(zip(
        test_df.groupby("race_id")["race_date"].first().index,
        test_df.groupby("race_id")["race_date"].first().values,
    ))

    # Daily aggregation
    daily: dict[str, dict] = defaultdict(lambda: {
        "total_races": 0, "bets": 0, "wins": 0, "payout": 0.0, "details": [],
    })

    for race_idx in range(n_races):
        rid = int(race_ids[race_idx])
        race_date = str(race_date_map.get(rid, "unknown"))
        daily[race_date]["total_races"] += 1

        b1_idx = b1_race_to_idx.get(rid)
        if b1_idx is None:
            continue

        b1_prob = float(b1_probs[b1_idx])
        if b1_prob >= B1_THRESHOLD:
            continue

        top_rank = 0
        top_boat = int(top_boats[race_idx, top_rank])
        if top_boat == 1:
            top_rank = 1
            top_boat = int(top_boats[race_idx, top_rank])

        odds = odds_map.get((rid, top_boat))
        if odds is None or odds <= 0:
            continue

        # EV filter
        boat_idx = np.where(boats_2d[race_idx] == top_boat)[0]
        if len(boat_idx) == 0:
            continue
        prob = float(rank_probs[race_idx, boat_idx[0]])
        ev = prob * odds - 1
        if ev < EV_THRESHOLD / 100:
            continue

        actual_winner = winner_map.get(rid)
        won = actual_winner == top_boat

        daily[race_date]["bets"] += 1
        if won:
            daily[race_date]["wins"] += 1
            daily[race_date]["payout"] += odds
        daily[race_date]["details"].append({
            "race_id": rid,
            "b1_prob": round(b1_prob, 3),
            "bet_boat": top_boat,
            "rank_prob": round(prob, 3),
            "odds": odds,
            "ev": round(ev, 3),
            "won": won,
        })

    # Output
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Anti-Favorite Daily Results (b1<{B1_THRESHOLD:.0%}, EV≥{EV_THRESHOLD}%)", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    total_bets = 0
    total_wins = 0
    total_payout = 0.0
    result_days = []

    for date in sorted(daily.keys()):
        d = daily[date]
        bets = d["bets"]
        wins = d["wins"]
        payout = d["payout"]
        roi = payout / bets if bets > 0 else 0
        total_bets += bets
        total_wins += wins
        total_payout += payout

        print(
            f"  {date}: {d['total_races']}R中 {bets}購入, {wins}的中 "
            f"({wins}/{bets}={wins/bets:.0%} if bets else 0%), "
            f"ROI {roi:.0%}, P/L ¥{(payout - bets) * 100:+,.0f}",
            file=sys.stderr,
        ) if bets > 0 else print(
            f"  {date}: {d['total_races']}R中 購入なし", file=sys.stderr,
        )

        result_days.append({
            "date": date,
            "total_races": d["total_races"],
            "bets": bets,
            "wins": wins,
            "hit_rate": round(wins / bets, 4) if bets > 0 else 0,
            "roi": round(roi, 4) if bets > 0 else 0,
            "profit_per_100yen": round((payout - bets) * 100) if bets > 0 else 0,
            "details": d["details"],
        })

    total_roi = total_payout / total_bets if total_bets > 0 else 0
    print(f"\n  合計: {total_bets}購入, {total_wins}的中 ({total_wins/total_bets:.0%}), "
          f"ROI {total_roi:.0%}, P/L ¥{(total_payout - total_bets) * 100:+,.0f}",
          file=sys.stderr)

    json.dump({
        "from_date": args.from_date,
        "to_date": args.to_date,
        "b1_threshold": B1_THRESHOLD,
        "ev_threshold": EV_THRESHOLD,
        "boat1_val_auc": b1_metrics.get("val_auc"),
        "days": result_days,
        "total": {
            "bets": total_bets,
            "wins": total_wins,
            "hit_rate": round(total_wins / total_bets, 4) if total_bets > 0 else 0,
            "roi": round(total_roi, 4) if total_bets > 0 else 0,
            "profit_per_100yen": round((total_payout - total_bets) * 100) if total_bets > 0 else 0,
        },
    }, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
