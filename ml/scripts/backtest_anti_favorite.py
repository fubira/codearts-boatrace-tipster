"""Backtest anti-favorite strategy: when AI predicts boat 1 loses, buy the top-ranked boat's tansho.

Uses boat1 binary model to identify "boat 1 will lose" races (prob < threshold),
then LambdaRank to identify the most likely winner among all boats.

Usage:
    uv run --directory ml python -m scripts.backtest_anti_favorite \
        --from 2025-08-01 --to 2026-04-03
"""

import argparse
import contextlib
import json
import sys

import numpy as np
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURE_COLS, prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model


FIELD_SIZE = 6


def _get_tansho_odds_map(db_path: str) -> dict[tuple[int, int], float]:
    """Load all tansho odds as {(race_id, boat_number): odds}."""
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
    """Get {race_id: winning_boat_number} from the full feature df."""
    winners = df[df["finish_position"] == 1][["race_id", "boat_number"]]
    return dict(zip(winners["race_id"].values, winners["boat_number"].values))


def _evaluate_fold(
    b1_probs: np.ndarray,
    meta_b1_test: pd.DataFrame,
    rank_scores: np.ndarray,
    meta_rank_test: pd.DataFrame,
    boats_2d: np.ndarray,
    race_ids: np.ndarray,
    odds_map: dict,
    winner_map: dict,
    n_races: int,
) -> list[dict]:
    """Evaluate anti-favorite strategy for one fold across threshold/EV grid."""
    # Softmax probabilities from ranking scores
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    exp_scores = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    b1_race_ids = meta_b1_test["race_id"].values
    b1_race_to_idx = {rid: i for i, rid in enumerate(b1_race_ids)}

    results = []
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for ev_thr in [-1, 0, 10, 20]:
            bets = 0
            wins = 0
            total_payout = 0.0

            for race_idx in range(n_races):
                rid = int(race_ids[race_idx])
                b1_idx = b1_race_to_idx.get(rid)
                if b1_idx is None:
                    continue

                b1_prob = float(b1_probs[b1_idx])
                if b1_prob >= threshold:
                    continue

                top_rank = 0
                top_boat = int(top_boats[race_idx, top_rank])
                if top_boat == 1:
                    top_rank = 1
                    top_boat = int(top_boats[race_idx, top_rank])

                odds = odds_map.get((rid, top_boat))
                if odds is None or odds <= 0:
                    continue

                if ev_thr >= 0:
                    boat_idx = np.where(boats_2d[race_idx] == top_boat)[0]
                    if len(boat_idx) == 0:
                        continue
                    prob = float(rank_probs[race_idx, boat_idx[0]])
                    if prob * odds - 1 < ev_thr / 100:
                        continue

                if winner_map.get(rid) == top_boat:
                    wins += 1
                    total_payout += odds
                bets += 1

            results.append({
                "b1_threshold": threshold,
                "ev_threshold": ev_thr,
                "bets": bets,
                "wins": wins,
                "hit_rate": round(wins / bets, 4) if bets > 0 else 0,
                "roi": round(total_payout / bets, 4) if bets > 0 else 0,
            })
    return results


def backtest_wfcv(
    db_path: str,
    n_folds: int = 4,
    fold_months: int = 2,
) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        df = build_features_df(db_path)

    odds_map = _get_tansho_odds_map(db_path)
    winner_map = _get_winner_map(df)

    # Build WF-CV splits manually using date ranges
    race_dates = pd.to_datetime(df["race_date"])
    last_date = race_dates.max()

    all_folds = []
    for i in range(n_folds):
        test_end = last_date - pd.DateOffset(months=fold_months * i)
        test_start = test_end - pd.DateOffset(months=fold_months)
        val_start = test_start - pd.DateOffset(months=1)

        test_end_s = str(test_end.date())
        test_start_s = str(test_start.date())
        val_start_s = str(val_start.date())

        train_df = df[df["race_date"] < val_start_s]
        val_df = df[(df["race_date"] >= val_start_s) & (df["race_date"] < test_start_s)]
        test_df = df[(df["race_date"] >= test_start_s) & (df["race_date"] < test_end_s)]

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            continue

        period = f"{test_start_s} ~ {test_end_s}"
        print(f"\n=== Fold {i+1}: {period} ===", file=sys.stderr)
        print(f"  Train: {len(train_df)//6}R, Val: {len(val_df)//6}R, Test: {len(test_df)//6}R", file=sys.stderr)

        # Train boat1 binary
        with contextlib.redirect_stdout(sys.stderr):
            X_b1_tr, y_b1_tr, _ = reshape_to_boat1(train_df)
            X_b1_val, y_b1_val, _ = reshape_to_boat1(val_df)
            X_b1_test, _, meta_b1_test = reshape_to_boat1(test_df)
            b1_model, b1_m = train_boat1_model(X_b1_tr, y_b1_tr, X_b1_val, y_b1_val)
            print(f"  Boat1 AUC: {b1_m.get('val_auc', 'N/A'):.4f}", file=sys.stderr)

        # Train LambdaRank
        with contextlib.redirect_stdout(sys.stderr):
            X_r_tr, y_r_tr, m_r_tr = prepare_feature_matrix(train_df)
            X_r_val, y_r_val, m_r_val = prepare_feature_matrix(val_df)
            X_r_test, _, m_r_test = prepare_feature_matrix(test_df)
            rank_model, _ = train_model(
                X_r_tr, y_r_tr, m_r_tr, X_r_val, y_r_val, m_r_val,
                early_stopping_rounds=50,
            )

        b1_probs = b1_model.predict_proba(X_b1_test)[:, 1]
        rank_scores = rank_model.predict(X_r_test)

        n_races = len(X_r_test) // FIELD_SIZE
        boats_2d = m_r_test["boat_number"].values.reshape(n_races, FIELD_SIZE)
        race_ids = m_r_test["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

        fold_results = _evaluate_fold(
            b1_probs, meta_b1_test, rank_scores, m_r_test,
            boats_2d, race_ids, odds_map, winner_map, n_races,
        )

        # Print key results for this fold
        for r in fold_results:
            if r["bets"] > 0 and r["ev_threshold"] == 0 and r["b1_threshold"] in [0.30, 0.40, 0.50]:
                print(
                    f"  b1<{r['b1_threshold']:.0%} EV≥0: {r['bets']} bets, ROI {r['roi']:.1%}",
                    file=sys.stderr,
                )

        all_folds.append({"period": period, "n_races": n_races, "results": fold_results})

    # Aggregate across folds
    summary = {}
    for r in all_folds[0]["results"]:
        key = (r["b1_threshold"], r["ev_threshold"])
        summary[key] = {"rois": [], "bets": [], "wins": []}

    for fold in all_folds:
        for r in fold["results"]:
            key = (r["b1_threshold"], r["ev_threshold"])
            summary[key]["rois"].append(r["roi"])
            summary[key]["bets"].append(r["bets"])
            summary[key]["wins"].append(r["wins"])

    agg = []
    for (thr, ev), v in sorted(summary.items()):
        rois = v["rois"]
        total_bets = sum(v["bets"])
        total_wins = sum(v["wins"])
        agg.append({
            "b1_threshold": thr,
            "ev_threshold": ev,
            "total_bets": total_bets,
            "total_wins": total_wins,
            "avg_roi": round(np.mean(rois), 4),
            "std_roi": round(np.std(rois), 4),
            "min_roi": round(min(rois), 4),
            "max_roi": round(max(rois), 4),
            "fold_rois": [round(r, 4) for r in rois],
        })

    return {"n_folds": len(all_folds), "folds": all_folds, "summary": agg}


def backtest(
    from_date: str,
    to_date: str,
    db_path: str,
    boat1_threshold: float = 0.40,
) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        df = build_features_df(db_path)

    odds_map = _get_tansho_odds_map(db_path)
    winner_map = _get_winner_map(df)

    train_df = df[df["race_date"] < from_date]
    test_df = df[(df["race_date"] >= from_date) & (df["race_date"] < to_date)]

    if len(test_df) == 0:
        return {"error": f"No race data in {from_date} ~ {to_date}"}

    # Val split for early stopping
    dates = sorted(train_df["race_date"].unique())
    val_start = dates[max(0, len(dates) - 60)]

    train_early = train_df[train_df["race_date"] < val_start]
    train_late = train_df[train_df["race_date"] >= val_start]

    # --- Train boat1 binary model ---
    with contextlib.redirect_stdout(sys.stderr):
        X_b1_train, y_b1_train, _ = reshape_to_boat1(train_early)
        X_b1_val, y_b1_val, _ = reshape_to_boat1(train_late)
        X_b1_test, y_b1_test, meta_b1_test = reshape_to_boat1(test_df)

        print(f"Boat1 - Train: {len(X_b1_train)}, Val: {len(X_b1_val)}, Test: {len(X_b1_test)}", file=sys.stderr)
        b1_model, b1_metrics = train_boat1_model(X_b1_train, y_b1_train, X_b1_val, y_b1_val)
        print(f"Boat1 Val AUC: {b1_metrics.get('val_auc', 'N/A')}", file=sys.stderr)

    # --- Train LambdaRank model ---
    with contextlib.redirect_stdout(sys.stderr):
        X_rank_train, y_rank_train, meta_rank_train = prepare_feature_matrix(train_early)
        X_rank_val, y_rank_val, meta_rank_val = prepare_feature_matrix(train_late)
        X_rank_test, y_rank_test, meta_rank_test = prepare_feature_matrix(test_df)

        print(f"Rank - Train: {len(X_rank_train)}, Val: {len(X_rank_val)}, Test: {len(X_rank_test)}", file=sys.stderr)
        rank_model, rank_metrics = train_model(
            X_rank_train, y_rank_train, meta_rank_train,
            X_rank_val, y_rank_val, meta_rank_val,
            early_stopping_rounds=50,
        )

    # --- Predict ---
    b1_probs = b1_model.predict_proba(X_b1_test)[:, 1]
    rank_scores = rank_model.predict(X_rank_test)

    # Reshape ranking scores to (n_races, 6)
    n_races = len(X_rank_test) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_rank_test["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids_2d = meta_rank_test["race_id"].values.reshape(n_races, FIELD_SIZE)

    # Top predicted boat per race
    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)
    race_ids = race_ids_2d[:, 0]

    # Map race_id to index in b1 predictions
    b1_race_ids = meta_b1_test["race_id"].values
    b1_race_to_idx = {rid: i for i, rid in enumerate(b1_race_ids)}

    # Softmax probabilities from ranking scores
    exp_scores = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # --- Evaluate across thresholds ---
    results_by_threshold = []
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        for ev_thr in [-1, 0, 10, 20, 30]:  # -1 = no EV filter
            bets = 0
            wins = 0
            total_payout = 0.0

            for race_idx in range(n_races):
                rid = int(race_ids[race_idx])
                b1_idx = b1_race_to_idx.get(rid)
                if b1_idx is None:
                    continue

                b1_prob = float(b1_probs[b1_idx])
                if b1_prob >= threshold:
                    continue  # AI thinks boat 1 will win → skip

                # AI's top pick (could be boat 1 or any boat)
                top_rank = 0
                top_boat = int(top_boats[race_idx, top_rank])

                # Skip if top pick is boat 1 (we're betting against boat 1)
                if top_boat == 1:
                    top_rank = 1
                    top_boat = int(top_boats[race_idx, top_rank])

                odds = odds_map.get((rid, top_boat))
                if odds is None or odds <= 0:
                    continue

                # EV filter using softmax prob
                if ev_thr >= 0:
                    # Get prob for the selected boat
                    boat_idx = np.where(boats_2d[race_idx] == top_boat)[0]
                    if len(boat_idx) == 0:
                        continue
                    prob = float(rank_probs[race_idx, boat_idx[0]])
                    ev = prob * odds - 1
                    if ev < ev_thr / 100:
                        continue

                actual_winner = winner_map.get(rid)
                won = actual_winner == top_boat

                bets += 1
                if won:
                    wins += 1
                    total_payout += odds

            roi = total_payout / bets if bets > 0 else 0
            hit_rate = wins / bets if bets > 0 else 0
            profit_100 = round((total_payout - bets) * 100)

            ev_label = "none" if ev_thr < 0 else f"{ev_thr}%"
            results_by_threshold.append({
                "b1_prob_threshold": threshold,
                "ev_threshold": ev_thr,
                "bets": bets,
                "wins": wins,
                "hit_rate": round(hit_rate, 4),
                "roi": round(roi, 4),
                "profit_per_100yen": profit_100,
            })

            if bets > 0:
                print(
                    f"  b1<{threshold:.0%} EV≥{ev_label}: {bets} bets, {wins} wins ({hit_rate:.1%}), "
                    f"ROI {roi:.1%}, P/L ¥{profit_100:,}",
                    file=sys.stderr,
                )

    # --- Also test: just buy top-ranked (include boat 1) for comparison ---
    bets_all = 0
    wins_all = 0
    payout_all = 0.0
    for race_idx in range(n_races):
        rid = int(race_ids[race_idx])
        top_boat = int(top_boats[race_idx, 0])
        odds = odds_map.get((rid, top_boat))
        if odds is None or odds <= 0:
            continue
        bets_all += 1
        if winner_map.get(rid) == top_boat:
            wins_all += 1
            payout_all += odds

    baseline = {
        "strategy": "LambdaRank top1 (all races)",
        "bets": bets_all,
        "wins": wins_all,
        "hit_rate": round(wins_all / bets_all, 4) if bets_all > 0 else 0,
        "roi": round(payout_all / bets_all, 4) if bets_all > 0 else 0,
    }

    return {
        "from_date": from_date,
        "to_date": to_date,
        "n_races": n_races,
        "boat1_val_auc": b1_metrics.get("val_auc"),
        "results": results_by_threshold,
        "baseline": baseline,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest anti-favorite strategy")
    parser.add_argument("--from", dest="from_date")
    parser.add_argument("--to", dest="to_date")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--wfcv", action="store_true", help="Run walk-forward CV")
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    if args.wfcv:
        result = backtest_wfcv(args.db_path, n_folds=args.folds)
    else:
        if not args.from_date or not args.to_date:
            parser.error("--from and --to required unless --wfcv")
        result = backtest(args.from_date, args.to_date, args.db_path)
    json.dump(result, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
