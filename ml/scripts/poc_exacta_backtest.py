"""PoC: Exacta (2連単) backtest using conditional 2nd-place models.

Pipeline:
  1. Binary model → boat1 loses (prob < 40%)
  2. LambdaRank → pick winner (top non-boat1)
  3. Conditional model → pick 2nd place (top 1 or 2 candidates)
  4. EV filter → bet exacta

Usage:
    uv run --directory ml python -m scripts.poc_exacta_backtest
"""

import contextlib
import io
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model

FIELD_SIZE = 6
B1_THRESHOLD = 0.40

SECOND_FEATURE_COLS = [
    "boat_number",
    "course_number",
    "national_win_rate",
    "national_top2_rate",
    "national_top3_rate",
    "local_win_rate",
    "local_top3_rate",
    "motor_top3_rate",
    "exhibition_time",
    "exhibition_st",
    "racer_class_code",
    "racer_weight",
    "average_st",
    "racer_course_win_rate",
    "racer_course_top2_rate",
    "stadium_course_win_rate",
    "recent_avg_position",
    "rel_national_win_rate",
    "rel_exhibition_time",
    "rel_exhibition_st",
    "wind_speed",
    "wave_height",
]


def train_second_place_models(
    train_df: pd.DataFrame,
) -> dict[int, lgb.LGBMClassifier]:
    """Train one 2nd-place classifier per winner boat (2-6)."""
    models = {}
    for winner in range(2, 7):
        winner_races = train_df[
            (train_df["boat_number"] == winner) & (train_df["finish_position"] == 1)
        ]["race_id"].unique()

        subset = train_df[
            (train_df["race_id"].isin(winner_races))
            & (train_df["boat_number"] != winner)
        ].copy()

        if len(subset) == 0:
            continue

        y = (subset["finish_position"] == 2).astype(int)
        available = [c for c in SECOND_FEATURE_COLS if c in subset.columns]
        X = subset[available].fillna(0)

        # Add course distance from winner
        winner_courses = train_df[
            (train_df["race_id"].isin(winner_races))
            & (train_df["boat_number"] == winner)
        ][["race_id", "course_number"]].rename(
            columns={"course_number": "winner_course"}
        )
        merged = subset[["race_id", "course_number"]].merge(
            winner_courses, on="race_id", how="left"
        )
        X["course_distance_from_winner"] = (
            merged["course_number"] - merged["winner_course"]
        ).abs()

        model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=300,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        model.fit(X, y)
        models[winner] = model

    return models


def predict_second_place(
    model: lgb.LGBMClassifier,
    race_rows: pd.DataFrame,
    winner_boat: int,
) -> list[tuple[int, float]]:
    """Predict 2nd place probabilities for a single race.

    Returns sorted list of (boat_number, probability) excluding winner.
    """
    candidates = race_rows[race_rows["boat_number"] != winner_boat].copy()
    if len(candidates) == 0:
        return []

    available = [c for c in SECOND_FEATURE_COLS if c in candidates.columns]
    X = candidates[available].fillna(0)

    # Add course distance from winner
    winner_row = race_rows[race_rows["boat_number"] == winner_boat]
    if len(winner_row) > 0:
        winner_course = winner_row.iloc[0]["course_number"]
        X["course_distance_from_winner"] = (
            candidates["course_number"] - winner_course
        ).abs()
    else:
        X["course_distance_from_winner"] = 0

    probs = model.predict_proba(X)[:, 1]
    results = list(zip(candidates["boat_number"].astype(int).values, probs))
    return sorted(results, key=lambda x: -x[1])


def main():
    print("Loading features...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(DEFAULT_DB_PATH)

    # Load odds
    conn = get_connection(DEFAULT_DB_PATH)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '2連単'"
    ).fetchall()
    exacta_odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    rows_t = conn.execute(
        "SELECT race_id, CAST(combination AS INTEGER), odds "
        "FROM db.race_odds WHERE bet_type = '単勝'"
    ).fetchall()
    tansho_odds = {(int(r[0]), int(r[1])): float(r[2]) for r in rows_t}
    conn.close()

    # Build finish map
    finish_map = {}
    for _, row in (
        df[["race_id", "boat_number", "finish_position"]].drop_duplicates().iterrows()
    ):
        if pd.notna(row["finish_position"]):
            finish_map[(int(row["race_id"]), int(row["boat_number"]))] = int(
                row["finish_position"]
            )

    # Split
    from_date = "2026-02-01"
    to_date = "2026-04-04"
    train_df = df[df["race_date"] < from_date]
    test_df = df[(df["race_date"] >= from_date) & (df["race_date"] < to_date)]

    # Val split for boat1 + LambdaRank
    dates = sorted(train_df["race_date"].unique())
    val_start = dates[max(0, len(dates) - 60)]
    train_early = train_df[train_df["race_date"] < val_start]
    train_late = train_df[train_df["race_date"] >= val_start]

    # Train boat1 binary model
    print("Training boat1 model...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        X_b1_train, y_b1_train, _ = reshape_to_boat1(train_early)
        X_b1_val, y_b1_val, _ = reshape_to_boat1(train_late)
        X_b1_test, _, meta_b1_test = reshape_to_boat1(test_df)
        b1_model, _ = train_boat1_model(
            X_b1_train, y_b1_train, X_b1_val, y_b1_val
        )

    # Train LambdaRank
    print("Training LambdaRank...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        X_r_train, y_r_train, m_r_train = prepare_feature_matrix(train_early)
        X_r_val, y_r_val, m_r_val = prepare_feature_matrix(train_late)
        X_r_test, _, meta_r_test = prepare_feature_matrix(test_df)
        rank_model, _ = train_model(
            X_r_train, y_r_train, m_r_train,
            X_r_val, y_r_val, m_r_val,
            early_stopping_rounds=50,
        )

    # Train conditional 2nd-place models
    print("Training 2nd-place models...", file=sys.stderr)
    second_models = train_second_place_models(train_df)
    print(f"  Trained models for winners: {list(second_models.keys())}", file=sys.stderr)

    # Predict with boat1 + LambdaRank
    b1_probs = b1_model.predict_proba(X_b1_test)[:, 1]
    rank_scores = rank_model.predict(X_r_test)

    n_races = len(X_r_test) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_r_test["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta_r_test["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    exp_scores = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    b1_race_to_idx = {
        rid: i for i, rid in enumerate(meta_b1_test["race_id"].values)
    }

    # Index test_df by race_id for quick lookup
    test_by_race = {rid: group for rid, group in test_df.groupby("race_id")}

    # Evaluate strategies
    print("\nEvaluating...", file=sys.stderr)

    strategies = {
        "tansho_af": {"bets": 0, "wins": 0, "payout": 0.0},
        "exacta_1pt_model": {"bets": 0, "wins": 0, "payout": 0.0},
        "exacta_2pt_model": {"bets": 0, "wins": 0, "payout": 0.0},
        "exacta_1pt_naive": {"bets": 0, "wins": 0, "payout": 0.0},
    }

    days = 63  # ~2 months test period

    for race_idx in range(n_races):
        rid = int(race_ids[race_idx])
        b1_idx = b1_race_to_idx.get(rid)
        if b1_idx is None:
            continue

        b1_prob = float(b1_probs[b1_idx])
        if b1_prob >= B1_THRESHOLD:
            continue

        # LambdaRank: top non-boat1 pick
        top_rank = 0
        winner_pick = int(top_boats[race_idx, top_rank])
        if winner_pick == 1:
            top_rank = 1
            winner_pick = int(top_boats[race_idx, top_rank])

        # Winner prob from softmax
        boat_idx = np.where(boats_2d[race_idx] == winner_pick)[0]
        if len(boat_idx) == 0:
            continue
        winner_prob = float(rank_probs[race_idx, boat_idx[0]])

        actual_winner = finish_map.get((rid, winner_pick))
        winner_won = actual_winner == 1

        # --- Tansho (baseline) ---
        t_odds = tansho_odds.get((rid, winner_pick))
        if t_odds and t_odds > 0:
            ev = winner_prob * t_odds - 1
            if ev >= 0:
                strategies["tansho_af"]["bets"] += 1
                if winner_won:
                    strategies["tansho_af"]["wins"] += 1
                    strategies["tansho_af"]["payout"] += t_odds

        # --- 2nd place prediction ---
        second_model = second_models.get(winner_pick)
        if second_model is None:
            continue

        race_rows = test_by_race.get(rid)
        if race_rows is None:
            continue

        second_preds = predict_second_place(second_model, race_rows, winner_pick)
        if not second_preds:
            continue

        # Top 1 pick for 2nd
        second_pick_1 = int(second_preds[0][0])
        second_prob_1 = second_preds[0][1]

        # Top 2 picks for 2nd
        second_pick_2 = int(second_preds[1][0]) if len(second_preds) > 1 else None
        second_prob_2 = second_preds[1][1] if len(second_preds) > 1 else 0

        # Exacta 1pt: winner → top1_2nd
        combo1 = f"{winner_pick}-{second_pick_1}"
        e_odds1 = exacta_odds.get((rid, combo1))
        if e_odds1 and e_odds1 > 0:
            exacta_prob = winner_prob * second_prob_1
            ev1 = exacta_prob * e_odds1 - 1
            if ev1 >= 0:
                strategies["exacta_1pt_model"]["bets"] += 1
                if winner_won and finish_map.get((rid, second_pick_1)) == 2:
                    strategies["exacta_1pt_model"]["wins"] += 1
                    strategies["exacta_1pt_model"]["payout"] += e_odds1

        # Exacta 2pt: winner → top1_2nd AND winner → top2_2nd
        for pick, prob in [(second_pick_1, second_prob_1), (second_pick_2, second_prob_2)]:
            if pick is None:
                continue
            combo = f"{winner_pick}-{pick}"
            e_odds = exacta_odds.get((rid, combo))
            if e_odds and e_odds > 0:
                exacta_prob = winner_prob * prob
                ev = exacta_prob * e_odds - 1
                if ev >= 0:
                    strategies["exacta_2pt_model"]["bets"] += 1
                    if winner_won and finish_map.get((rid, pick)) == 2:
                        strategies["exacta_2pt_model"]["wins"] += 1
                        strategies["exacta_2pt_model"]["payout"] += e_odds

        # Naive exacta: winner → boat1 (always)
        combo_naive = f"{winner_pick}-1"
        e_odds_naive = exacta_odds.get((rid, combo_naive))
        if e_odds_naive and e_odds_naive > 0:
            # Use historical 40% rate for boat1 2nd
            naive_prob = winner_prob * 0.40
            ev_naive = naive_prob * e_odds_naive - 1
            if ev_naive >= 0:
                strategies["exacta_1pt_naive"]["bets"] += 1
                if winner_won and finish_map.get((rid, 1)) == 2:
                    strategies["exacta_1pt_naive"]["wins"] += 1
                    strategies["exacta_1pt_naive"]["payout"] += e_odds_naive

    # Results
    print()
    print(f"Test: {from_date} ~ {to_date} ({n_races} races, {days} days)")
    print(f"Anti-favorite filter: b1_prob < {B1_THRESHOLD:.0%}, EV >= 0")
    print()
    print(f"{'Strategy':<25} {'Bets':>5} {'B/d':>5} {'Wins':>4} {'Hit%':>6} {'ROI':>6} {'AvgW':>6} {'P/L_mo':>10}")
    print("-" * 80)

    for name, s in strategies.items():
        if s["bets"] > 0:
            roi = s["payout"] / s["bets"]
            hr = s["wins"] / s["bets"]
            avg_w = s["payout"] / s["wins"] if s["wins"] > 0 else 0
            bpd = s["bets"] / days
            # Monthly P/L at ¥500/bet (reasonable for exacta pool)
            monthly_pl = (roi - 1) * s["bets"] / days * 30 * 500
            print(
                f"  {name:<23} {s['bets']:>5} {bpd:>5.1f} {s['wins']:>4} "
                f"{hr:>5.1%} {roi:>5.0%} {avg_w:>5.1f}x "
                f"¥{monthly_pl:>+9,.0f}"
            )
        else:
            print(f"  {name:<23} no qualifying bets")


if __name__ == "__main__":
    main()
