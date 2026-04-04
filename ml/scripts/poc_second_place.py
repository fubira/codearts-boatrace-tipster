"""PoC: Conditional 2nd-place prediction given winner.

For each winner X (2-6), train a binary classifier per candidate boat
to predict P(boat Y finishes 2nd | boat X wins).

Usage:
    uv run --directory ml python -m scripts.poc_second_place
"""

import contextlib
import io
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df

FIELD_SIZE = 6


def build_second_place_features(
    race_df: pd.DataFrame,
    winner_boat: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build features for predicting 2nd place given winner_boat won.

    Filter to races where winner_boat actually finished 1st,
    then for each remaining boat, create features and target (is_2nd).
    """
    # Filter to races where winner_boat actually won
    winner_rows = race_df[
        (race_df["boat_number"] == winner_boat)
        & (race_df["finish_position"] == 1)
    ]["race_id"].unique()

    subset = race_df[
        (race_df["race_id"].isin(winner_rows))
        & (race_df["boat_number"] != winner_boat)
    ].copy()

    if len(subset) == 0:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    # Target: did this boat finish 2nd?
    y = (subset["finish_position"] == 2).astype(int)

    # Features: boat's own stats + relative to race + winner's stats
    feature_cols = [
        # Boat's own strength
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
        # Rolling/tournament features
        "racer_course_win_rate",
        "racer_course_top2_rate",
        "stadium_course_win_rate",
        "recent_avg_position",
        # Relative features (z-scored within race)
        "rel_national_win_rate",
        "rel_exhibition_time",
        "rel_exhibition_st",
        # Race context
        "wind_speed",
        "wave_height",
    ]

    # Only keep columns that exist
    available = [c for c in feature_cols if c in subset.columns]
    X = subset[available].copy()

    # Add: distance from winner's course (proximity matters for 2nd place)
    winner_courses = race_df[
        (race_df["race_id"].isin(winner_rows))
        & (race_df["boat_number"] == winner_boat)
    ][["race_id", "course_number"]].rename(columns={"course_number": "winner_course"})

    subset_with_winner = subset.merge(winner_courses, on="race_id", how="left")
    X["course_distance_from_winner"] = (
        subset_with_winner["course_number"] - subset_with_winner["winner_course"]
    ).abs()

    meta = subset[["race_id", "boat_number", "race_date"]].copy()

    return X, y, meta


def evaluate_model(
    winner_boat: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Train and evaluate 2nd place predictor for a specific winner."""
    X_train, y_train, _ = build_second_place_features(train_df, winner_boat)
    X_test, y_test, meta_test = build_second_place_features(test_df, winner_boat)

    if len(X_train) == 0 or len(X_test) == 0:
        return {"winner": winner_boat, "error": "no data"}

    # Fill NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train LightGBM binary classifier
    model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
    )

    # Val split from training data
    train_dates = sorted(X_train.index.map(lambda i: train_df.iloc[i % len(train_df)]["race_date"] if i < len(train_df) else "").unique()) if False else None

    model.fit(X_train, y_train)

    # Predict
    probs = model.predict_proba(X_test)[:, 1]

    # AUC
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0

    # Per-race evaluation: pick the boat with highest prob as 2nd
    meta_test = meta_test.copy()
    meta_test["prob"] = probs
    meta_test["actual_2nd"] = y_test.values

    # Group by race, pick top prob
    correct = 0
    total = 0
    for race_id, group in meta_test.groupby("race_id"):
        top_pick = group.loc[group["prob"].idxmax()]
        total += 1
        if top_pick["actual_2nd"] == 1:
            correct += 1

    model_acc = correct / total if total > 0 else 0

    # Naive baseline: always pick the most common 2nd-place boat
    # (from training data distribution)
    _, y_train_full, meta_train = build_second_place_features(train_df, winner_boat)
    train_2nd = meta_train[y_train_full == 1]["boat_number"].value_counts()
    naive_pick = train_2nd.index[0] if len(train_2nd) > 0 else 1

    naive_correct = 0
    for race_id, group in meta_test.groupby("race_id"):
        actual = group[group["actual_2nd"] == 1]
        if len(actual) > 0 and actual.iloc[0]["boat_number"] == naive_pick:
            naive_correct += 1
    naive_acc = naive_correct / total if total > 0 else 0

    # Top-2 accuracy (does actual 2nd appear in model's top 2 picks?)
    top2_correct = 0
    for race_id, group in meta_test.groupby("race_id"):
        top2 = group.nlargest(2, "prob")["boat_number"].values
        actual = group[group["actual_2nd"] == 1]
        if len(actual) > 0 and actual.iloc[0]["boat_number"] in top2:
            top2_correct += 1
    top2_acc = top2_correct / total if total > 0 else 0

    # Feature importance
    imp = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: -x[1],
    )[:5]

    return {
        "winner": winner_boat,
        "n_races_test": total,
        "auc": round(auc, 4),
        "model_acc": round(model_acc, 4),
        "naive_acc": round(naive_acc, 4),
        "naive_pick": int(naive_pick),
        "lift": round(model_acc / naive_acc, 2) if naive_acc > 0 else 0,
        "top2_acc": round(top2_acc, 4),
        "top5_features": [(f, int(v)) for f, v in imp],
    }


def main():
    print("Loading features...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(DEFAULT_DB_PATH)

    # Train/test split
    train_df = df[df["race_date"] < "2026-02-01"]
    test_df = df[(df["race_date"] >= "2026-02-01") & (df["race_date"] < "2026-04-04")]

    print(
        f"Train: {train_df['race_id'].nunique()} races, "
        f"Test: {test_df['race_id'].nunique()} races",
        file=sys.stderr,
    )

    print()
    print(f"{'Winner':>6} | {'Races':>5} | {'AUC':>6} | {'Model':>6} | {'Naive':>6} | {'Lift':>5} | {'Top2':>6} | Top Features")
    print("-" * 100)

    for winner in range(2, 7):
        result = evaluate_model(winner, train_df, test_df)
        if "error" in result:
            print(f"  {winner}号艇: {result['error']}")
            continue

        feats = ", ".join(f"{f}({v})" for f, v in result["top5_features"])
        print(
            f"  {winner}号艇 | {result['n_races_test']:>5} | "
            f"{result['auc']:.4f} | {result['model_acc']:>5.1%} | "
            f"{result['naive_acc']:>5.1%} | {result['lift']:>4.1f}x | "
            f"{result['top2_acc']:>5.1%} | {feats}"
        )

    # Combined: overall accuracy across all winner types
    print()
    print("Combined evaluation (anti-favorite pipeline):")
    print("  = Use binary model to identify boat1-loses races")
    print("  = Use LambdaRank to pick winner")
    print("  = Use conditional model to pick 2nd place")
    print("  = Bet exacta (winner → 2nd)")


if __name__ == "__main__":
    main()
