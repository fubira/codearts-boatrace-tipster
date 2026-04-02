"""Train binary classifiers for specific 2連単 patterns and evaluate EV strategy.

Each pattern (e.g., "2-3") gets its own binary classifier predicting
whether that exact combination hits. EV = model_prob × odds - 1.

Usage:
    uv run --directory ml python -m scripts.train_exacta_binary
"""

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.evaluate import _load_payouts
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import time_series_split

DB_PATH = DEFAULT_DB_PATH
FIELD_SIZE = 6

# Top 2連単 patterns to model (by overall frequency)
TARGET_PATTERNS = [
    "1-2", "1-3", "1-4", "1-5",   # boat 1 wins
    "2-1", "3-1", "4-1", "5-1",   # boat 1 is 2nd
    "2-3", "3-2", "4-2", "2-4",   # boat 1 outside top 2
    "3-4", "4-3", "4-5", "3-5",
]

MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 100,  # conservative: low base rate targets
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "max_depth": 5,
    "random_state": 42,
    "verbose": -1,
}


def _build_race_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build race-level features with per-course boat data.

    Returns (features_df, meta_df) with 1 row per race.
    Features include each course's boat stats and race conditions.
    """
    # Columns to extract per course
    per_boat_cols = [
        "national_win_rate", "national_top2_rate",
        "local_win_rate", "motor_top3_rate",
        "exhibition_time", "exhibition_st",
        "racer_course_win_rate", "racer_course_top2_rate",
        "recent_avg_position", "racer_class_code", "racer_weight",
        "average_st", "st_stability",
        "rolling_st_mean", "rolling_win_rate", "rolling_avg_position",
        "tourn_exhibition_delta", "tourn_st_delta", "tourn_avg_position",
        "self_exhibition_delta",
    ]

    # Race-level columns
    race_cols = [
        "stadium_id", "race_grade_code", "weather_code",
        "wind_speed", "wave_height",
        "has_front_taking",
    ]

    n_races = len(df) // FIELD_SIZE
    race_ids = df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    race_dates = df["race_date"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    courses = df["course_number"].values.reshape(n_races, FIELD_SIZE)

    result = pd.DataFrame({"race_id": race_ids})

    # Per-course features: for each course position 1-6, extract the boat's stats
    for col in per_boat_cols:
        if col not in df.columns:
            continue
        vals = df[col].values.reshape(n_races, FIELD_SIZE)
        for course in range(1, 7):
            # Find which column index has this course number in each race
            feature_vals = np.full(n_races, np.nan)
            for i in range(n_races):
                idx = np.where(courses[i] == course)[0]
                if len(idx) > 0:
                    feature_vals[i] = vals[i, idx[0]]
            result[f"c{course}_{col}"] = feature_vals

    # Race-level features
    for col in race_cols:
        if col in df.columns:
            result[col] = df[col].values.reshape(n_races, FIELD_SIZE)[:, 0]

    # Derived: pairwise gaps for key features
    for col in ["national_win_rate", "exhibition_time"]:
        if col not in df.columns:
            continue
        for c1 in range(1, 5):
            for c2 in range(c1 + 1, min(c1 + 3, 7)):
                result[f"gap_{col}_c{c1}_c{c2}"] = (
                    result[f"c{c1}_{col}"] - result[f"c{c2}_{col}"]
                )

    meta = pd.DataFrame({"race_id": race_ids, "race_date": race_dates})

    # Fill categoricals
    for col in ["stadium_id", "race_grade_code", "weather_code"]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    # Drop race_id from features
    feature_cols = [c for c in result.columns if c != "race_id"]
    X = result[feature_cols]

    return X, meta


def _build_target(df: pd.DataFrame, pattern: str) -> pd.Series:
    """Build binary target for a specific 2連単 pattern (e.g., '2-3').

    Pattern is in boat_number format (matching the combination in race_odds).
    """
    c1_bn, c2_bn = int(pattern.split("-")[0]), int(pattern.split("-")[1])

    n_races = len(df) // FIELD_SIZE
    boats = df["boat_number"].values.reshape(n_races, FIELD_SIZE)
    pos = df["finish_position"].values.reshape(n_races, FIELD_SIZE)

    target = np.zeros(n_races, dtype=int)
    for i in range(n_races):
        first_idx = np.where(pos[i] == 1)[0]
        second_idx = np.where(pos[i] == 2)[0]
        if len(first_idx) > 0 and len(second_idx) > 0:
            if boats[i, first_idx[0]] == c1_bn and boats[i, second_idx[0]] == c2_bn:
                target[i] = 1

    return pd.Series(target)


def _load_exacta_odds(race_ids: np.ndarray) -> dict[int, dict[str, float]]:
    """Load 2連単 odds: {race_id: {combination: odds}}."""
    conn = get_connection(DB_PATH)
    id_list = ",".join(str(int(r)) for r in race_ids)
    rows = conn.execute(
        f"SELECT race_id, combination, odds "
        f"FROM db.race_odds WHERE bet_type = '2連単' "
        f"AND race_id IN ({id_list})"
    ).fetchall()
    conn.close()

    result: dict[int, dict[str, float]] = {}
    for race_id, combo, odds in rows:
        result.setdefault(int(race_id), {})[combo] = odds
    return result


def main():
    t0 = time.time()
    print("Building features...")
    df = build_features_df(DB_PATH)
    n_races = len(df) // FIELD_SIZE
    print(f"  {n_races} races")

    print("Building race-level features...")
    X, meta = _build_race_features(df)
    print(f"  {X.shape[1]} features")

    # Build targets for all patterns
    print("Building targets...")
    targets = {}
    for pattern in TARGET_PATTERNS:
        targets[pattern] = _build_target(df, pattern)
        rate = targets[pattern].mean()
        print(f"  {pattern}: {rate:.1%}")

    # Split
    # Use a dummy y for splitting (splits are date-based, y doesn't matter)
    splits = time_series_split(X, targets["1-2"], meta)
    for name, data in splits.items():
        print(f"  {name}: {len(data['X'])} races")

    # Load odds and payouts for test set
    test_race_ids = splits["test"]["meta"]["race_id"].values
    print("\nLoading odds and payouts...")
    exacta_odds = _load_exacta_odds(test_race_ids)
    payouts_db = _load_payouts(DB_PATH, test_race_ids)
    print(f"  Odds: {len(exacta_odds)} races, Payouts: {len(payouts_db)} races")

    # Train and evaluate each pattern
    print("\n" + "=" * 90)
    print("2連単 Pattern Binary Classifiers + EV Strategy")
    print("=" * 90)

    all_ev_bets = []  # collect all positive EV bets across patterns

    for pattern in TARGET_PATTERNS:
        y_all = targets[pattern]

        # Split targets
        train_mask = splits["train"]["X"].index
        val_mask = splits["val"]["X"].index
        test_mask = splits["test"]["X"].index

        y_train = y_all.iloc[train_mask]
        y_val = y_all.iloc[val_mask]
        y_test = y_all.iloc[test_mask]

        base_rate = y_test.mean()

        # Train
        model = LGBMClassifier(**MODEL_PARAMS)
        model.fit(
            splits["train"]["X"], y_train,
            eval_set=[(splits["val"]["X"], y_val)],
            callbacks=[early_stopping(50, verbose=False)],
        )

        # Predict
        y_prob = model.predict_proba(splits["test"]["X"])[:, 1]
        y_true = y_test.values

        # AUC (skip if only one class in test)
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.0

        # EV analysis with actual odds
        total_bet = {t: 0 for t in [0, 10, 20, 30]}
        total_payout = {t: 0 for t in [0, 10, 20, 30]}
        n_bets = {t: 0 for t in [0, 10, 20, 30]}
        hits = {t: 0 for t in [0, 10, 20, 30]}

        for i, rid in enumerate(test_race_ids):
            rid = int(rid)
            race_odds = exacta_odds.get(rid)
            if not race_odds:
                continue
            odds = race_odds.get(pattern)
            if odds is None:
                continue

            ev = y_prob[i] * odds * 100 - 100

            for threshold in [0, 10, 20, 30]:
                if ev >= threshold:
                    total_bet[threshold] += 100
                    n_bets[threshold] += 1
                    rp = payouts_db.get(rid)
                    if rp:
                        exacta_payouts = rp.get("2連単")
                        if exacta_payouts:
                            p = exacta_payouts.get(pattern)
                            if p:
                                total_payout[threshold] += p
                                hits[threshold] += 1

                    if threshold == 0:
                        all_ev_bets.append((rid, pattern, ev, y_prob[i], odds))

        # Print results
        ev0_roi = total_payout[0] / total_bet[0] if total_bet[0] > 0 else 0
        ev0_bets = n_bets[0]
        ev0_hit = hits[0] / n_bets[0] if n_bets[0] > 0 else 0
        roi_str = f"{ev0_roi:.0%}" if ev0_bets > 0 else "N/A"

        print(f"\n  {pattern}  base={base_rate:.1%}  AUC={auc:.3f}  |  EV≥0: {ev0_bets:>5d} bets  hit={ev0_hit:.1%}  ROI={roi_str}")

        for threshold in [0, 10, 20, 30]:
            if total_bet[threshold] > 0:
                roi = total_payout[threshold] / total_bet[threshold]
                hr = hits[threshold] / n_bets[threshold]
                profit = total_payout[threshold] - total_bet[threshold]
                profit_str = f"+¥{profit:,}" if profit >= 0 else f"-¥{abs(profit):,}"
                print(f"    EV≥{threshold:+3d}: {n_bets[threshold]:>5d} bets  hit={hr:.1%}  ROI={roi:.1%}  {profit_str}")

    # Combined portfolio: all positive EV bets across all patterns
    print("\n" + "=" * 90)
    print("Combined Portfolio (all patterns, EV≥0)")
    print("=" * 90)

    total_bet = 0
    total_payout = 0
    total_hits = 0
    for rid, pattern, ev, prob, odds in all_ev_bets:
        rp = payouts_db.get(rid)
        if not rp:
            continue
        exacta_payouts = rp.get("2連単")
        if not exacta_payouts:
            continue
        total_bet += 100
        p = exacta_payouts.get(pattern)
        if p:
            total_payout += p
            total_hits += 1

    if total_bet > 0:
        n = total_bet // 100
        roi = total_payout / total_bet
        hit_rate = total_hits / n
        profit = total_payout - total_bet
        profit_str = f"+¥{profit:,}" if profit >= 0 else f"-¥{abs(profit):,}"
        print(f"  Total bets: {n}")
        print(f"  Hit rate: {hit_rate:.2%}")
        print(f"  ROI: {roi:.1%}")
        print(f"  Profit: {profit_str}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
