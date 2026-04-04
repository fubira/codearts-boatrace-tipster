"""Optuna hyperparameter tuning for trifecta X-noB1-noB1 strategy.

Optimizes LambdaRank + boat1 binary model jointly for 3連単 ROI.

Usage:
    uv run --directory ml python -m scripts.tune_trifecta --trials 100
    uv run --directory ml python -m scripts.tune_trifecta --trials 100 --seed 123

Tuned parameters:
    - LambdaRank: num_leaves, max_depth, learning_rate, n_estimators, etc.
    - LambdaRank: relevance_scheme (linear, top_heavy, win_only, podium)
    - Strategy: b1_threshold, ev_threshold
"""

import argparse
import contextlib
import io
import sys
import time
from collections import defaultdict

import numpy as np
import optuna
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits

FIELD_SIZE = 6


def _load_odds(db_path: str) -> tuple[dict, dict]:
    """Load trifecta odds and trifecta-implied win probabilities."""
    conn = get_connection(db_path)

    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '3連単'"
    ).fetchall()
    trifecta_odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    # Build trifecta-implied win probability
    tri_win_prob: dict[tuple[int, int], float] = defaultdict(float)
    for r in rows:
        rid, combo, odds = int(r[0]), r[1], float(r[2])
        if odds <= 0:
            continue
        first_boat = int(combo.split("-")[0])
        tri_win_prob[(rid, first_boat)] += 0.75 / odds

    conn.close()
    return trifecta_odds, dict(tri_win_prob)


def evaluate_trifecta_strategy(
    b1_probs: np.ndarray,
    meta_b1: pd.DataFrame,
    rank_scores: np.ndarray,
    meta_rank: pd.DataFrame,
    finish_map: dict[tuple[int, int], int],
    trifecta_odds: dict[tuple[int, str], float],
    tri_win_prob: dict[tuple[int, int], float],
    b1_threshold: float = 0.40,
    ev_threshold: float = 0.30,
) -> dict:
    """Evaluate X-noB1-noB1 trifecta strategy on test data."""
    n_races = len(rank_scores) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_rank["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta_rank["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    b1_map = {rid: i for i, rid in enumerate(meta_b1["race_id"].values)}

    total_races = 0
    total_tickets = 0
    total_wins = 0
    total_payout = 0.0

    for ri in range(n_races):
        rid = int(race_ids[ri])
        bi = b1_map.get(rid)
        if bi is None:
            continue
        if float(b1_probs[bi]) >= b1_threshold:
            continue

        wp = int(top_boats[ri, 0])
        if wp == 1:
            wp = int(top_boats[ri, 1])

        bidx = np.where(boats_2d[ri] == wp)[0]
        if len(bidx) == 0:
            continue
        wprob = float(rank_probs[ri, bidx[0]])

        # EV filter using trifecta-implied market probability
        mkt_prob = tri_win_prob.get((rid, wp), 0)
        if mkt_prob <= 0:
            continue
        ev = wprob / mkt_prob * 0.75 - 1
        if ev < ev_threshold:
            continue

        # Build X-noB1-noB1 tickets
        excluded = {wp, 1}
        flow = [int(b) for b in boats_2d[ri] if int(b) not in excluded]
        tkts = []
        for b2 in flow:
            for b3 in flow:
                if b2 != b3:
                    c = f"{wp}-{b2}-{b3}"
                    if (rid, c) in trifecta_odds:
                        tkts.append(c)
        if not tkts:
            continue

        total_races += 1
        total_tickets += len(tkts)

        # Check result
        a2 = a3 = None
        for b in range(1, 7):
            fp = finish_map.get((rid, b))
            if fp == 2:
                a2 = b
            if fp == 3:
                a3 = b

        if finish_map.get((rid, wp)) == 1 and a2 and a3:
            hc = f"{wp}-{a2}-{a3}"
            if hc in tkts:
                ho = trifecta_odds.get((rid, hc))
                if ho:
                    total_wins += 1
                    total_payout += ho

    roi = total_payout / total_tickets if total_tickets > 0 else 0
    return {
        "races": total_races,
        "tickets": total_tickets,
        "wins": total_wins,
        "payout": total_payout,
        "roi": roi,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune trifecta strategy with Optuna")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    print(f"Trials: {args.trials}, Folds: {args.n_folds}, Seed: {args.seed}")
    t0 = time.time()

    # Load data
    print("Loading features...")
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    print("Loading odds...")
    trifecta_odds, tri_win_prob = _load_odds(args.db_path)

    # Build finish map
    finish_map: dict[tuple[int, int], int] = {}
    for _, row in (
        df[["race_id", "boat_number", "finish_position"]].drop_duplicates().iterrows()
    ):
        if pd.notna(row["finish_position"]):
            finish_map[(int(row["race_id"]), int(row["boat_number"]))] = int(
                row["finish_position"]
            )

    # Prepare ranking features for fold generation
    X_rank, y_rank, meta_rank = prepare_feature_matrix(df)

    # Generate WF-CV folds for ranking model
    folds = walk_forward_splits(
        X_rank,
        y_rank,
        meta_rank,
        n_folds=args.n_folds,
        fold_months=args.fold_months,
    )
    if not folds:
        print("ERROR: No valid folds")
        sys.exit(1)

    print(f"Folds: {len(folds)}")
    for i, fold in enumerate(folds):
        p = fold["period"]
        print(f"  Fold {i+1}: test={p['test']}")

    # Pre-compute boat1 features for each fold's test set
    # (boat1 model is fixed — not tuned here, only LambdaRank + strategy params)
    print("Pre-training boat1 models per fold...")
    fold_b1_data = []
    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]

        train_fold = df[df["race_date"] < test_from]
        test_fold = df[
            (df["race_date"] >= test_from) & (df["race_date"] < test_to)
        ]

        dates_t = sorted(train_fold["race_date"].unique())
        val_start = dates_t[max(0, len(dates_t) - 60)]
        train_early = train_fold[train_fold["race_date"] < val_start]
        train_late = train_fold[train_fold["race_date"] >= val_start]

        with contextlib.redirect_stdout(io.StringIO()):
            X_b1_tr, y_b1_tr, _ = reshape_to_boat1(train_early)
            X_b1_v, y_b1_v, _ = reshape_to_boat1(train_late)
            X_b1_te, _, meta_b1_te = reshape_to_boat1(test_fold)
            b1_model, _ = train_boat1_model(
                X_b1_tr, y_b1_tr, X_b1_v, y_b1_v
            )

        b1_probs = b1_model.predict_proba(X_b1_te)[:, 1]
        fold_b1_data.append({
            "b1_probs": b1_probs,
            "meta_b1": meta_b1_te,
        })
        print(f"  Fold {i+1}: boat1 AUC computed, n_test={len(X_b1_te)}")

    print(f"Setup done in {time.time() - t0:.1f}s\n")

    def objective(trial: optuna.Trial) -> float:
        # LambdaRank hyperparameters
        rank_params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        n_estimators = trial.suggest_int("n_estimators", 100, 1500)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        relevance = trial.suggest_categorical(
            "relevance", ["linear", "top_heavy", "win_only", "podium"]
        )

        # Strategy parameters
        b1_threshold = trial.suggest_float("b1_threshold", 0.30, 0.55)
        ev_threshold = trial.suggest_float("ev_threshold", -0.1, 0.5)

        rois = []
        total_races = 0
        for i, fold in enumerate(folds):
            with contextlib.redirect_stdout(io.StringIO()):
                rank_model, _ = train_model(
                    fold["train"]["X"],
                    fold["train"]["y"],
                    fold["train"]["meta"],
                    fold["val"]["X"],
                    fold["val"]["y"],
                    fold["val"]["meta"],
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    relevance_scheme=relevance,
                    extra_params=rank_params,
                    early_stopping_rounds=50,
                )

            rank_scores = rank_model.predict(fold["test"]["X"])

            result = evaluate_trifecta_strategy(
                b1_probs=fold_b1_data[i]["b1_probs"],
                meta_b1=fold_b1_data[i]["meta_b1"],
                rank_scores=rank_scores,
                meta_rank=fold["test"]["meta"],
                finish_map=finish_map,
                trifecta_odds=trifecta_odds,
                tri_win_prob=tri_win_prob,
                b1_threshold=b1_threshold,
                ev_threshold=ev_threshold,
            )
            rois.append(result["roi"])
            total_races += result["races"]

        mean_roi = float(np.mean(rois))
        std_roi = float(np.std(rois))
        min_roi = float(np.min(rois))

        # Require minimum bet volume (too few races = unreliable Sharpe)
        if total_races < 50:
            return -999.0

        sharpe = (mean_roi - 1.0) / std_roi if std_roi > 0 else 0

        trial.set_user_attr("mean_roi", mean_roi)
        trial.set_user_attr("roi_std", std_roi)
        trial.set_user_attr("roi_min", min_roi)
        trial.set_user_attr("roi_max", float(np.max(rois)))
        trial.set_user_attr("rois", [round(r, 4) for r in rois])
        trial.set_user_attr("sharpe", round(sharpe, 3))
        trial.set_user_attr("total_races", total_races)

        # Optimize Sharpe ratio (stability-adjusted ROI)
        return sharpe

    study = optuna.create_study(
        direction="maximize",
        study_name="trifecta-x-nob1-nob1",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 70)
    print("Optuna Search Complete — Trifecta X-noB1-noB1")
    print("=" * 70)
    print(f"Best Sharpe: {study.best_value:.3f}")
    bp = study.best_params
    print(f"Best params:")
    for k, v in sorted(bp.items()):
        print(f"  {k}: {v}")

    ba = study.best_trial.user_attrs
    print(f"\nBest trial metrics:")
    print(f"  Mean ROI: {ba['mean_roi']:.1%}")
    print(f"  ROI std:  {ba['roi_std']:.1%}")
    print(f"  ROI min:  {ba['roi_min']:.1%}")
    print(f"  ROI max:  {ba['roi_max']:.1%}")
    print(f"  Sharpe:   {ba['sharpe']:.3f}")
    print(f"  Folds:    {ba['rois']}")

    # Top 10 trials
    print(f"\nTop 10 trials (by Sharpe):")
    trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value else -999,
        reverse=True,
    )
    for t in trials[:10]:
        if t.value is None:
            continue
        ua = t.user_attrs
        rel = t.params.get("relevance", "?")
        b1t = t.params.get("b1_threshold", 0)
        evt = t.params.get("ev_threshold", 0)
        races = ua.get("total_races", "?")
        print(
            f"  #{t.number:>3}: Sharpe={t.value:.2f} "
            f"ROI={ua['mean_roi']:.0%}±{ua['roi_std']:.0%} "
            f"[{ua['roi_min']:.0%}-{ua['roi_max']:.0%}] "
            f"n={races} rel={rel} b1<{b1t:.0%} ev>{evt:+.0%}"
        )

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
