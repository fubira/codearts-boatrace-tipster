"""Optuna tuning for P2 trifecta strategy.

Strategy: B1 prediction + gap23 filter + 3連単 EV adaptive tickets.
Model: Non-odds features LambdaRank (no boat1 binary classifier needed).

Usage:
    uv run --directory ml python -m scripts.tune_p2 --trials 50
    uv run --directory ml python -m scripts.tune_p2 --trials 100 --seed 123
    uv run --directory ml python -m scripts.tune_p2 --trials 100 --fix-thresholds "gap23=0.15,ev=0.1"
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits

FIELD_SIZE = 6

# Non-odds features
FEATURES = [
    "exhibition_time", "rel_exhibition_time", "rel_exhibition_st",
    "kado_x_exhibition",
    "bc_lap_zscore", "bc_turn_zscore", "bc_straight_zscore", "bc_slit_zscore",
    "tourn_exhibition_delta", "tourn_st_delta", "tourn_avg_position",
    "racer_course_win_rate", "racer_course_top2_rate",
    "recent_avg_position", "stadium_course_win_rate",
    "wind_speed_x_boat", "rolling_position_alpha", "position_alpha",
    "avg_course_diff", "course_taking_rate_at_boat", "race_min_avg_course_diff",
]


def _load_trifecta_odds(db_path: str) -> dict:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '3連単'"
    ).fetchall()
    conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def evaluate_p2_strategy(
    rank_scores: np.ndarray,
    meta_rank: pd.DataFrame,
    trifecta_odds: dict,
    gap23_threshold: float,
    ev_threshold: float,
    *,
    top3_conc_threshold: float = 0.0,
    per_race: bool = False,
) -> dict | list[dict]:
    """Evaluate P2 adaptive strategy.

    For each race:
    1. Check if top-1 prediction is boat 1
    2. Check if top3_concentration >= threshold (rank2+3 separated from rest)
    3. Check if gap23 >= threshold
    4. For P2 tickets (1-2-3 and 1-3-2), compute 3連単 EV
    5. Buy only tickets with EV >= threshold

    Returns summary dict or per-race list.
    """
    n_races = len(rank_scores) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_rank["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta_rank["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    y_2d = meta_rank["finish_position"].values.reshape(n_races, FIELD_SIZE) if "finish_position" in meta_rank.columns else None

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)

    # Softmax probabilities
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    total_cost = 0.0
    total_payout = 0.0
    total_races = 0
    total_wins = 0
    total_tickets = 0
    results: list[dict] = []

    for i in range(n_races):
        rid = int(race_ids[i])

        # Filter 1: top-1 must be boat 1
        if top_boats[i, 0] != 1:
            continue

        p1_prob = model_probs[i, pred_order[i, 0]]
        p2_prob = model_probs[i, pred_order[i, 1]]
        p3_prob = model_probs[i, pred_order[i, 2]]

        # Filter 2: top3_concentration threshold
        top3_conc = (p2_prob + p3_prob) / (1 - p1_prob + 1e-10)
        if top3_conc < top3_conc_threshold:
            continue

        # Filter 3: gap23 threshold
        gap23 = p2_prob - p3_prob
        if gap23 < gap23_threshold:
            continue

        # P2 tickets: 1-2-3 and 1-3-2
        r1, r2, r3 = top_boats[i, 0], top_boats[i, 1], top_boats[i, 2]
        i1, i2, i3 = pred_order[i, 0], pred_order[i, 1], pred_order[i, 2]

        tickets_bought = []
        for combo, prob_fn in [
            (f"{r1}-{r2}-{r3}", lambda: _trifecta_prob(model_probs[i], i1, i2, i3)),
            (f"{r1}-{r3}-{r2}", lambda: _trifecta_prob(model_probs[i], i1, i3, i2)),
        ]:
            mkt_odds = trifecta_odds.get((rid, combo))
            if not mkt_odds or mkt_odds <= 0:
                continue
            model_prob = prob_fn()
            mkt_prob = 1.0 / mkt_odds
            ev = model_prob / mkt_prob * 0.75 - 1
            if ev >= ev_threshold:
                tickets_bought.append(combo)

        if not tickets_bought:
            continue

        total_races += 1
        total_tickets += len(tickets_bought)
        total_cost += len(tickets_bought) * 100

        # Check results
        race_won = False
        race_payout = 0.0
        if y_2d is not None:
            actual_order = np.argsort(y_2d[i])
            a1, a2, a3 = boats_2d[i, actual_order[0]], boats_2d[i, actual_order[1]], boats_2d[i, actual_order[2]]
            hit = f"{int(a1)}-{int(a2)}-{int(a3)}"
            for t in tickets_bought:
                if t == hit:
                    odds = trifecta_odds.get((rid, t))
                    if odds and odds > 0:
                        race_won = True
                        race_payout = odds * 100
                        total_wins += 1
                        total_payout += race_payout
                    break

        if per_race:
            results.append({
                "race_id": rid,
                "tickets": len(tickets_bought),
                "cost": len(tickets_bought) * 100,
                "won": race_won,
                "payout": race_payout,
            })

    if per_race:
        return results

    roi = total_payout / total_cost if total_cost > 0 else 0
    return {
        "races": total_races,
        "tickets": total_tickets,
        "cost": total_cost,
        "wins": total_wins,
        "payout": total_payout,
        "roi": roi,
    }


def _trifecta_prob(probs_6, i1, i2, i3):
    """P(i1=1st, i2=2nd, i3=3rd) from softmax probabilities."""
    p1 = probs_6[i1]
    p2 = probs_6[i2]
    p3 = probs_6[i3]
    if p1 >= 1.0 or (p1 + p2) >= 1.0:
        return 0.0
    return p1 * (p2 / (1 - p1)) * (p3 / (1 - p1 - p2))


def main():
    parser = argparse.ArgumentParser(description="Tune P2 trifecta strategy with Optuna")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--objective", choices=["growth", "kelly"], default="growth",
                        help="Optimization objective: growth (daily compound, default) or kelly (log-ROI, favors high ROI)")
    parser.add_argument("--relevance", default=None,
                        help="Fix relevance scheme (linear/top_heavy/podium). If omitted, included in search space.")
    parser.add_argument("--fix-thresholds", default=None,
                        help="Fix strategy thresholds. Format: gap23=0.15,ev=0.1,top3_conc=0.7")
    args = parser.parse_args()

    # Parse fixed thresholds
    fixed_thresholds: dict[str, float] = {}
    if args.fix_thresholds:
        for pair in args.fix_thresholds.split(","):
            k, v = pair.strip().split("=")
            fixed_thresholds[k.strip()] = float(v.strip())
        print(f"Fixed thresholds: {fixed_thresholds}")

    print(f"Trials: {args.trials}, Folds: {args.n_folds}, Seed: {args.seed}")
    t0 = time.time()

    # Load data
    print("Loading features...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    # Verify all features exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        sys.exit(1)

    print("Loading 3連単 odds...", flush=True)
    trifecta_odds = _load_trifecta_odds(args.db_path)

    # Prepare data for WF-CV (use NON_ODDS features, not FEATURE_COLS)
    X = df[FEATURES].copy()
    y = df["finish_position"]
    # meta needs race_id, racer_id, race_date, boat_number, finish_position
    meta = df[["race_id", "racer_id", "race_date", "boat_number", "finish_position"]].copy()

    # Generate WF-CV folds
    folds = walk_forward_splits(X, y, meta, n_folds=args.n_folds, fold_months=args.fold_months)
    if not folds:
        print("ERROR: No valid folds")
        sys.exit(1)

    print(f"Folds: {len(folds)}")
    for i, fold in enumerate(folds):
        n_test = len(fold["test"]["X"]) // FIELD_SIZE
        print(f"  Fold {i+1}: test={fold['period']['test']} ({n_test}R)")

    print(f"Setup done in {time.time() - t0:.1f}s\n", flush=True)

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
        if args.relevance:
            relevance = args.relevance
        else:
            relevance = trial.suggest_categorical(
                "relevance", ["linear", "top_heavy", "podium"]
            )

        # Strategy thresholds
        if "gap23" in fixed_thresholds:
            gap23_threshold = fixed_thresholds["gap23"]
        else:
            gap23_threshold = trial.suggest_float("gap23_threshold", 0.0, 0.25)
        if "ev" in fixed_thresholds:
            ev_threshold = fixed_thresholds["ev"]
        else:
            ev_threshold = trial.suggest_float("ev_threshold", -0.3, 0.5)
        if "top3_conc" in fixed_thresholds:
            top3_conc_threshold = fixed_thresholds["top3_conc"]
        else:
            top3_conc_threshold = trial.suggest_float("top3_conc_threshold", 0.0, 0.85)

        fold_profits = []
        total_races = 0
        total_cost = 0.0
        total_payout = 0.0
        rois = []

        for fold in folds:
            train_X = fold["train"]["X"][FEATURES] if set(FEATURES).issubset(fold["train"]["X"].columns) else fold["train"]["X"]
            val_X = fold["val"]["X"][FEATURES] if set(FEATURES).issubset(fold["val"]["X"].columns) else fold["val"]["X"]

            with contextlib.redirect_stdout(io.StringIO()):
                rank_model, _ = train_model(
                    train_X, fold["train"]["y"], fold["train"]["meta"],
                    val_X, fold["val"]["y"], fold["val"]["meta"],
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    relevance_scheme=relevance,
                    extra_params=rank_params,
                    early_stopping_rounds=200,
                )

            test_X = fold["test"]["X"][FEATURES] if set(FEATURES).issubset(fold["test"]["X"].columns) else fold["test"]["X"]
            rank_scores = rank_model.predict(test_X)

            # Build meta with finish_position for evaluation
            test_meta = fold["test"]["meta"].copy()
            test_meta["finish_position"] = fold["test"]["y"].values

            result = evaluate_p2_strategy(
                rank_scores=rank_scores,
                meta_rank=test_meta,
                trifecta_odds=trifecta_odds,
                gap23_threshold=gap23_threshold,
                ev_threshold=ev_threshold,
                top3_conc_threshold=top3_conc_threshold,
            )

            fold_profit = result["payout"] - result["cost"]
            fold_profits.append(fold_profit)
            total_races += result["races"]
            total_cost += result["cost"]
            total_payout += result["payout"]
            rois.append(result["roi"] if result["cost"] > 0 else 0)

        # Require minimum volume
        if total_races < 20:
            return -999.0

        mean_roi = float(np.mean(rois)) if rois else 0
        profit = total_payout - total_cost

        # Growth: daily compound growth rate
        bankroll = 70000.0
        days_per_fold = args.fold_months * 30
        growth_rates = []
        for fp in fold_profits:
            daily_profit = fp / days_per_fold
            growth_rates.append(np.log(max(1 + daily_profit / bankroll, 1e-6)))
        growth = float(np.mean(growth_rates))

        # Kelly: mean of log(ROI) — rewards high ROI regardless of volume
        log_rois = [np.log(max(r, 1e-6)) for r in rois]
        kelly = float(np.mean(log_rois))

        trial.set_user_attr("mean_roi", round(mean_roi, 4))
        trial.set_user_attr("rois", [round(r, 4) for r in rois])
        trial.set_user_attr("total_races", total_races)
        trial.set_user_attr("profit", round(profit, 1))
        trial.set_user_attr("growth", round(growth, 6))
        trial.set_user_attr("kelly", round(kelly, 4))
        trial.set_user_attr("relevance", relevance)

        if args.objective == "kelly":
            return kelly
        return growth

    study = optuna.create_study(
        direction="maximize",
        study_name="p2-trifecta",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Results
    obj_label = {"growth": "Growth", "kelly": "Kelly"}[args.objective]
    print("\n" + "=" * 70)
    print(f"Optuna Search Complete — P2 Trifecta Strategy (objective: {args.objective})")
    print("=" * 70)
    print(f"Best {obj_label}: {study.best_value:.6f}")
    bp = study.best_params
    print(f"Best params:")
    for k, v in sorted(bp.items()):
        print(f"  {k}: {v}")

    ba = study.best_trial.user_attrs
    print(f"\nBest trial metrics:")
    print(f"  Mean ROI: {ba['mean_roi']:.1%}")
    print(f"  Profit:   {ba['profit']:+,.0f}円")
    print(f"  Races:    {ba['total_races']}")
    print(f"  Growth:   {ba['growth']:.6f}")
    print(f"  Kelly:    {ba.get('kelly', 'N/A')}")
    print(f"  Folds:    {ba['rois']}")

    # Top 10 trials
    print(f"\nTop 10 trials (by {obj_label}):")
    trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -999],
        key=lambda t: t.value,
        reverse=True,
    )
    for t in trials[:10]:
        ua = t.user_attrs
        rel = ua.get("relevance", "?")
        g23 = t.params.get("gap23_threshold", fixed_thresholds.get("gap23", "?"))
        evt = t.params.get("ev_threshold", fixed_thresholds.get("ev", "?"))
        t3c = t.params.get("top3_conc_threshold", fixed_thresholds.get("top3_conc", "?"))
        kelly_v = ua.get("kelly", "?")
        print(
            f"  #{t.number:>3}: growth={ua['growth']:.6f} kelly={kelly_v} ROI={ua['mean_roi']:.0%} "
            f"P/L={ua['profit']:+,.0f} n={ua['total_races']} "
            f"rel={rel} gap23={g23} ev={evt} conc={t3c}"
        )

    # Save best params
    meta_dir = "models/tune_result"
    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    from boatrace_tipster_ml.model import save_model_meta
    best_hp = {k: bp[k] for k in
               ["num_leaves", "max_depth", "min_child_samples",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
                "n_estimators", "learning_rate"]}
    best_hp["relevance_scheme"] = bp.get("relevance") or ba.get("relevance", "podium")
    strategy = {
        "type": "P2",
        "bet_pattern": "1-(2,3)-(2,3) adaptive",
        "gap23_threshold": bp.get("gap23_threshold", fixed_thresholds.get("gap23")),
        "ev_threshold": bp.get("ev_threshold", fixed_thresholds.get("ev")),
        "top3_conc_threshold": bp.get("top3_conc_threshold", fixed_thresholds.get("top3_conc")),
        "ev_basis": "3連単 odds per ticket",
        "features": "non_odds_21",
    }
    save_model_meta(
        meta_dir,
        feature_columns=FEATURES,
        hyperparameters=best_hp,
        training={
            "note": f"Optuna P2 {args.trials}t seed={args.seed} "
                    f"#{study.best_trial.number} (growth {study.best_value:.6f})",
        },
    )
    meta_path = Path(meta_dir) / "model_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["strategy"] = strategy
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved best params to {meta_path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
