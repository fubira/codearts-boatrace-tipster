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


def _load_enqueue_params(model_dir: str) -> dict:
    """Load HP + strategy thresholds from model_meta.json for warm-start.

    Returns a dict compatible with Optuna's enqueue_trial(). Only includes
    parameters that are part of the search space.
    """
    meta_path = Path(model_dir) / "ranking" / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No model_meta.json at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    hp = meta.get("hyperparameters", {})
    strategy = meta.get("strategy", {})

    # Core HP from the suggest_* calls in objective()
    params = {
        "num_leaves": hp["num_leaves"],
        "max_depth": hp["max_depth"],
        "min_child_samples": hp["min_child_samples"],
        "subsample": hp["subsample"],
        "colsample_bytree": hp["colsample_bytree"],
        "reg_alpha": hp["reg_alpha"],
        "reg_lambda": hp["reg_lambda"],
        "n_estimators": hp["n_estimators"],
        "learning_rate": hp["learning_rate"],
    }
    # Strategy thresholds (included only if they're in the Optuna search space)
    if "top3_conc_threshold" in strategy and strategy["top3_conc_threshold"] is not None:
        params["top3_conc_threshold"] = strategy["top3_conc_threshold"]
    if "gap23_threshold" in strategy and strategy["gap23_threshold"] is not None:
        params["gap23_threshold"] = strategy["gap23_threshold"]
    if "ev_threshold" in strategy and strategy["ev_threshold"] is not None:
        params["ev_threshold"] = strategy["ev_threshold"]
    # Relevance scheme if it's in the search space
    if "relevance_scheme" in hp:
        params["relevance"] = hp["relevance_scheme"]
    return params


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
    parser.add_argument("--pruner-percentile", type=int, default=10,
                        help="PercentilePruner threshold (0=disable, default: 10)")
    parser.add_argument("--pruner-warmup", type=int, default=20,
                        help="Number of trials before pruning starts (default: 20)")
    parser.add_argument("--output-json", default="models/tune_result/trials.json",
                        help="Path to save all trial details as JSON")
    parser.add_argument("--from-model", default=None,
                        help="Comma-separated model directories to seed the search "
                             "with. HP is read from model_meta.json and enqueued as "
                             "initial trials. Example: models/p2_v1,models/aa_294")
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
        fold_best_iters = []
        total_races = 0
        total_cost = 0.0
        total_payout = 0.0
        rois = []

        bankroll = 70000.0
        days_per_fold = args.fold_months * 30

        for fold_idx, fold in enumerate(folds):
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
            # Track effective iteration count for production training
            best_it = getattr(rank_model, "best_iteration_", None)
            fold_best_iters.append(best_it if best_it is not None else n_estimators)

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

            # Report running growth for pruning
            running_growth_rates = [
                np.log(max(1 + (fp / days_per_fold) / bankroll, 1e-6))
                for fp in fold_profits
            ]
            running_growth = float(np.mean(running_growth_rates))
            trial.report(running_growth, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Require minimum volume
        if total_races < 20:
            return -999.0

        mean_roi = float(np.mean(rois)) if rois else 0
        profit = total_payout - total_cost

        # Growth: daily compound growth rate (already computed incrementally above)
        growth = running_growth

        # Kelly: mean of log(ROI) — rewards high ROI regardless of volume
        log_rois = [np.log(max(r, 1e-6)) for r in rois]
        kelly = float(np.mean(log_rois))

        avg_best_iter = int(round(sum(fold_best_iters) / len(fold_best_iters)))
        trial.set_user_attr("mean_roi", round(mean_roi, 4))
        trial.set_user_attr("rois", [round(r, 4) for r in rois])
        trial.set_user_attr("total_races", total_races)
        trial.set_user_attr("profit", round(profit, 1))
        trial.set_user_attr("growth", round(growth, 6))
        trial.set_user_attr("kelly", round(kelly, 4))
        trial.set_user_attr("relevance", relevance)
        trial.set_user_attr("avg_best_iter", avg_best_iter)
        trial.set_user_attr("fold_best_iters", fold_best_iters)

        if args.objective == "kelly":
            return kelly
        return growth

    pruner: optuna.pruners.BasePruner
    if args.pruner_percentile > 0:
        # PercentilePruner(percentile=X) prunes trials below the X-th percentile
        # n_warmup_steps=0: allow pruning from fold 1 (step 0) onward
        pruner = optuna.pruners.PercentilePruner(
            percentile=args.pruner_percentile,
            n_startup_trials=args.pruner_warmup,
            n_warmup_steps=0,
        )
        print(
            f"Pruner: PercentilePruner(bottom {args.pruner_percentile}%, "
            f"warmup={args.pruner_warmup} trials)"
        )
    else:
        pruner = optuna.pruners.NopPruner()
        print("Pruner: disabled")

    study = optuna.create_study(
        direction="maximize",
        study_name="p2-trifecta",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
    )

    # Seed search with existing models' HPs (enqueued as initial trials)
    if args.from_model:
        for model_dir in args.from_model.split(","):
            model_dir = model_dir.strip()
            try:
                params = _load_enqueue_params(model_dir)
                # Drop params that are fixed by --fix-thresholds
                for k in list(params.keys()):
                    short = k.replace("_threshold", "")
                    if short in fixed_thresholds:
                        del params[k]
                # Drop relevance if --relevance fixed
                if args.relevance and "relevance" in params:
                    del params["relevance"]
                study.enqueue_trial(params)
                print(f"Warm-start enqueued: {model_dir}")
            except Exception as err:
                print(f"WARN: warm-start failed for {model_dir}: {err}")

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Pruning summary
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    print(
        f"\nTrial states: {n_complete} complete, {n_pruned} pruned "
        f"({n_pruned / args.trials * 100:.0f}%)"
    )

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

    # Save all completed trials to JSON (for train_dev_model.py to read)
    # Includes params + user_attrs (best_iter etc.) so dev training can match
    # Optuna's effective training configuration.
    trials_json_path = Path(args.output_json)
    trials_json_path.parent.mkdir(parents=True, exist_ok=True)
    trials_data = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_trials": args.trials,
        "seed": args.seed,
        "fix_thresholds": fixed_thresholds,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "params": dict(t.params),
                "user_attrs": dict(t.user_attrs),
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    with open(trials_json_path, "w") as f:
        json.dump(trials_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved {len(trials_data['trials'])} completed trials to {trials_json_path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
