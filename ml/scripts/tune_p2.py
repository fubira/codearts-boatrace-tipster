"""Optuna tuning for P2 trifecta strategy.

Strategy: B1 prediction + gap23 filter + 3連単 EV adaptive tickets.
Model: Non-odds features LambdaRank (no boat1 binary classifier needed).

Usage:
    uv run --directory ml python -m scripts.tune_p2 --trials 50
    uv run --directory ml python -m scripts.tune_p2 --trials 100 --seed 123
    uv run --directory ml python -m scripts.tune_p2 --trials 100 --fix-thresholds "gap23=0.15,ev=0.1"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits
from boatrace_tipster_ml.registry import next_prefix

FIELD_SIZE = 6


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
    if "gap12_min_threshold" in strategy and strategy["gap12_min_threshold"] is not None:
        params["gap12_min_threshold"] = strategy["gap12_min_threshold"]
    # Relevance scheme if it's in the search space
    if "relevance_scheme" in hp:
        params["relevance"] = hp["relevance_scheme"]
    return params


# ---------------------------------------------------------------------------
# --narrow: search space restriction around a seed model
# ---------------------------------------------------------------------------
#
# When --narrow is given together with --from-model, every Optuna suggestion is
# constrained to a small neighborhood of the seed model's hyperparameters.
# Linear-space parameters use a ratio (±ratio relative to center). Log-space
# parameters use a multiplicative factor (center / factor .. center * factor).
# Threshold parameters use an absolute delta because they can be 0 or near-0.
#
# Constants are tuned to match the existing global ranges roughly while keeping
# exploration tight enough to be meaningfully different from a fresh search.

NARROW_RATIO = {
    # int linear (±ratio of center, clamped to global bounds)
    "num_leaves": 0.30,
    "max_depth": 0.25,
    "min_child_samples": 0.50,
    "n_estimators": 0.30,
    # float linear
    "subsample": 0.20,
    "colsample_bytree": 0.20,
}
NARROW_LOG_FACTOR = {
    # log scale: [center / factor, center * factor]
    "learning_rate": 2.0,
    "reg_alpha": 10.0,
    "reg_lambda": 10.0,
}
NARROW_ABS_DELTA = {
    # additive delta — used for thresholds that can be 0 or near-zero
    "top3_conc_threshold": 0.15,
    "gap23_threshold": 0.05,
    "ev_threshold": 0.15,
    "gap12_min_threshold": 0.04,
}


def _narrow_int(center: float, ratio: float, lo: int, hi: int) -> tuple[int, int]:
    """Narrow int range to ``center * (1 ± ratio)``, clamped to ``[lo, hi]``.
    Falls back to a single point at the clamped center if the range collapses.
    """
    new_lo = max(lo, int(center * (1 - ratio)))
    new_hi = min(hi, int(center * (1 + ratio)))
    if new_lo >= new_hi:
        c = max(lo, min(hi, int(round(center))))
        return c, c
    return new_lo, new_hi


def _narrow_float(center: float, ratio: float, lo: float, hi: float) -> tuple[float, float]:
    """Narrow float range to ``center * (1 ± ratio)``, clamped to ``[lo, hi]``."""
    new_lo = max(lo, center * (1 - ratio))
    new_hi = min(hi, center * (1 + ratio))
    if new_lo >= new_hi:
        c = max(lo, min(hi, center))
        return c, c
    return new_lo, new_hi


def _narrow_log(center: float, factor: float, lo: float, hi: float) -> tuple[float, float]:
    """Narrow log-scale range to ``[center/factor, center*factor]``, clamped."""
    new_lo = max(lo, center / factor)
    new_hi = min(hi, center * factor)
    if new_lo >= new_hi:
        c = max(lo, min(hi, center))
        return c, c
    return new_lo, new_hi


def _narrow_abs(center: float, delta: float, lo: float, hi: float) -> tuple[float, float]:
    """Narrow range to ``center ± delta``, clamped. Used for thresholds whose
    center can be 0 or near-zero (where multiplicative ratios collapse)."""
    new_lo = max(lo, center - delta)
    new_hi = min(hi, center + delta)
    if new_lo >= new_hi:
        c = max(lo, min(hi, center))
        return c, c
    return new_lo, new_hi


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
    gap12_min_threshold: float = 0.0,
    per_race: bool = False,
) -> dict | list[dict]:
    """Evaluate P2 adaptive strategy.

    For each race:
    1. Check if top-1 prediction is boat 1
    2. Check if gap12 >= threshold (model's 1/2 confidence gap)
    3. Check if top3_concentration >= threshold (rank2+3 separated from rest)
    4. Check if gap23 >= threshold
    5. For P2 tickets (1-2-3 and 1-3-2), compute 3連単 EV
    6. Buy only tickets with EV >= threshold

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

        # Filter 2: gap12 threshold (model's 1 vs 2 confidence gap must be meaningful)
        gap12 = p1_prob - p2_prob
        if gap12 < gap12_min_threshold:
            continue

        # Filter 3: top3_concentration threshold
        top3_conc = (p2_prob + p3_prob) / (1 - p1_prob + 1e-10)
        if top3_conc < top3_conc_threshold:
            continue

        # Filter 4: gap23 threshold
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
    parser.add_argument(
        "--objective", choices=["profit", "growth", "kelly"], default="profit",
        help=(
            "Optimization objective. Default 'profit' (WF-CV total P/L with "
            "volume + hit_pct floors) is aligned with Phase 2 selection so "
            "Phase 1 value ↔ Phase 2 candidate pool stay consistent. "
            "'growth' (log-compressed daily P/L) and 'kelly' (mean log-ROI) "
            "are legacy alternatives; both diverge from Phase 2 selection."
        ),
    )
    parser.add_argument("--relevance", default=None,
                        help="Fix relevance scheme (linear/top_heavy/podium). If omitted, included in search space.")
    parser.add_argument("--fix-thresholds", default=None,
                        help="Fix strategy thresholds. Format: gap23=0.15,ev=0.1,top3_conc=0.7,gap12=0.04")
    parser.add_argument("--output-json", default="models/tune_result/trials.json",
                        help="Path to save all trial details as JSON")
    parser.add_argument("--from-model", default=None,
                        help="Comma-separated model directories to seed the search "
                             "with. HP is read from model_meta.json and enqueued as "
                             "initial trials. Example: models/p2_v1,models/aa_294")
    parser.add_argument("--narrow", action="store_true",
                        help="Constrain every Optuna suggestion to a small "
                             "neighborhood of the seed model's HP. Requires "
                             "--from-model. The first listed model is used as "
                             "the center.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of trials to run in parallel (default: 1). "
                             "Pair with --num-threads to avoid LightGBM oversubscription.")
    parser.add_argument("--num-threads", type=int, default=0,
                        help="LightGBM num_threads per trial. 0 = LightGBM default "
                             "(all cores). With --n-jobs N, set this to "
                             "ceil(physical_cores / N) to avoid oversubscription.")
    parser.add_argument("--run-prefix", default=None,
                        help="Tune run identifier (e.g., 'ab') saved into "
                             "trials.json. All dev models trained from this "
                             "tune share the prefix. Auto-allocated from the "
                             "local registry if omitted.")
    parser.add_argument("--final-top-n", type=int, default=10,
                        help="Number of trials shown in the final ranking "
                             "log output (growth top / Kelly top). Normally "
                             "set to match Phase 2 candidate count so the "
                             "log shows exactly the HPs that Phase 2 evaluates.")
    args = parser.parse_args()

    if args.narrow and not args.from_model:
        print("ERROR: --narrow requires --from-model", file=sys.stderr)
        sys.exit(1)

    # Parallel tuning competes with anything else running on the machine
    # (runner, scraper, interactive work). Only allow it on the tune server,
    # which is signaled via env var set by scripts/server-tune.sh.
    if args.n_jobs > 1 and os.environ.get("BOATRACE_TUNE_PARALLEL") != "1":
        print(
            "ERROR: --n-jobs > 1 is only allowed on the tune server.\n"
            "  Use scripts/server-tune.sh, which sets BOATRACE_TUNE_PARALLEL=1.\n"
            "  (Local single-trial tuning still works with the default --n-jobs 1.)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve the run prefix. server-tune.sh allocates locally and passes it
    # via --run-prefix; direct local invocation falls back to consuming a
    # fresh prefix from the registry. Either way, every tune run claims a
    # unique identifier so dev models from this run share the same prefix.
    run_prefix: str = args.run_prefix or next_prefix()
    print(f"Tune run prefix: {run_prefix}")

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

    # Load narrow seed (HP center for --narrow mode). The first --from-model
    # entry wins; additional entries are still enqueued as warm-start trials
    # but the search neighborhood is anchored to the first one. Optuna will
    # warn and ignore enqueued values that fall outside the narrowed ranges,
    # which is the correct behavior — the second seed is treated as a "tried
    # but out of scope" data point.
    narrow_seed: dict[str, float | str] | None = None
    if args.narrow:
        first_model = args.from_model.split(",")[0].strip()
        narrow_seed = _load_enqueue_params(first_model)
        print(f"Narrow mode: anchoring search to {first_model}")
        for k, v in narrow_seed.items():
            print(f"    {k}: {v}")

    print(f"Setup done in {time.time() - t0:.1f}s\n", flush=True)

    def objective(trial: optuna.Trial) -> float:
        # LambdaRank hyperparameters
        if narrow_seed:
            nl_lo, nl_hi = _narrow_int(
                narrow_seed["num_leaves"], NARROW_RATIO["num_leaves"], 15, 127,
            )
            md_lo, md_hi = _narrow_int(
                narrow_seed["max_depth"], NARROW_RATIO["max_depth"], 3, 12,
            )
            mc_lo, mc_hi = _narrow_int(
                narrow_seed["min_child_samples"], NARROW_RATIO["min_child_samples"],
                5, 100,
            )
            ss_lo, ss_hi = _narrow_float(
                narrow_seed["subsample"], NARROW_RATIO["subsample"], 0.4, 1.0,
            )
            cs_lo, cs_hi = _narrow_float(
                narrow_seed["colsample_bytree"], NARROW_RATIO["colsample_bytree"],
                0.4, 1.0,
            )
            ra_lo, ra_hi = _narrow_log(
                narrow_seed["reg_alpha"], NARROW_LOG_FACTOR["reg_alpha"],
                1e-8, 10.0,
            )
            rl_lo, rl_hi = _narrow_log(
                narrow_seed["reg_lambda"], NARROW_LOG_FACTOR["reg_lambda"],
                1e-8, 10.0,
            )
            ne_lo, ne_hi = _narrow_int(
                narrow_seed["n_estimators"], NARROW_RATIO["n_estimators"], 100, 1500,
            )
            lr_lo, lr_hi = _narrow_log(
                narrow_seed["learning_rate"], NARROW_LOG_FACTOR["learning_rate"],
                0.005, 0.2,
            )
        else:
            nl_lo, nl_hi = 15, 127
            md_lo, md_hi = 3, 12
            mc_lo, mc_hi = 5, 100
            ss_lo, ss_hi = 0.4, 1.0
            cs_lo, cs_hi = 0.4, 1.0
            ra_lo, ra_hi = 1e-8, 10.0
            rl_lo, rl_hi = 1e-8, 10.0
            ne_lo, ne_hi = 100, 1500
            lr_lo, lr_hi = 0.005, 0.2

        rank_params = {
            "num_leaves": trial.suggest_int("num_leaves", nl_lo, nl_hi),
            "max_depth": trial.suggest_int("max_depth", md_lo, md_hi),
            "min_child_samples": trial.suggest_int("min_child_samples", mc_lo, mc_hi),
            "subsample": trial.suggest_float("subsample", ss_lo, ss_hi),
            "colsample_bytree": trial.suggest_float("colsample_bytree", cs_lo, cs_hi),
            "reg_alpha": trial.suggest_float("reg_alpha", ra_lo, ra_hi, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", rl_lo, rl_hi, log=True),
        }
        if args.num_threads > 0:
            rank_params["num_threads"] = args.num_threads
        n_estimators = trial.suggest_int("n_estimators", ne_lo, ne_hi)
        learning_rate = trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True)
        if args.relevance:
            relevance = args.relevance
        elif narrow_seed and "relevance" in narrow_seed:
            # Lock relevance in narrow mode (categorical can't be narrowed)
            relevance = narrow_seed["relevance"]
        else:
            relevance = trial.suggest_categorical(
                "relevance", ["linear", "top_heavy", "podium"]
            )

        # Strategy thresholds
        if "gap23" in fixed_thresholds:
            gap23_threshold = fixed_thresholds["gap23"]
        elif narrow_seed and "gap23_threshold" in narrow_seed:
            g_lo, g_hi = _narrow_abs(
                narrow_seed["gap23_threshold"], NARROW_ABS_DELTA["gap23_threshold"],
                0.0, 0.25,
            )
            gap23_threshold = trial.suggest_float("gap23_threshold", g_lo, g_hi)
        else:
            gap23_threshold = trial.suggest_float("gap23_threshold", 0.0, 0.25)

        if "ev" in fixed_thresholds:
            ev_threshold = fixed_thresholds["ev"]
        elif narrow_seed and "ev_threshold" in narrow_seed:
            e_lo, e_hi = _narrow_abs(
                narrow_seed["ev_threshold"], NARROW_ABS_DELTA["ev_threshold"],
                -0.3, 0.5,
            )
            ev_threshold = trial.suggest_float("ev_threshold", e_lo, e_hi)
        else:
            ev_threshold = trial.suggest_float("ev_threshold", -0.3, 0.5)

        if "top3_conc" in fixed_thresholds:
            top3_conc_threshold = fixed_thresholds["top3_conc"]
        elif narrow_seed and "top3_conc_threshold" in narrow_seed:
            c_lo, c_hi = _narrow_abs(
                narrow_seed["top3_conc_threshold"],
                NARROW_ABS_DELTA["top3_conc_threshold"],
                0.0, 0.85,
            )
            top3_conc_threshold = trial.suggest_float("top3_conc_threshold", c_lo, c_hi)
        else:
            top3_conc_threshold = trial.suggest_float("top3_conc_threshold", 0.0, 0.85)

        if "gap12" in fixed_thresholds:
            gap12_min_threshold = fixed_thresholds["gap12"]
        elif narrow_seed and "gap12_min_threshold" in narrow_seed:
            g12_lo, g12_hi = _narrow_abs(
                narrow_seed["gap12_min_threshold"],
                NARROW_ABS_DELTA["gap12_min_threshold"],
                0.0, 0.20,
            )
            gap12_min_threshold = trial.suggest_float("gap12_min_threshold", g12_lo, g12_hi)
        else:
            gap12_min_threshold = trial.suggest_float("gap12_min_threshold", 0.0, 0.20)

        fold_profits = []
        fold_best_iters = []
        total_races = 0
        total_wins = 0
        total_cost = 0.0
        total_payout = 0.0
        rois = []

        bankroll = 70000.0
        days_per_fold = args.fold_months * 30

        for fold in folds:
            # X is pre-filtered to FEATURES before walk_forward_splits, so fold
            # X already has exactly the FEATURES columns. LightGBM is silent
            # (verbose=-1 in DEFAULT_PARAMS), so no stdout redirect needed
            # (stdout redirect is thread-unsafe under --n-jobs > 1).
            # Early stopping is disabled: with a 1-month val window the val
            # score curve is noise-dominated, so best_iter shifts wildly from
            # tiny train perturbations. Full training matches the production
            # training regime.
            rank_model, _ = train_model(
                fold["train"]["X"], fold["train"]["y"], fold["train"]["meta"],
                fold["val"]["X"], fold["val"]["y"], fold["val"]["meta"],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                relevance_scheme=relevance,
                extra_params=rank_params,
                early_stopping_rounds=None,
            )
            # ES disabled, so best_iter == n_estimators (kept for avg_best_iter compat)
            fold_best_iters.append(n_estimators)

            rank_scores = rank_model.predict(fold["test"]["X"])

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
                gap12_min_threshold=gap12_min_threshold,
            )

            fold_profit = result["payout"] - result["cost"]
            fold_profits.append(fold_profit)
            total_races += result["races"]
            total_wins += result["wins"]
            total_cost += result["cost"]
            total_payout += result["payout"]
            rois.append(result["roi"] if result["cost"] > 0 else 0)

        # Compute metrics
        mean_roi = float(np.mean(rois)) if rois else 0
        profit = total_payout - total_cost

        # Growth: mean log daily growth across folds
        growth_rates = [
            np.log(max(1 + (fp / days_per_fold) / bankroll, 1e-6))
            for fp in fold_profits
        ]
        growth = float(np.mean(growth_rates)) if growth_rates else -999.0

        # Kelly: mean of log(ROI) — rewards high ROI regardless of volume
        log_rois = [np.log(max(r, 1e-6)) for r in rois]
        kelly = float(np.mean(log_rois)) if log_rois else -999.0

        avg_best_iter = (
            int(round(sum(fold_best_iters) / len(fold_best_iters)))
            if fold_best_iters
            else 0
        )

        # Hit rate across all folds. Used by downstream tools (Phase 2
        # selection filter, feedback_hit_rate_10pct_boundary). <10% is a
        # Kelly stability risk — low-hit × high-ROI HPs blow up variance.
        hit_pct = (100 * total_wins / total_races) if total_races > 0 else 0.0

        # Set user_attrs BEFORE any early return so downstream tools (Top 10
        # print, trials.json, train_dev_model.py) can safely read them for all
        # COMPLETE trials.
        trial.set_user_attr("mean_roi", round(mean_roi, 4))
        trial.set_user_attr("rois", [round(r, 4) for r in rois])
        trial.set_user_attr("total_races", total_races)
        trial.set_user_attr("total_wins", total_wins)
        trial.set_user_attr("hit_pct", round(hit_pct, 2))
        trial.set_user_attr("profit", round(profit, 1))
        trial.set_user_attr("growth", round(growth, 6))
        trial.set_user_attr("kelly", round(kelly, 4))
        trial.set_user_attr("relevance", relevance)
        trial.set_user_attr("avg_best_iter", avg_best_iter)
        trial.set_user_attr("fold_best_iters", fold_best_iters)

        # Dual floor for the profit objective (default):
        #   - volume: reject HPs that buy <PHASE2_MIN_RACES races across the
        #     WF-CV horizon (naive exploitation via lucky high-odds hits).
        #   - quality: reject HPs that hit <7.0% (favorite-heavy + losing ROI
        #     combos that would slip past a pure volume gate at ev=-0.25).
        # Sentinel is returned AFTER user_attrs so floored trials remain
        # inspectable in logs/trials.json.
        # growth/kelly legacy objectives keep the looser <20-race sentinel
        # for backward compatibility with archived tune-run comparisons.
        from scripts.seed_stability_check import PHASE2_MIN_RACES
        if args.objective == "profit":
            if total_races < PHASE2_MIN_RACES or hit_pct < 7.0:
                return -1e9
            return profit
        if total_races < 20:
            return -999.0
        if args.objective == "kelly":
            return kelly
        return growth

    study = optuna.create_study(
        direction="maximize",
        study_name="p2-trifecta",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
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

    # Per-trial callback: surface FAIL trials immediately. COMPLETE trials are
    # already logged by Optuna's built-in logger.
    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[I {ts}] Trial {trial.number} FAILED", flush=True)

    if args.n_jobs > 1:
        print(
            f"Parallel tuning: n_jobs={args.n_jobs}, "
            f"num_threads={args.num_threads or 'lgb-default'}",
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        callbacks=[_trial_callback],
    )

    # Results
    obj_label = {"profit": "Profit", "growth": "Growth", "kelly": "Kelly"}[args.objective]
    print("\n" + "=" * 70)
    print(f"Optuna Search Complete — P2 Trifecta Strategy (objective: {args.objective})")
    print("=" * 70)
    # profit prints as yen; growth/kelly keep 6-decimal scientific display.
    if args.objective == "profit":
        print(f"Best {obj_label}: {study.best_value:+,.0f}円")
    else:
        print(f"Best {obj_label}: {study.best_value:.6f}")
    bp = study.best_params
    print(f"Best params:")
    for k, v in sorted(bp.items()):
        print(f"  {k}: {v}")

    ba = study.best_trial.user_attrs
    print(f"\nBest trial metrics:")
    print(f"  Mean ROI: {ba.get('mean_roi', 0):.1%}")
    print(f"  Hit%:     {ba.get('hit_pct', 0):.1f}%")
    print(f"  Profit:   {ba.get('profit', 0):+,.0f}円")
    print(f"  Races:    {ba.get('total_races', 0)}")
    print(f"  Growth:   {ba.get('growth', 'N/A')}")
    print(f"  Kelly:    {ba.get('kelly', 'N/A')}")
    print(f"  Folds:    {ba.get('rois', 'N/A')}")

    # Save all completed trials to JSON FIRST (before prints that could crash).
    # Includes params + user_attrs (best_iter etc.) so train_dev_model.py can
    # match Optuna's effective training configuration.
    trials_json_path = Path(args.output_json)
    trials_json_path.parent.mkdir(parents=True, exist_ok=True)
    trials_data = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_trials": args.trials,
        "seed": args.seed,
        "run_prefix": run_prefix,
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

    # Top 10 trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def _fmt_trial(t) -> str:
        ua = t.user_attrs
        rel = ua.get("relevance", "?")
        g23 = t.params.get("gap23_threshold", fixed_thresholds.get("gap23", "?"))
        evt = t.params.get("ev_threshold", fixed_thresholds.get("ev", "?"))
        t3c = t.params.get("top3_conc_threshold", fixed_thresholds.get("top3_conc", "?"))
        g12 = t.params.get("gap12_min_threshold", fixed_thresholds.get("gap12", "?"))
        growth = ua.get("growth", 0.0)
        kelly_v = ua.get("kelly", "?")
        roi = ua.get("mean_roi", 0.0)
        pl = ua.get("profit", 0.0)
        n = ua.get("total_races", 0)
        hit = ua.get("hit_pct", 0.0)
        return (
            f"  #{t.number:>3}: growth={growth:.6f} kelly={kelly_v} ROI={roi:.0%} "
            f"hit={hit:.1f}% P/L={pl:+,.0f} n={n} "
            f"rel={rel} gap23={g23} ev={evt} conc={t3c} gap12={g12}"
        )

    top_n = args.final_top_n
    print(f"\nTop {top_n} trials (by {obj_label}):")
    for t in sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]:
        print(_fmt_trial(t))

    # Phase 2 candidate ranking: matches seed_stability_check selection.
    # Same floors as the profit objective (volume >= PHASE2_MIN_RACES AND
    # hit_pct >= 7.0) so Phase 1 value and Phase 2 pool agree regardless of
    # which --objective was used for the tune.
    from scripts.seed_stability_check import PHASE2_MIN_RACES

    def _profit_key(t):
        ua = t.user_attrs or {}
        races = ua.get("total_races") or 0
        hit = ua.get("hit_pct") or 0.0
        if races < PHASE2_MIN_RACES or hit < 7.0:
            return float("-inf")
        pr = ua.get("profit")
        return pr if pr is not None else float("-inf")

    profit_sorted = sorted(completed, key=_profit_key, reverse=True)
    profit_top = [
        t for t in profit_sorted if _profit_key(t) != float("-inf")
    ][:top_n]
    print(
        f"\nTop {top_n} trials (by profit, Phase 2 candidates, "
        f"volume>={PHASE2_MIN_RACES} AND hit>=7.0%):"
    )
    for t in profit_top:
        print(_fmt_trial(t))

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
        "gap12_min_threshold": bp.get("gap12_min_threshold", fixed_thresholds.get("gap12")),
        "ev_basis": "3連単 odds per ticket",
        "features": "non_odds_21",
    }
    # Inherit excluded_stadiums from the first --from-model seed (if any).
    # Stadium exclusions are a strategic decision decoupled from HP tuning,
    # so they should persist across retunes rather than being dropped.
    # Emit a warning on read failure: a silent drop here would recreate the
    # same "invisible information loss" failure mode tracked in knowledge/.
    if args.from_model:
        first_seed = args.from_model.split(",")[0].strip()
        seed_meta_path = Path(first_seed) / "ranking" / "model_meta.json"
        if seed_meta_path.exists():
            try:
                with open(seed_meta_path) as f:
                    seed_meta = json.load(f)
                seed_excluded = seed_meta.get("strategy", {}).get("excluded_stadiums")
                if seed_excluded:
                    strategy["excluded_stadiums"] = seed_excluded
            except (OSError, json.JSONDecodeError) as err:
                print(
                    f"WARN: failed to inherit excluded_stadiums from "
                    f"{seed_meta_path}: {err}",
                    file=sys.stderr,
                )
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
