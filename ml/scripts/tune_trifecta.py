"""Optuna hyperparameter tuning for trifecta X-allflow strategy.

Optimizes LambdaRank + boat1 binary model jointly for 3連単 ROI.

Usage:
    uv run --directory ml python -m scripts.tune_trifecta --trials 100
    uv run --directory ml python -m scripts.tune_trifecta --trials 100 --seed 123
    uv run --directory ml python -m scripts.tune_trifecta --trials 200 --objective profit

Objectives:
    - sharpe (default): (mean_roi - 1) / std_roi — stability-adjusted ROI
    - profit: total P/L across folds — rewards buying more when profitable

Tuned parameters:
    - LambdaRank: num_leaves, max_depth, learning_rate, n_estimators, etc.
    - LambdaRank: relevance_scheme (linear, top_heavy, win_only, podium)
    - Strategy: b1_threshold, ev_threshold, r2_ev_threshold
"""

import argparse
import contextlib
import io
import sys
import time

import numpy as np
import optuna
import pandas as pd

from collections import defaultdict

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.evaluate import evaluate_trifecta_strategy
from boatrace_tipster_ml.feature_config import FEATURE_COLS, prepare_feature_matrix
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

    tri_win_prob: dict[tuple[int, int], float] = defaultdict(float)
    for r in rows:
        rid, combo, odds = int(r[0]), r[1], float(r[2])
        if odds <= 0:
            continue
        first_boat = int(combo.split("-")[0])
        tri_win_prob[(rid, first_boat)] += 0.75 / odds

    conn.close()
    return trifecta_odds, dict(tri_win_prob)


def _validate_top_trials(
    study: optuna.Study,
    df: pd.DataFrame,
    trifecta_odds: dict,
    tri_win_prob: dict,
    finish_map: dict,
    folds: list,
    n_top: int,
    fixed_thresholds: dict[str, float],
):
    """Phase 1.5: Train fixed models for top N trials and evaluate on OOS."""
    # OOS = last fold's test period to end of data
    last_test = folds[-1]["period"]["test"]
    oos_start = last_test.split("~")[0].strip()
    max_date = df["race_date"].max()

    print(f"\n{'='*70}")
    print(f"Phase 1.5: OOS Validation of Top {n_top} Trials")
    print(f"{'='*70}")
    print(f"Train: < {oos_start}, OOS: {oos_start} ~ {max_date}")

    train_df = df[df["race_date"] < oos_start]
    oos_df = df[df["race_date"] >= oos_start]
    n_oos_races = len(oos_df) // 6
    print(f"OOS: {n_oos_races} races")

    if n_oos_races == 0:
        print("WARNING: No OOS data")
        return

    # Train boat1 once (not tuned)
    print("Training boat1 for validation...", end="", flush=True)
    dates_all = sorted(train_df["race_date"].unique())
    val_start = dates_all[max(0, len(dates_all) - 60)]
    with contextlib.redirect_stdout(io.StringIO()):
        X_b1_tr, y_b1_tr, _ = reshape_to_boat1(train_df[train_df["race_date"] < val_start])
        X_b1_v, y_b1_v, _ = reshape_to_boat1(train_df[train_df["race_date"] >= val_start])
        X_b1_oos, _, meta_b1_oos = reshape_to_boat1(oos_df)
        b1_model, _ = train_boat1_model(X_b1_tr, y_b1_tr, X_b1_v, y_b1_v)
    b1_probs_oos = b1_model.predict_proba(X_b1_oos)[:, 1]
    print(" done")

    # Prepare ranking features (train split + OOS)
    X_train_full, y_train_full, meta_train_full = prepare_feature_matrix(train_df)
    X_oos, _, meta_oos = prepare_feature_matrix(oos_df)
    train_mask = meta_train_full["race_date"].values < val_start
    val_mask = ~train_mask

    # Threshold grid (include fixed thresholds if specified)
    b1_vals = sorted(set(np.arange(0.28, 0.51, 0.02).tolist() + [fixed_thresholds.get("b1", 0.35)]))
    ev_vals = sorted(set(np.arange(0.0, 0.51, 0.05).tolist() + [fixed_thresholds.get("ev", 0.29)]))

    # Get top N trials
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None and t.value > -999],
        key=lambda t: t.value,
        reverse=True,
    )[:n_top]

    results = []
    for rank, trial in enumerate(trials_sorted, 1):
        t0 = time.time()
        p = trial.params
        ua = trial.user_attrs
        relevance = p.get("relevance") or ua.get("relevance", "podium")

        rank_params = {k: p[k] for k in
                       ["num_leaves", "max_depth", "min_child_samples",
                        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}

        with contextlib.redirect_stdout(io.StringIO()):
            rank_model, _ = train_model(
                X_train_full[train_mask], y_train_full[train_mask], meta_train_full[train_mask],
                X_train_full[val_mask], y_train_full[val_mask], meta_train_full[val_mask],
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                relevance_scheme=relevance,
                extra_params=rank_params,
                early_stopping_rounds=50,
            )

        rank_scores_oos = rank_model.predict(X_oos)

        # Sweep thresholds
        best_pl = -99999.0
        best_b1 = best_ev = 0.0
        best_races = best_wins = 0
        best_roi = 0.0
        fixed_result = None

        for b1 in b1_vals:
            for ev in ev_vals:
                r = evaluate_trifecta_strategy(
                    b1_probs=b1_probs_oos, meta_b1=meta_b1_oos,
                    rank_scores=rank_scores_oos, meta_rank=meta_oos,
                    finish_map=finish_map, trifecta_odds=trifecta_odds,
                    tri_win_prob=tri_win_prob,
                    b1_threshold=b1, ev_threshold=ev,
                )
                pl = r["payout"] - r["cost"]
                if pl > best_pl and r["races"] >= 10:
                    best_pl, best_b1, best_ev = pl, float(b1), float(ev)
                    best_races, best_wins, best_roi = r["races"], r["wins"], r["roi"]

                # Capture result at fixed thresholds
                if (fixed_thresholds
                        and abs(b1 - fixed_thresholds["b1"]) < 0.005
                        and abs(ev - fixed_thresholds["ev"]) < 0.005):
                    fixed_result = {"pl": pl, "races": r["races"], "wins": r["wins"], "roi": r["roi"]}

        elapsed = time.time() - t0

        entry = {
            "trial": trial.number,
            "wfcv_pl": ua.get("profit", 0),
            "wfcv_roi": ua.get("mean_roi", 0),
            "wfcv_n": ua.get("total_races", 0),
            "wfcv_sharpe": ua.get("sharpe", 0),
            "oos_pl": best_pl, "oos_roi": best_roi,
            "oos_n": best_races, "oos_w": best_wins,
            "oos_b1": best_b1, "oos_ev": best_ev,
            "fixed": fixed_result,
        }
        results.append(entry)

        fx = ""
        if fixed_result:
            fx = f" fixed={fixed_result['pl']:+.0f}({fixed_result['races']}R)"
        print(
            f"  #{trial.number:>3}: best={best_pl:+.0f} "
            f"({best_races}R {best_wins}W {best_roi:.0%} "
            f"b1<{best_b1:.0%} ev>{best_ev:+.0%}){fx} [{elapsed:.0f}s]"
        )

    # Summary table sorted by OOS best P/L
    results.sort(key=lambda r: r["oos_pl"], reverse=True)
    print(f"\n{'--- OOS Ranking (by best P/L) ---':^70}")
    print(f"{'OOS':>4} {'#':>4} {'WF-CV':>8} {'WF-CV':>6} {'WF-CV':>5} "
          f"{'OOS':>8} {'OOS':>5} {'OOS':>5} {'opt':>5} {'opt':>5}")
    print(f"{'Rank':>4} {'':>4} {'P/L':>8} {'ROI':>6} {'n':>5} "
          f"{'P/L':>8} {'ROI':>5} {'n':>5} {'b1':>5} {'ev':>5}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(
            f"{i:>4} #{r['trial']:>3} "
            f"{r['wfcv_pl']:>+8.0f} {r['wfcv_roi']:>5.0%} {r['wfcv_n']:>5} "
            f"{r['oos_pl']:>+8.0f} {r['oos_roi']:>4.0%} {r['oos_n']:>5} "
            f"<{r['oos_b1']:.0%} >{r['oos_ev']:+.0%}"
        )


def main():
    parser = argparse.ArgumentParser(description="Tune trifecta strategy with Optuna")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--warm-start", action="store_true",
                        help="Seed search with current best params from model_meta.json")
    parser.add_argument("--objective", choices=["growth", "sharpe", "profit", "kelly"], default="growth",
                        help="Optimization objective: growth (daily compound growth, default), kelly (geometric mean ROI), sharpe (stability), or profit (total P/L)")
    parser.add_argument("--relevance", default=None,
                        help="Fix relevance scheme (linear/top_heavy/win_only/podium). If omitted, included in search space.")
    parser.add_argument("--with-r2", action="store_true",
                        help="Enable rank-2 fallback (search r2_ev_threshold). Disabled by default.")
    parser.add_argument("--fix-thresholds", default=None,
                        help="Fix strategy thresholds (skip threshold search). Format: b1=0.35,ev=0.29")
    parser.add_argument("--validate-top", type=int, default=0,
                        help="After search, train fixed models for top N trials and evaluate on OOS (Phase 1.5)")
    args = parser.parse_args()

    # Parse fixed thresholds
    fixed_thresholds: dict[str, float] = {}
    if args.fix_thresholds:
        for pair in args.fix_thresholds.split(","):
            k, v = pair.strip().split("=")
            fixed_thresholds[k.strip()] = float(v.strip())
        if "b1" not in fixed_thresholds or "ev" not in fixed_thresholds:
            parser.error("--fix-thresholds requires both b1 and ev (e.g., b1=0.35,ev=0.29)")
        print(f"Fixed thresholds: b1<{fixed_thresholds['b1']:.0%}, ev>{fixed_thresholds['ev']:+.0%}")

    print(f"Trials: {args.trials}, Folds: {args.n_folds}, Seed: {args.seed}, Objective: {args.objective}")
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

    obj_mode = args.objective
    fixed_relevance = args.relevance

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
        if fixed_relevance:
            relevance = fixed_relevance
        else:
            relevance = trial.suggest_categorical(
                "relevance", ["linear", "top_heavy", "win_only", "podium"]
            )

        # Strategy parameters
        if fixed_thresholds:
            b1_threshold = fixed_thresholds["b1"]
            ev_threshold = fixed_thresholds["ev"]
        else:
            b1_threshold = trial.suggest_float("b1_threshold", 0.30, 0.55)
            ev_threshold = trial.suggest_float("ev_threshold", -0.1, 0.5)
        r2_ev_threshold = None if not args.with_r2 else trial.suggest_float("r2_ev_threshold", 0.5, 2.0)

        rois = []
        fold_profits = []
        total_races = 0
        total_cost = 0.0
        total_payout = 0.0
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
                r2_ev_threshold=r2_ev_threshold,
            )
            rois.append(result["roi"])
            fold_profit = result["payout"] - result["cost"]
            fold_profits.append(fold_profit)
            total_races += result["races"]
            total_cost += result["cost"]
            total_payout += result["payout"]

        mean_roi = float(np.mean(rois))
        std_roi = float(np.std(rois))
        min_roi = float(np.min(rois))

        # Require minimum bet volume
        if total_races < 50:
            return -999.0

        sharpe = (mean_roi - 1.0) / std_roi if std_roi > 0 else 0
        profit = total_payout - total_cost
        # Kelly: geometric mean of fold ROIs (log-space average)
        log_rois = np.log(np.clip(rois, 1e-6, None))
        kelly = float(np.mean(log_rois))
        # Growth: daily compound growth rate with bankroll-relative sizing
        # Rewards both volume and quality — n_bets × per_bet_profit naturally balanced
        bankroll = 70000.0
        days_per_fold = args.fold_months * 30
        growth_rates = []
        for fp in fold_profits:
            daily_profit = fp / days_per_fold
            growth_rates.append(np.log(max(1 + daily_profit / bankroll, 1e-6)))
        growth = float(np.mean(growth_rates))

        trial.set_user_attr("mean_roi", mean_roi)
        trial.set_user_attr("roi_std", std_roi)
        trial.set_user_attr("roi_min", min_roi)
        trial.set_user_attr("roi_max", float(np.max(rois)))
        trial.set_user_attr("rois", [round(r, 4) for r in rois])
        trial.set_user_attr("sharpe", round(sharpe, 3))
        trial.set_user_attr("kelly", round(kelly, 4))
        trial.set_user_attr("growth", round(growth, 6))
        trial.set_user_attr("total_races", total_races)
        trial.set_user_attr("profit", round(profit, 1))
        trial.set_user_attr("relevance", relevance)

        if obj_mode == "profit":
            return profit
        if obj_mode == "kelly":
            return kelly
        if obj_mode == "growth":
            return growth
        return sharpe

    study = optuna.create_study(
        direction="maximize",
        study_name="trifecta-x-allflow",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if args.warm_start:
        # Seed with current best params so TPE explores nearby first
        from boatrace_tipster_ml.model import load_model_meta
        meta = load_model_meta("models/trifecta_v1/ranking")
        if meta and "hyperparameters" in meta:
            hp = meta["hyperparameters"]
            st = meta.get("strategy", {})
            seed_params = {
                "num_leaves": hp.get("num_leaves", 92),
                "max_depth": hp.get("max_depth", 10),
                "min_child_samples": hp.get("min_child_samples", 15),
                "subsample": hp.get("subsample", 0.54),
                "colsample_bytree": hp.get("colsample_bytree", 0.62),
                "reg_alpha": hp.get("reg_alpha", 0.0002),
                "reg_lambda": hp.get("reg_lambda", 3.27),
                "n_estimators": hp.get("n_estimators", 1476),
                "learning_rate": hp.get("learning_rate", 0.029),
                "relevance": hp.get("relevance_scheme", "win_only"),
                "b1_threshold": st.get("b1_threshold", 0.42),
                "ev_threshold": st.get("ev_threshold", 0.36),
            }
            if args.with_r2:
                seed_params["r2_ev_threshold"] = st.get("r2_ev_threshold", 1.20)
            study.enqueue_trial(seed_params)
            print(f"Warm start: seeded with model_meta params (trial 0)")

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 70)
    print(f"Optuna Search Complete — Trifecta X-allflow (objective: {obj_mode})")
    print("=" * 70)
    obj_label = {"sharpe": "Sharpe", "profit": "Profit", "kelly": "Kelly", "growth": "Growth"}[obj_mode]
    fmt = ".6f" if obj_mode == "growth" else ".3f"
    print(f"Best {obj_label}: {study.best_value:{fmt}}")
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
    print(f"  Kelly:    {ba.get('kelly', 'N/A')}")
    print(f"  Growth:   {ba.get('growth', 'N/A')}")
    print(f"  Profit:   {ba['profit']:+.0f} (per ¥100 unit)")
    print(f"  Races:    {ba['total_races']}")
    print(f"  Folds:    {ba['rois']}")

    # Top 10 trials
    print(f"\nTop 10 trials (by {obj_label}):")
    trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value else -999,
        reverse=True,
    )
    for t in trials[:10]:
        if t.value is None:
            continue
        ua = t.user_attrs
        rel = t.params.get("relevance") or t.user_attrs.get("relevance", "?")
        b1t = t.params.get("b1_threshold") or fixed_thresholds.get("b1", 0)
        evt = t.params.get("ev_threshold") or fixed_thresholds.get("ev", 0)
        r2t = t.params.get("r2_ev_threshold")
        races = ua.get("total_races", "?")
        pft = ua.get("profit", 0)
        r2_str = f" r2>{r2t:+.0%}" if r2t is not None else ""
        print(
            f"  #{t.number:>3}: Sharpe={ua['sharpe']:.2f} P/L={pft:+.0f} "
            f"ROI={ua['mean_roi']:.0%}±{ua['roi_std']:.0%} "
            f"n={races} rel={rel} b1<{b1t:.0%} ev>{evt:+.0%}{r2_str}"
        )

    # Phase 1.5: OOS validation of top trials
    if args.validate_top > 0:
        _validate_top_trials(
            study, df, trifecta_odds, tri_win_prob, finish_map, folds,
            n_top=args.validate_top, fixed_thresholds=fixed_thresholds,
        )

    # Auto-save best params to model_meta.json
    best_hp = {
        "num_leaves": bp["num_leaves"],
        "max_depth": bp["max_depth"],
        "min_child_samples": bp["min_child_samples"],
        "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample_bytree"],
        "reg_alpha": bp["reg_alpha"],
        "reg_lambda": bp["reg_lambda"],
        "n_estimators": bp["n_estimators"],
        "learning_rate": bp["learning_rate"],
        "relevance_scheme": bp.get("relevance") or study.best_trial.user_attrs.get("relevance", "top_heavy"),
    }
    best_strategy = {
        "b1_threshold": bp.get("b1_threshold") or fixed_thresholds.get("b1", 0.35),
        "ev_threshold": bp.get("ev_threshold") or fixed_thresholds.get("ev", 0.29),
        "bet_pattern": "X-allflow (20pt)" + (" with rank-2 fallback" if args.with_r2 else ""),
        "ev_basis": "trifecta inverse (sum 0.75/odds)",
    }
    if args.with_r2:
        best_strategy["r2_ev_threshold"] = bp["r2_ev_threshold"]
    meta_dir = "models/tune_result"
    from pathlib import Path
    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    from boatrace_tipster_ml.model import save_model_meta
    save_model_meta(
        meta_dir,
        feature_columns=FEATURE_COLS,
        hyperparameters=best_hp,
        training={
            "note": f"Optuna {args.trials}t seed={args.seed} obj={obj_mode} "
                    f"#{study.best_trial.number} ({obj_label} {study.best_value:.2f})",
        },
    )
    # Append strategy to saved meta
    import json
    meta_path = Path(meta_dir) / "model_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["strategy"] = best_strategy
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved best params to {meta_path} (NOTE: not production model)")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
