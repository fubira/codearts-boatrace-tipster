"""Seed stability check — re-train top tune trials with multiple LightGBM
seeds and compare OOS performance. Detects winner's curse trials whose
WF-CV growth came from a single lucky seed rather than robust HP.

Workflow:
  1. Parse a tune log (trials.json sidecar)
  2. Pick top N trials by tune objective value (growth/kelly)
  3. For each trial, re-train K final models with different random_state
     values (end_date=2026-01-01 to keep OOS clean)
  4. Evaluate each model on the OOS period (analyze_model.evaluate_period)
  5. Print mean/std/per-seed P/L per trial and rank by mean P/L

Usage:
    cd ml && uv run python scripts/seed_stability_check.py \\
        --tune-log ../logs/tune/2026-04-13_2047_server-tune.log \\
        --top-n 5 --seeds 42,100,200,300,400 \\
        --from 2026-01-01 --to 2026-04-13

No models are saved to disk — all training and evaluation happens in memory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model
from scripts.analyze_model import evaluate_period
from scripts.train_dev_model import parse_tune_log, params_to_hp

FIELD_SIZE = 6
DEFAULT_SEEDS = [42, 100, 200, 300, 400]

# Minimum WF-CV races (across all folds) a trial must have to be a
# Phase 2 candidate. Kelly = mean(log(fold_ROI)) explodes for low-volume
# trials that catch a single lucky hit per fold — 4-14 tune had trial #12
# with 51 races, Kelly 1.43 (fake), OOS mean +2,650. p2_v2-class trials
# land around 500-1200 Phase 1 races, so 500 is a reasonable floor.
PHASE2_MIN_RACES = 500


def _load_trifecta_odds(db_path: str) -> dict:
    """Load all confirmed 3連単 odds. Same shape as analyze_model uses."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type='3連単'"
        ).fetchall()
    finally:
        conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def _build_strategy_meta(
    feature_means: dict[str, float],
    conc_th: float,
    gap12_th: float,
    gap23_th: float,
    ev_th: float,
) -> dict:
    """Build the model_meta dict shape expected by analyze_model.evaluate_period.

    feature_columns + strategy + feature_means are read by evaluate_period
    and fill_nan_with_means. feature_means must be precomputed from the
    train slice (NOT from the eval slice) to match what train_dev_model
    persists into the saved meta.
    """
    return {
        "feature_columns": FEATURES,
        "strategy": {
            "type": "P2",
            "gap23_threshold": gap23_th,
            "top3_conc_threshold": conc_th,
            "gap12_min_threshold": gap12_th,
            "ev_threshold": ev_th,
        },
        "feature_means": feature_means,
    }


def _train_one_seed(
    df_train: pd.DataFrame,
    val_mask: pd.Series,
    hp: dict,
    lr: float,
    n_est: int,
    relevance: str,
    seed: int,
) -> Any:
    """Train a single LightGBM model with a specific random_state seed.

    Re-uses the WF-CV-derived n_est (no early stopping to keep behavior
    deterministic w.r.t. the saved trial config). Returns the model object.
    """
    extra = dict(hp)
    extra["random_state"] = seed

    X = df_train[FEATURES].copy()
    y = df_train["finish_position"]
    meta = df_train[["race_id", "racer_id", "race_date", "boat_number"]].copy()

    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            X[~val_mask], y[~val_mask], meta[~val_mask],
            X[val_mask], y[val_mask], meta[val_mask],
            n_estimators=n_est, learning_rate=lr,
            relevance_scheme=relevance, extra_params=extra,
            early_stopping_rounds=None,
        )
    return model


def _evaluate(
    model,
    df_eval: pd.DataFrame,
    feature_means: dict[str, float],
    conc_th: float,
    gap12_th: float,
    gap23_th: float,
    ev_th: float,
    odds: dict,
    from_date: str,
    to_date: str,
) -> dict:
    """Run evaluate_period and return aggregate stats.

    feature_means must be precomputed from the train slice (same data the
    model was trained on) — using the eval slice would produce subtle drift
    vs train_dev_model's saved meta.

    Returns dict with: races, wins, hit_pct, roi_pct, pl, total_cost.
    """
    meta = _build_strategy_meta(feature_means, conc_th, gap12_th, gap23_th, ev_th)
    purchases, _ = evaluate_period(model, meta, df_eval, odds, from_date, to_date)
    total_cost = sum(p.cost for p in purchases)
    total_payout = sum(p.payout for p in purchases)
    wins = sum(1 for p in purchases if p.won)
    return {
        "races": len(purchases),
        "wins": wins,
        "hit_pct": 100 * wins / len(purchases) if purchases else 0.0,
        "roi_pct": 100 * total_payout / total_cost if total_cost > 0 else 0.0,
        "pl": total_payout - total_cost,
        "cost": total_cost,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-log", required=True,
                        help="Path to tune log (or trials.json sidecar)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top trials to evaluate (default: 5). "
                             "Ignored when --trials is given.")
    parser.add_argument("--trials", default=None,
                        help="Explicit comma-separated trial numbers to evaluate "
                             "(e.g. '17,19'). Overrides --top-n.")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated LightGBM random_state seeds")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--end-date", default="2026-01-01",
                        help="Train data cutoff (default: 2026-01-01 OOS guard)")
    parser.add_argument("--val-months", type=int, default=2)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--gap12-th", type=float, default=None,
                        help="Override gap12 threshold for OOS eval. Defaults to "
                             "the tune's fix_thresholds value (or 0 if absent). "
                             "Useful when the tune was launched before gap12 was "
                             "added: pass --gap12-th 0.04 to evaluate at the "
                             "current production filter.")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}", file=sys.stderr)

    print(f"Parsing {args.tune_log}...", file=sys.stderr)
    log_info = parse_tune_log(Path(args.tune_log))
    fix_th = log_info["fix_thresholds"]
    gap23_th = fix_th.get("gap23", 0.13)
    ev_th = fix_th.get("ev", 0.0)
    conc_default = fix_th.get("top3_conc", 0.0)
    gap12_default = fix_th.get("gap12", 0.0)

    # log_info["trials"] is {trial_number: {growth, params, user_attrs}}.
    if args.trials:
        # Explicit selection: trial numbers given by user, preserve their order.
        wanted = [int(t.strip()) for t in args.trials.split(",")]
        trial_items = []
        for num in wanted:
            if num in log_info["trials"]:
                trial_items.append((num, log_info["trials"][num]))
            else:
                print(f"WARN: trial #{num} not in log, skipping", file=sys.stderr)
    else:
        # Top N by Kelly, with a volume floor to exclude winner's curse.
        # Kelly = mean(log(fold_ROI)) explodes when a fold catches a single
        # lucky hit with few races (1 hit / 10 races → ROI 300%+ → log(3) = 1.1).
        # 4-14 tune confirmed trial #12 with 51 races had Kelly 1.43 (fake)
        # while OOS mean was only +2,650. p2_v2-class trials land at 500-1200
        # Phase 1 races; PHASE2_MIN_RACES=500 excludes exploit-volume trials.
        #
        # Why Kelly over growth: 4-12 tune showed growth #1 (#266, kelly 0.28)
        # was OOS-worst (+6,618 mean, -2,860 min) while kelly #1 (#294, growth
        # 4位) became p2_v2 production. Kelly after the volume filter is the
        # best Phase 1 signal for OOS performance in our data.
        def _kelly(kv):
            ua = kv[1].get("user_attrs") or {}
            k = ua.get("kelly")
            races = ua.get("total_races") or 0
            # Volume gate: low-volume trials have unreliable Kelly.
            if races < PHASE2_MIN_RACES:
                return float("-inf")
            return k if k is not None else float("-inf")
        trial_items = sorted(
            log_info["trials"].items(), key=_kelly, reverse=True,
        )[: args.top_n]
        # Drop trials with -inf (all below volume floor). Should be rare
        # but defensive: if the tune is entirely low-volume, Phase 2 just
        # evaluates the survivors (fewer than top_n).
        trial_items = [
            (num, t) for num, t in trial_items
            if ((t.get("user_attrs") or {}).get("total_races") or 0) >= PHASE2_MIN_RACES
        ]
    if not trial_items:
        print("No trials found.", file=sys.stderr)
        sys.exit(1)

    print(f"Selected top {len(trial_items)} trials. "
          f"Loading features (end_date={args.end_date})...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        df_train = build_features_df(args.db_path, end_date=args.end_date)
        df_full = build_features_df(args.db_path)

    val_start = pd.Timestamp(args.end_date) - pd.DateOffset(months=args.val_months)
    val_mask = df_train["race_date"] >= str(val_start.date())

    # feature_means must come from df_train (the actual training data) to
    # match what train_dev_model.train_one() persists into the saved meta.
    # Computing from df_full[<from_date] would use a different cache build
    # and introduce ~5% numerical drift in OOS evaluation.
    feature_means = {
        c: float(df_train[c].astype("float64").mean()) for c in FEATURES
    }

    odds = _load_trifecta_odds(args.db_path)

    # Run trial × seed grid
    results: list[dict] = []
    for ti, (trial_num, trial) in enumerate(trial_items, 1):
        params = trial["params"]
        hp, lr, n_est_upper, conc_th, gap12_th = params_to_hp(
            params, conc_default=conc_default, gap12_default=gap12_default,
        )
        if args.gap12_th is not None:
            gap12_th = args.gap12_th
        user_attrs = trial.get("user_attrs", {}) or {}
        avg_best_iter = user_attrs.get("avg_best_iter")
        effective_n_est = int(avg_best_iter) if avg_best_iter else n_est_upper
        # relevance: search-space param > user_attrs (when fixed via --relevance) > default
        relevance = (
            params.get("relevance")
            or user_attrs.get("relevance")
            or "podium"
        )

        trial_pls: list[float] = []
        per_seed_rows: list[tuple[int, dict]] = []
        t0 = time.time()
        for seed in seeds:
            model = _train_one_seed(
                df_train, val_mask, hp, lr, effective_n_est, relevance, seed,
            )
            stats = _evaluate(
                model, df_full, feature_means, conc_th, gap12_th, gap23_th, ev_th,
                odds, args.from_date, args.to_date,
            )
            trial_pls.append(stats["pl"])
            per_seed_rows.append((seed, stats))
        elapsed = time.time() - t0

        mean_pl = statistics.mean(trial_pls)
        std_pl = statistics.stdev(trial_pls) if len(trial_pls) > 1 else 0.0
        # Hit rate is critical for Kelly stability: a trial with high mean P/L
        # but low hit rate accumulates consecutive losses, blowing up bankroll
        # variance under fractional Kelly. Track mean hit rate alongside P/L.
        trial_hits = [s["hit_pct"] for _, s in per_seed_rows]
        mean_hit = statistics.mean(trial_hits)
        results.append({
            "trial": trial_num,
            "wf_cv_value": trial.get("growth"),
            "mean_pl": mean_pl,
            "std_pl": std_pl,
            "mean_hit": mean_hit,
            "per_seed": per_seed_rows,
            "elapsed": elapsed,
            "relevance": relevance,
            "n_est": effective_n_est,
            "lr": lr,
        })
        print(
            f"  [{ti}/{len(trial_items)}] trial #{trial_num}: "
            f"mean P/L {mean_pl:+,.0f} std {std_pl:,.0f} ({elapsed:.0f}s)",
            file=sys.stderr,
        )

    # Output: per-trial summary then per-seed details.
    # Sort by stability_score = mean - std (penalize variance). This
    # automatically demotes winner's curse trials and surfaces robust HPs.
    for r in results:
        r["stability_score"] = r["mean_pl"] - r["std_pl"]

    print(f"\n=== Seed stability check ===")
    print(f"Tune log: {args.tune_log}")
    print(f"Period: {args.from_date} ~ {args.to_date}")
    print(f"Seeds: {seeds}")
    print(f"Ranked by stability_score = mean - std (higher = more robust)")
    print(f"Hit% is mean across seeds. Hit% < 10% destabilizes Kelly betting.")
    print()
    print(
        f"{'trial':>6} {'WF-CV':>9} {'stability':>11} {'mean P/L':>11} "
        f"{'std P/L':>10} {'min':>10} {'max':>10} {'hit%':>6} {'rel':>10}"
    )
    print("-" * 92)
    for r in sorted(results, key=lambda r: -r["stability_score"]):
        pls = [s["pl"] for _, s in r["per_seed"]]
        # Mark hit rate < 10% with a warning glyph (Kelly instability risk)
        hit_marker = " " if r["mean_hit"] >= 10.0 else "!"
        print(
            f"  #{r['trial']:>3} {r['wf_cv_value']:>9.6f} "
            f"{r['stability_score']:>+11,.0f} "
            f"{r['mean_pl']:>+11,.0f} {r['std_pl']:>10,.0f} "
            f"{min(pls):>+10,.0f} {max(pls):>+10,.0f} "
            f"{r['mean_hit']:>5.1f}{hit_marker} {r['relevance']:>9}"
        )

    print()
    print("=== Per-seed details ===")
    for r in results:
        print(f"\nTrial #{r['trial']} (lr={r['lr']:.4f} n_est={r['n_est']}):")
        print(
            f"  {'seed':>5} {'races':>6} {'wins':>5} {'hit%':>6} "
            f"{'ROI%':>6} {'P/L':>11}"
        )
        for seed, stats in r["per_seed"]:
            print(
                f"  {seed:>5} {stats['races']:>6} {stats['wins']:>5} "
                f"{stats['hit_pct']:>5.1f}% {stats['roi_pct']:>5.0f}% "
                f"{stats['pl']:>+11,.0f}"
            )


if __name__ == "__main__":
    main()
