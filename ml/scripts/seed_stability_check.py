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
from scripts.analyze_model import evaluate_period, evaluate_period_sweep
from scripts.train_dev_model import parse_tune_log, params_to_hp

FIELD_SIZE = 6
DEFAULT_SEEDS = [42, 100, 200, 300, 400]

# Minimum WF-CV races (across all folds) for a trial to be a Phase 2
# candidate. Kelly = mean(log(fold_ROI)) explodes for low-volume trials
# that catch a single lucky hit per fold, so a volume floor is needed.
# Production-class trials land around 500-1200 Phase 1 races.
PHASE2_MIN_RACES = 500


def _compute_month_windows(
    start_date: str, end_date: str
) -> list[tuple[str, str, str]]:
    """Return (name, from, to) for calendar months in [start, end].

    The last month in the range may be partial (clamped to end_date) so the
    full evaluation period is always covered. stability_score is computed
    from per-seed full-period sums (sum across all returned windows per
    seed), so the partial last month contributes correctly to ranking.
    Per-month breakdown is used for diagnostic display only.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    windows: list[tuple[str, str, str]] = []
    while cur < end:
        nxt = cur + pd.DateOffset(months=1)
        win_end = min(nxt, end)
        if cur >= start and cur < end:
            windows.append(
                (cur.strftime("%Y-%m"), cur.strftime("%Y-%m-%d"),
                 win_end.strftime("%Y-%m-%d"))
            )
        cur = nxt
    return windows


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


def _evaluate_sweep(
    model,
    df_eval: pd.DataFrame,
    feature_means: dict[str, float],
    conc_th: float,
    gap12_th: float,
    gap23_th: float,
    ev_levels: list[float],
    odds: dict,
    from_date: str,
    to_date: str,
) -> dict[float, dict]:
    """EV sweep version of _evaluate.

    Predict once and re-filter tickets per ev level. ev_levels の各 ev で
    同じ stats 構造を返すので、呼び出し側は既存の集計ロジックを ev 軸に
    拡張するだけで再利用できる。

    Returns {ev_th: {races, wins, hit_pct, roi_pct, pl, cost}}
    """
    # ev_th=0.0 として meta を作るが、evaluate_period_sweep は strategy.ev_threshold
    # を参照しないので値はダミー。conc/gap12/gap23 は filter で使われる。
    meta = _build_strategy_meta(feature_means, conc_th, gap12_th, gap23_th, 0.0)
    purchases_by_ev, _ = evaluate_period_sweep(
        model, meta, df_eval, odds, from_date, to_date, ev_levels,
    )
    out: dict[float, dict] = {}
    for ev_th, purchases in purchases_by_ev.items():
        total_cost = sum(p.cost for p in purchases)
        total_payout = sum(p.payout for p in purchases)
        wins = sum(1 for p in purchases if p.won)
        out[ev_th] = {
            "races": len(purchases),
            "wins": wins,
            "hit_pct": 100 * wins / len(purchases) if purchases else 0.0,
            "roi_pct": 100 * total_payout / total_cost if total_cost > 0 else 0.0,
            "pl": total_payout - total_cost,
            "cost": total_cost,
        }
    return out


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
    parser.add_argument(
        "--from", dest="from_date", default=None,
        help="Evaluation range start (default: --end-date). The actual "
             "windows are complete calendar months within [from, to).",
    )
    parser.add_argument(
        "--to", dest="to_date", default=None,
        help="Evaluation range end (default: today).",
    )
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
    parser.add_argument("--ev-threshold", type=float, default=None,
                        help="Override ev_threshold for OOS eval. Defaults to "
                             "the tune's fix_thresholds value (or 0 if absent). "
                             "Required for cross-tune comparisons when logs use "
                             "different ev defaults, or to evaluate an older tune's "
                             "trials at the current production ev (e.g. -0.25).")
    parser.add_argument("--ev-sweep", default=None,
                        help="Comma-separated ev list for peak-EV search "
                             "(e.g. '0.0,-0.05,-0.10,-0.15,-0.20,-0.25,-0.30,-0.35'). "
                             "When given, each trial × seed is evaluated at all ev "
                             "levels; the peak ev (max stability_score = mean-std of "
                             "per-seed full-OOS P/L) is chosen per trial and ranking "
                             "is reported at each trial's peak. Mutually exclusive "
                             "with --ev-threshold.")
    args = parser.parse_args()

    if args.ev_sweep and args.ev_threshold is not None:
        parser.error("--ev-sweep and --ev-threshold are mutually exclusive")

    # Default --from = --end-date (start of OOS), --to = today
    if args.from_date is None:
        args.from_date = args.end_date
    if args.to_date is None:
        args.to_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}", file=sys.stderr)

    ev_sweep_levels: list[float] | None = None
    if args.ev_sweep:
        ev_sweep_levels = sorted(
            {float(x.strip()) for x in args.ev_sweep.split(",")},
            reverse=True,
        )
        print(f"EV sweep levels: {ev_sweep_levels}", file=sys.stderr)

    print(f"Parsing {args.tune_log}...", file=sys.stderr)
    log_info = parse_tune_log(Path(args.tune_log))
    fix_th = log_info["fix_thresholds"]
    gap23_th = fix_th.get("gap23", 0.13)
    ev_th = args.ev_threshold if args.ev_threshold is not None else fix_th.get("ev", 0.0)
    conc_default = fix_th.get("top3_conc", 0.0)
    gap12_default = fix_th.get("gap12", 0.0)
    if args.ev_threshold is not None:
        print(
            f"ev_threshold override: tune log had ev={fix_th.get('ev', 0.0)}, "
            f"evaluating with ev={args.ev_threshold}",
            file=sys.stderr,
        )

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
        # lucky hit with few races (1 hit / 10 races → ROI 300%+ → log(3) = 1.1),
        # so low-volume trials can score fake-high Kelly while having poor OOS.
        # Kelly after the volume filter is the most reliable Phase 1 signal
        # for OOS performance; raw growth ranking is dominated by high-variance
        # trials that got lucky on one fold.
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

    month_windows = _compute_month_windows(args.from_date, args.to_date)
    if not month_windows:
        print(
            f"ERROR: empty evaluation range [{args.from_date}, {args.to_date}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Evaluation: full period {args.from_date}〜{args.to_date} "
        f"({len(month_windows)} month bucket(s) for diagnostic: "
        f"{', '.join(w[0] for w in month_windows)})",
        file=sys.stderr,
    )

    # Run trial × seed × window grid
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
        relevance = (
            params.get("relevance")
            or user_attrs.get("relevance")
            or "podium"
        )

        # per_seed_windows[i][wname] = stats dict for seed[i], window wname.
        # In sweep mode, per_seed_windows_sweep[i][wname][ev] holds per-ev
        # stats and per_seed_windows is filled from it at the peak ev below.
        per_seed_windows: list[dict[str, dict]] = []
        per_seed_windows_sweep: list[dict[str, dict[float, dict]]] = []
        t0 = time.time()
        for seed in seeds:
            model = _train_one_seed(
                df_train, val_mask, hp, lr, effective_n_est, relevance, seed,
            )
            if ev_sweep_levels:
                seed_window_ev_stats: dict[str, dict[float, dict]] = {}
                for wname, wfrom, wto in month_windows:
                    seed_window_ev_stats[wname] = _evaluate_sweep(
                        model, df_full, feature_means, conc_th, gap12_th,
                        gap23_th, ev_sweep_levels, odds, wfrom, wto,
                    )
                per_seed_windows_sweep.append(seed_window_ev_stats)
            else:
                seed_window_stats: dict[str, dict] = {}
                for wname, wfrom, wto in month_windows:
                    seed_window_stats[wname] = _evaluate(
                        model, df_full, feature_means, conc_th, gap12_th,
                        gap23_th, ev_th, odds, wfrom, wto,
                    )
                per_seed_windows.append(seed_window_stats)
        elapsed = time.time() - t0

        # In sweep mode, pick peak ev per trial (max stability across seeds,
        # tie-break by higher ev = more conservative) and reconstruct
        # per_seed_windows so downstream aggregation is identical.
        trial_peak_ev: float | None = None
        trial_ev_profile: dict[float, dict] | None = None
        if ev_sweep_levels:
            trial_ev_profile = {}
            for ev in ev_sweep_levels:
                pls = [
                    sum(s[wname][ev]["pl"] for wname, _, _ in month_windows)
                    for s in per_seed_windows_sweep
                ]
                mean_pl_ev = statistics.mean(pls)
                std_pl_ev = (
                    statistics.stdev(pls) if len(pls) > 1 else 0.0
                )
                trial_ev_profile[ev] = {
                    "mean_pl": mean_pl_ev,
                    "std_pl": std_pl_ev,
                    "stability": mean_pl_ev - std_pl_ev,
                }
            trial_peak_ev = max(
                ev_sweep_levels,
                key=lambda e: (trial_ev_profile[e]["stability"], e),
            )
            per_seed_windows = [
                {
                    wname: s[wname][trial_peak_ev]
                    for wname, _, _ in month_windows
                }
                for s in per_seed_windows_sweep
            ]

        # Aggregate per-window across seeds
        window_summary: dict[str, dict] = {}
        for wname, wfrom, wto in month_windows:
            pls = [s[wname]["pl"] for s in per_seed_windows]
            hits = [s[wname]["hit_pct"] for s in per_seed_windows]
            races = [s[wname]["races"] for s in per_seed_windows]
            wins = [s[wname]["wins"] for s in per_seed_windows]
            days = (pd.Timestamp(wto) - pd.Timestamp(wfrom)).days
            window_summary[wname] = {
                "mean_pl": statistics.mean(pls),
                "std_pl": statistics.stdev(pls) if len(pls) > 1 else 0.0,
                "mean_hit": statistics.mean(hits),
                "mean_races": statistics.mean(races),
                "mean_wins": statistics.mean(wins),
                "days": days,
                "min_pl": min(pls),
                "max_pl": max(pls),
            }

        # Full OOS = sum of monthly windows per seed, then aggregate across seeds
        seed_total_pls = [
            sum(s[wname]["pl"] for wname, _, _ in month_windows)
            for s in per_seed_windows
        ]
        seed_total_races = [
            sum(s[wname]["races"] for wname, _, _ in month_windows)
            for s in per_seed_windows
        ]
        seed_total_wins = [
            sum(s[wname]["wins"] for wname, _, _ in month_windows)
            for s in per_seed_windows
        ]
        seed_total_hits = [
            100 * w / r if r > 0 else 0.0
            for w, r in zip(seed_total_wins, seed_total_races)
        ]

        mean_pl = statistics.mean(seed_total_pls)
        std_pl = statistics.stdev(seed_total_pls) if len(seed_total_pls) > 1 else 0.0
        mean_hit = statistics.mean(seed_total_hits)
        mean_races = statistics.mean(seed_total_races)
        total_days = sum(w["days"] for w in window_summary.values())
        bets_per_day = mean_races / total_days if total_days > 0 else 0.0

        min_window_pl = min(w["mean_pl"] for w in window_summary.values())
        worst_window_hit = min(w["mean_hit"] for w in window_summary.values())

        results.append({
            "trial": trial_num,
            "wf_cv_growth": (trial.get("user_attrs") or {}).get("growth")
                or trial.get("growth"),
            "wf_cv_kelly": (trial.get("user_attrs") or {}).get("kelly"),
            "mean_pl": mean_pl,
            "std_pl": std_pl,
            "mean_hit": mean_hit,
            "mean_races": mean_races,
            "bets_per_day": bets_per_day,
            "total_days": total_days,
            "min_window_pl": min_window_pl,
            "worst_window_hit": worst_window_hit,
            "window_summary": window_summary,
            "seed_total_pls": seed_total_pls,
            "per_seed_windows": per_seed_windows,
            "elapsed": elapsed,
            "hp": hp,
            "relevance": relevance,
            "n_est": effective_n_est,
            "lr": lr,
            "peak_ev": trial_peak_ev,
            "ev_profile": trial_ev_profile,
        })
        peak_tag = (
            f" peak_ev={trial_peak_ev:+.2f}" if trial_peak_ev is not None else ""
        )
        print(
            f"  [{ti}/{len(trial_items)}] trial #{trial_num}:{peak_tag} "
            f"mean P/L {mean_pl:+,.0f} std {std_pl:,.0f} "
            f"min_win {min_window_pl:+,.0f} ({elapsed:.0f}s)",
            file=sys.stderr,
        )

    # Sort by stability_score = mean - std (full OOS across all windows).
    # This penalizes seed variance. Per-window breakdown below surfaces
    # regime fragility as diagnostic info; no hard filter is applied.
    for r in results:
        r["stability_score"] = r["mean_pl"] - r["std_pl"]

    window_names = [w[0] for w in month_windows]

    def _fmt_kelly(k):
        return f"{k:>7.4f}" if k is not None else "      ?"

    sweep_on = ev_sweep_levels is not None

    print("\n=== Seed stability check ===")
    print(f"Tune log: {args.tune_log}")
    print(f"Period: {args.from_date} ~ {args.to_date}")
    print(f"Seeds: {seeds}")
    print(f"Windows: {', '.join(window_names)}")
    if sweep_on:
        print(f"EV sweep: {ev_sweep_levels} (peak ev reported per trial)")
    print("Ranked by stability_score = mean - std (higher = more robust)")
    print()
    peak_col = f"{'peak ev':>8} " if sweep_on else ""
    print(
        f"{'trial':>6} {peak_col}{'WF-CV g':>9} {'WF-CV k':>8} {'stability':>11} "
        f"{'mean P/L':>11} {'std P/L':>10} {'min win':>11} "
        f"{'worst h%':>9} {'full h%':>8} {'bets':>6} {'b/day':>6} {'rel':>8}"
    )
    print("-" * (130 if sweep_on else 120))
    for r in sorted(results, key=lambda r: -r["stability_score"]):
        peak_val = (
            f"{r['peak_ev']:>+8.2f} "
            if sweep_on and r["peak_ev"] is not None
            else ""
        )
        print(
            f"  #{r['trial']:>3} {peak_val}"
            f"{r['wf_cv_growth']:>9.6f} "
            f"{_fmt_kelly(r['wf_cv_kelly'])} "
            f"{r['stability_score']:>+11,.0f} "
            f"{r['mean_pl']:>+11,.0f} {r['std_pl']:>10,.0f} "
            f"{r['min_window_pl']:>+11,.0f} "
            f"{r['worst_window_hit']:>8.2f}% "
            f"{r['mean_hit']:>7.2f}% "
            f"{r['mean_races']:>6.0f} "
            f"{r['bets_per_day']:>6.2f} "
            f"{r['relevance']:>8}"
        )

    # Per-trial detail: HP + per-window rows + per-seed rows
    print("\n=== Per-trial detail (ranked by stability) ===")
    for r in sorted(results, key=lambda r: -r["stability_score"]):
        hp = r["hp"]
        peak_hdr = (
            f"  peak_ev={r['peak_ev']:+.2f}" if r.get("peak_ev") is not None else ""
        )
        print(
            f"\nTrial #{r['trial']}  "
            f"stability={r['stability_score']:+,.0f}  "
            f"bets/day={r['bets_per_day']:.2f}{peak_hdr}"
        )
        print(
            f"  HP: nl={hp.get('num_leaves')} md={hp.get('max_depth')} "
            f"mc={hp.get('min_child_samples')} "
            f"sub={hp.get('subsample'):.3f} col={hp.get('colsample_bytree'):.3f} "
            f"ra={hp.get('reg_alpha'):.1e} rl={hp.get('reg_lambda'):.4f} "
            f"lr={r['lr']:.4f} n_est={r['n_est']} rel={r['relevance']}"
        )
        if r.get("ev_profile"):
            print(f"  {'ev sweep':>10}")
            print(
                f"  {'ev':>6} {'mean P/L':>11} {'std P/L':>10} {'stability':>11}"
            )
            for ev, prof in sorted(
                r["ev_profile"].items(), key=lambda kv: -kv[0]
            ):
                marker = " *" if ev == r["peak_ev"] else "  "
                print(
                    f"  {ev:>+6.2f} {prof['mean_pl']:>+11,.0f} "
                    f"{prof['std_pl']:>10,.0f} "
                    f"{prof['stability']:>+11,.0f}{marker}"
                )
        print(
            f"  {'window':>8} {'days':>5} {'races':>6} {'wins':>5} "
            f"{'hit%':>6} {'mean P/L':>11} {'std':>9}"
        )
        for wname in window_names:
            w = r["window_summary"][wname]
            print(
                f"  {wname:>8} {w['days']:>5} "
                f"{w['mean_races']:>6.0f} {w['mean_wins']:>5.1f} "
                f"{w['mean_hit']:>5.2f}% "
                f"{w['mean_pl']:>+11,.0f} {w['std_pl']:>9,.0f}"
            )
        print(f"  {'per-seed':>8}")
        for si, seed in enumerate(seeds):
            total_races = sum(
                r["per_seed_windows"][si][w]["races"] for w in window_names
            )
            total_wins = sum(
                r["per_seed_windows"][si][w]["wins"] for w in window_names
            )
            total_pl = r["seed_total_pls"][si]
            hit_pct = 100 * total_wins / total_races if total_races > 0 else 0.0
            print(
                f"  seed={seed:<5} races={total_races:>4} "
                f"wins={total_wins:>3} hit={hit_pct:>5.2f}% "
                f"P/L={total_pl:>+11,.0f}"
            )


if __name__ == "__main__":
    main()
