"""Backtest trifecta X-allflow strategy.

Evaluates 3連単 strategy: fix 1st place (anti-favorite pick),
buy all 2nd-3rd combinations (20pt allflow).

Modes:
    --from/--to:  Period backtest with daily breakdown
    --wfcv:       Walk-forward cross-validation (4 folds)
    --ev-sweep:   EV threshold sweep across WF-CV

Usage:
    uv run --directory ml python -m scripts.backtest_trifecta --from 2026-03-21 --to 2026-04-04
    uv run --directory ml python -m scripts.backtest_trifecta --wfcv
    uv run --directory ml python -m scripts.backtest_trifecta --ev-sweep
"""

import argparse
import contextlib
import io
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import train_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.evaluate import evaluate_trifecta_strategy
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits

# train_model / train_boat1_model are only used by --wfcv and --ev-sweep (WF-CV retraining).
# --from/--to loads saved production models instead.


def load_data(db_path: str):
    """Load features, odds, and finish data."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(db_path)

    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '3連単'"
    ).fetchall()
    trifecta_odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    # Trifecta-implied win probability
    tri_win_prob: dict[tuple[int, int], float] = defaultdict(float)
    for r in rows:
        rid, combo, odds = int(r[0]), r[1], float(r[2])
        if odds <= 0:
            continue
        first_boat = int(combo.split("-")[0])
        tri_win_prob[(rid, first_boat)] += 0.75 / odds

    # Exacta odds (2連単)
    exacta_rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type = '2連単'"
    ).fetchall()
    exacta_odds = {(int(r[0]), r[1]): float(r[2]) for r in exacta_rows}

    conn.close()

    # Finish map and date map
    finish_map: dict[tuple[int, int], int] = {}
    race_date_map: dict[int, str] = {}
    for _, row in (
        df[["race_id", "boat_number", "finish_position", "race_date"]]
        .drop_duplicates()
        .iterrows()
    ):
        if pd.notna(row["finish_position"]):
            finish_map[(int(row["race_id"]), int(row["boat_number"]))] = int(
                row["finish_position"]
            )
        race_date_map[int(row["race_id"])] = str(row["race_date"])

    return df, trifecta_odds, dict(tri_win_prob), finish_map, race_date_map, exacta_odds


def print_daily(results: list[dict], label: str = ""):
    """Print daily breakdown from per-race results."""
    if label:
        print(f"\n=== {label} ===")

    TICKETS_PER_BET = 20  # allflow

    daily: dict[str, dict] = defaultdict(
        lambda: {"races": 0, "wins": 0, "payout": 0.0}
    )
    for r in results:
        d = daily[r["date"]]
        d["races"] += 1
        if r["pick_1st"] and r["allflow_odds"] > 0:
            d["wins"] += 1
            d["payout"] += r["allflow_odds"]

    cum = 0.0
    total_r = total_w = 0
    total_cost = 0.0
    total_p = 0.0
    win_days = 0

    for date in sorted(daily.keys()):
        d = daily[date]
        cost = d["races"] * TICKETS_PER_BET
        pl = d["payout"] - cost
        cum += pl
        total_r += d["races"]
        total_cost += cost
        total_w += d["wins"]
        total_p += d["payout"]
        if pl > 0:
            win_days += 1
        marker = "+" if pl > 0 else "-"
        hit_pct = d["wins"] / d["races"] * 100 if d["races"] > 0 else 0
        print(
            f"  {date}: {d['races']:>2}R {d['wins']}W({hit_pct:>3.0f}%) "
            f"P/L {pl:>+8.1f} cum {cum:>+9.1f} {marker}"
        )

    days = len(daily)
    if total_cost > 0:
        roi = total_p / total_cost
        print(f"\n  {days} days, {total_r}R(×{TICKETS_PER_BET}pt), {total_w}W")
        print(f"  Hit: {total_w/total_r:.0%}, ROI: {roi:.0%}, P/L: {total_p - total_cost:+.1f}")
        print(f"  R/d: {total_r/days:.1f}, Win days: {win_days}/{days} ({win_days/days:.0%})")

    return {
        "days": days,
        "races": total_r,
        "tickets_per_bet": TICKETS_PER_BET,
        "wins": total_w,
        "payout": total_p,
        "cost": total_cost,
        "roi": total_p / total_cost if total_cost > 0 else 0,
    }


def run_period(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds=None, *, ranking_params=None, boat1_params=None):
    """Period backtest using saved production models (no retraining)."""
    from boatrace_tipster_ml.boat1_model import load_boat1_model
    from boatrace_tipster_ml.model import load_model

    test_df = df[
        (df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)
    ]

    print(f"Loading saved models from {args.model_dir}...", file=sys.stderr)
    b1_model = load_boat1_model(f"{args.model_dir}/boat1")
    rank_model = load_model(f"{args.model_dir}/ranking")

    # Inference with saved models
    with contextlib.redirect_stdout(io.StringIO()):
        X_b1, _, meta_b1 = reshape_to_boat1(test_df)
    b1_probs = b1_model.predict_proba(X_b1)[:, 1]

    X_rank, _, meta_rank = prepare_feature_matrix(test_df)
    rank_scores = rank_model.predict(X_rank)

    results = evaluate_trifecta_strategy(
        b1_probs=b1_probs,
        meta_b1=meta_b1,
        rank_scores=rank_scores,
        meta_rank=meta_rank,
        finish_map=finish_map,
        trifecta_odds=trifecta_odds,
        tri_win_prob=tri_win_prob,
        b1_threshold=args.b1_threshold,
        ev_threshold=args.ev_threshold,
        r2_ev_threshold=args.r2_threshold,
        race_date_map=race_date_map,
        per_race=True,
    )

    if args.json:
        # Compute summary without printing
        from collections import defaultdict as _dd
        TICKETS = 20
        daily: dict = _dd(lambda: {"races": 0, "wins": 0, "payout": 0.0})
        for r in results:
            d = daily[r["date"]]
            d["races"] += 1
            if r["pick_1st"] and r.get("allflow_odds", 0) > 0:
                d["wins"] += 1
                d["payout"] += r["allflow_odds"]
        total_r = sum(d["races"] for d in daily.values())
        total_w = sum(d["wins"] for d in daily.values())
        total_cost = total_r * TICKETS
        total_p = sum(d["payout"] for d in daily.values())
        summary = {
            "days": len(daily), "races": total_r, "tickets_per_bet": TICKETS,
            "wins": total_w, "payout": total_p, "cost": total_cost,
            "roi": total_p / total_cost if total_cost > 0 else 0,
        }
        json.dump(
            {"params": vars(args), "summary": summary, "races": results},
            sys.stdout,
            ensure_ascii=False,
            default=str,
        )
    else:
        print_daily(
            results,
            f"X-allflow(20pt), b1<{args.b1_threshold:.0%}, EV>={args.ev_threshold:.0%} "
            f"({args.from_date} ~ {args.to_date})",
        )


def run_wfcv(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds=None, *, ranking_params=None, boat1_params=None):
    """Walk-forward CV — uses same data flow as tune_trifecta.py."""
    rp = ranking_params or {}
    bp = boat1_params or {}

    # Prepare feature matrices once on full data, then slice (same as tune)
    X_rank, y_rank, meta_rank = prepare_feature_matrix(df)
    folds = walk_forward_splits(
        X_rank, y_rank, meta_rank,
        n_folds=args.n_folds, fold_months=args.fold_months,
    )

    TICKETS_PER_BET = 20
    print(f"WF-CV: {len(folds)} folds, b1<{args.b1_threshold:.0%}, EV>={args.ev_threshold:.0%}")

    # Pre-train boat1 models per fold (same as tune)
    fold_b1_data = []
    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]

        train_fold = df[df["race_date"] < test_from]
        test_fold = df[(df["race_date"] >= test_from) & (df["race_date"] < test_to)]

        dates = sorted(train_fold["race_date"].unique())
        val_start = dates[max(0, len(dates) - 60)]

        with contextlib.redirect_stdout(io.StringIO()):
            X_b1_tr, y_b1_tr, _ = reshape_to_boat1(train_fold[train_fold["race_date"] < val_start])
            X_b1_v, y_b1_v, _ = reshape_to_boat1(train_fold[train_fold["race_date"] >= val_start])
            X_b1_te, _, meta_b1_te = reshape_to_boat1(test_fold)
            b1_model, _ = train_boat1_model(
                X_b1_tr, y_b1_tr, X_b1_v, y_b1_v,
                n_estimators=bp.get("n_estimators"),
                learning_rate=bp.get("learning_rate"),
                extra_params=bp.get("extra_params"),
            )

        b1_probs = b1_model.predict_proba(X_b1_te)[:, 1]
        fold_b1_data.append({"b1_probs": b1_probs, "meta_b1": meta_b1_te})

    # Train ranking + evaluate per fold (same data flow as tune)
    fold_rois = []
    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]
        print(f"\nFold {i+1}: {test_from} ~ {test_to}", file=sys.stderr)

        # Train ranking model using pre-sliced X/y/meta from walk_forward_splits
        with contextlib.redirect_stdout(io.StringIO()):
            rank_model, _ = train_model(
                fold["train"]["X"], fold["train"]["y"], fold["train"]["meta"],
                fold["val"]["X"], fold["val"]["y"], fold["val"]["meta"],
                n_estimators=rp.get("n_estimators"),
                learning_rate=rp.get("learning_rate"),
                extra_params=rp.get("extra_params"),
                relevance_scheme=rp.get("relevance_scheme", "linear"),
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
            b1_threshold=args.b1_threshold,
            ev_threshold=args.ev_threshold,
            r2_ev_threshold=args.r2_threshold,
        )
        fold_rois.append(result["roi"])

        print(f"  Fold {i+1}: {result['races']}R(×{TICKETS_PER_BET}pt), {result['wins']}W, ROI {result['roi']:.0%}")

    print(f"\n{'='*50}")
    print(f"Mean ROI: {np.mean(fold_rois):.0%} ± {np.std(fold_rois):.0%}")
    print(f"Min: {np.min(fold_rois):.0%}, Max: {np.max(fold_rois):.0%}")
    sharpe = (np.mean(fold_rois) - 1) / np.std(fold_rois) if np.std(fold_rois) > 0 else 0
    print(f"Sharpe: {sharpe:.2f}")


def run_ev_sweep(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds=None, *, ranking_params=None, boat1_params=None):
    """EV threshold sweep across WF-CV — same data flow as tune."""
    rp = ranking_params or {}
    bp = boat1_params or {}

    X_rank, y_rank, meta_rank = prepare_feature_matrix(df)
    folds = walk_forward_splits(
        X_rank, y_rank, meta_rank,
        n_folds=args.n_folds, fold_months=args.fold_months,
    )

    print(f"EV sweep: {len(folds)} folds")

    # Pre-train boat1 + ranking models per fold
    fold_b1_data = []
    fold_rank_scores = []
    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]

        train_fold = df[df["race_date"] < test_from]
        test_fold = df[(df["race_date"] >= test_from) & (df["race_date"] < test_to)]
        dates = sorted(train_fold["race_date"].unique())
        val_start = dates[max(0, len(dates) - 60)]

        print(f"  Training fold {i+1}: {test_from} ~ {test_to}", file=sys.stderr)
        with contextlib.redirect_stdout(io.StringIO()):
            # Boat1
            X_b1_tr, y_b1_tr, _ = reshape_to_boat1(train_fold[train_fold["race_date"] < val_start])
            X_b1_v, y_b1_v, _ = reshape_to_boat1(train_fold[train_fold["race_date"] >= val_start])
            X_b1_te, _, meta_b1_te = reshape_to_boat1(test_fold)
            b1_model, _ = train_boat1_model(
                X_b1_tr, y_b1_tr, X_b1_v, y_b1_v,
                n_estimators=bp.get("n_estimators"),
                learning_rate=bp.get("learning_rate"),
                extra_params=bp.get("extra_params"),
            )
            # Ranking (using pre-sliced data)
            rank_model, _ = train_model(
                fold["train"]["X"], fold["train"]["y"], fold["train"]["meta"],
                fold["val"]["X"], fold["val"]["y"], fold["val"]["meta"],
                n_estimators=rp.get("n_estimators"),
                learning_rate=rp.get("learning_rate"),
                extra_params=rp.get("extra_params"),
                relevance_scheme=rp.get("relevance_scheme", "linear"),
                early_stopping_rounds=50,
            )

        b1_probs = b1_model.predict_proba(X_b1_te)[:, 1]
        fold_b1_data.append({"b1_probs": b1_probs, "meta_b1": meta_b1_te})
        fold_rank_scores.append(rank_model.predict(fold["test"]["X"]))

    TICKETS_PER_BET = 20
    ev_thresholds = [-0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{'EV':>6} | {'F1':>5} {'F2':>5} {'F3':>5} {'F4':>5} | {'Mean':>5} {'Std':>4} {'Min':>5} {'Shrp':>5} {'R/d':>5}")
    print("-" * 75)

    for ev_thr in ev_thresholds:
        fold_rois = []
        fold_rpd = []

        for i, fold in enumerate(folds):
            test_dates = fold["period"]["test"]
            test_from, test_to = [d.strip() for d in test_dates.split("~")]
            test_days = (pd.Timestamp(test_to) - pd.Timestamp(test_from)).days

            result = evaluate_trifecta_strategy(
                b1_probs=fold_b1_data[i]["b1_probs"],
                meta_b1=fold_b1_data[i]["meta_b1"],
                rank_scores=fold_rank_scores[i],
                meta_rank=fold["test"]["meta"],
                finish_map=finish_map,
                trifecta_odds=trifecta_odds,
                tri_win_prob=tri_win_prob,
                b1_threshold=args.b1_threshold,
                ev_threshold=ev_thr,
                r2_ev_threshold=args.r2_threshold,
            )

            roi = result["roi"]
            rpd = result["races"] / test_days if test_days > 0 else 0
            fold_rois.append(roi)
            fold_rpd.append(rpd)

        m = np.mean(fold_rois)
        s = np.std(fold_rois)
        mn = np.min(fold_rois)
        sh = (m - 1) / s if s > 0 else 0
        avg_rpd = np.mean(fold_rpd)
        rstr = " ".join(f"{r:>4.0%}" for r in fold_rois)
        marker = " **" if mn >= 1.0 else " *" if m >= 1.1 else ""
        print(f"{ev_thr:>+5.0%} | {rstr} | {m:>4.0%} {s:>3.0%} {mn:>4.0%} {sh:>5.1f} {avg_rpd:>5.1f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Backtest trifecta X-allflow")
    parser.add_argument("--from", dest="from_date")
    parser.add_argument("--to", dest="to_date")
    parser.add_argument("--wfcv", action="store_true")
    parser.add_argument("--ev-sweep", action="store_true")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--model-dir", default="models/trifecta_v1",
                        help="Model directory to load hyperparams from")
    parser.add_argument("--b1-threshold", type=float, default=None)
    parser.add_argument("--ev-threshold", type=float, default=None)
    parser.add_argument("--r2-threshold", type=float, default=None,
                        help="Rank-2 EV threshold for fallback (default: from model_meta or None)")
    parser.add_argument("--start-date", default=None,
                        help="Earliest date for training data (YYYY-MM-DD)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Load hyperparameters from model_meta.json (single source of truth)
    from boatrace_tipster_ml.model import load_model_meta, load_training_params
    from boatrace_tipster_ml.boat1_model import load_boat1_training_params

    ranking_params = load_training_params(f"{args.model_dir}/ranking")
    boat1_params = load_boat1_training_params(f"{args.model_dir}/boat1")

    # Strategy defaults from model_meta
    rank_meta = load_model_meta(f"{args.model_dir}/ranking")
    strategy = rank_meta.get("strategy", {}) if rank_meta else {}
    if args.b1_threshold is None:
        args.b1_threshold = strategy.get("b1_threshold", 0.42)
    if args.ev_threshold is None:
        args.ev_threshold = strategy.get("ev_threshold", 0.36)
    if args.r2_threshold is None:
        r2 = strategy.get("r2_ev_threshold")
        args.r2_threshold = float(r2) if r2 is not None else None

    r2_label = f", R2>={args.r2_threshold:.0%}" if args.r2_threshold is not None else ""
    print(f"Model: {args.model_dir}", file=sys.stderr)
    print(f"  Ranking: relevance={ranking_params['relevance_scheme']}, "
          f"n_est={ranking_params['n_estimators']}, lr={ranking_params['learning_rate']:.4f}, "
          f"extra={ranking_params['extra_params']}", file=sys.stderr)
    print(f"  Strategy: b1<{args.b1_threshold:.0%}, EV>={args.ev_threshold:.0%}{r2_label}", file=sys.stderr)

    t0 = time.time()
    print("Loading data...", file=sys.stderr)
    df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds = load_data(
        args.db_path
    )
    if args.start_date:
        n_before = len(df) // 6
        df = df[df["race_date"] >= args.start_date].reset_index(drop=True)
        n_after = len(df) // 6
        print(f"Filtered to >= {args.start_date}: {n_before}R → {n_after}R", file=sys.stderr)
    print(f"Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    common = dict(ranking_params=ranking_params, boat1_params=boat1_params)
    if args.ev_sweep:
        run_ev_sweep(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds, **common)
    elif args.wfcv:
        run_wfcv(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds, **common)
    elif args.from_date and args.to_date:
        run_period(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map, exacta_odds, **common)
    else:
        parser.error("--from/--to, --wfcv, or --ev-sweep required")

    print(f"\nTotal: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
