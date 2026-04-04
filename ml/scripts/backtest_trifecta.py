"""Backtest trifecta X-noB1-noB1 strategy.

Evaluates 3連単 strategy: fix 1st place (anti-favorite pick), exclude boat 1
from 2nd/3rd, buy all remaining combinations (12pt).

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
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import train_model, walk_forward_splits

FIELD_SIZE = 6


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

    return df, trifecta_odds, dict(tri_win_prob), finish_map, race_date_map


def train_models(train_df, val_df=None):
    """Train boat1 binary + LambdaRank models."""
    with contextlib.redirect_stdout(io.StringIO()):
        X_b1_tr, y_b1_tr, _ = reshape_to_boat1(train_df if val_df is not None else train_df)
        if val_df is not None:
            X_b1_v, y_b1_v, _ = reshape_to_boat1(val_df)
        else:
            X_b1_v, y_b1_v = None, None
        b1_model, _ = train_boat1_model(X_b1_tr, y_b1_tr, X_b1_v, y_b1_v)

        X_r_tr, y_r_tr, m_r_tr = prepare_feature_matrix(train_df if val_df is not None else train_df)
        if val_df is not None:
            X_r_v, y_r_v, m_r_v = prepare_feature_matrix(val_df)
        else:
            X_r_v, y_r_v, m_r_v = None, None, None
        rank_model, _ = train_model(
            X_r_tr, y_r_tr, m_r_tr,
            X_r_v, y_r_v, m_r_v,
            early_stopping_rounds=50,
        )

    return b1_model, rank_model


def evaluate_period(
    b1_model,
    rank_model,
    test_df: pd.DataFrame,
    trifecta_odds: dict,
    tri_win_prob: dict,
    finish_map: dict,
    race_date_map: dict,
    b1_threshold: float = 0.40,
    ev_threshold: float = 0.10,
) -> list[dict]:
    """Evaluate strategy on test period, return per-race results."""
    with contextlib.redirect_stdout(io.StringIO()):
        X_b1_te, _, meta_b1_te = reshape_to_boat1(test_df)
        X_r_te, _, meta_r_te = prepare_feature_matrix(test_df)

    b1_probs = b1_model.predict_proba(X_b1_te)[:, 1]
    rank_scores = rank_model.predict(X_r_te)

    n_races = len(X_r_te) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_r_te["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta_r_te["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    b1_map = {rid: i for i, rid in enumerate(meta_b1_te["race_id"].values)}

    results = []
    for ri in range(n_races):
        rid = int(race_ids[ri])
        bi = b1_map.get(rid)
        if bi is None:
            continue
        b1p = float(b1_probs[bi])
        if b1p >= b1_threshold:
            continue

        wp = int(top_boats[ri, 0])
        if wp == 1:
            wp = int(top_boats[ri, 1])

        bidx = np.where(boats_2d[ri] == wp)[0]
        if len(bidx) == 0:
            continue
        wprob = float(rank_probs[ri, bidx[0]])

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

        # Check result
        a2 = a3 = None
        for b in range(1, 7):
            fp = finish_map.get((rid, b))
            if fp == 2:
                a2 = b
            if fp == 3:
                a3 = b

        hit_odds = 0.0
        if finish_map.get((rid, wp)) == 1 and a2 and a3:
            hc = f"{wp}-{a2}-{a3}"
            if hc in tkts:
                ho = trifecta_odds.get((rid, hc))
                if ho:
                    hit_odds = ho

        results.append({
            "race_id": rid,
            "date": race_date_map.get(rid, ""),
            "winner_pick": wp,
            "b1_prob": round(b1p, 3),
            "winner_prob": round(wprob, 3),
            "ev": round(ev, 3),
            "tickets": len(tkts),
            "hit_odds": round(hit_odds, 1),
            "won": hit_odds > 0,
        })

    return results


def print_daily(results: list[dict], label: str = ""):
    """Print daily breakdown from per-race results."""
    if label:
        print(f"\n=== {label} ===")

    daily: dict[str, dict] = defaultdict(
        lambda: {"races": 0, "tickets": 0, "wins": 0, "payout": 0.0}
    )
    for r in results:
        d = daily[r["date"]]
        d["races"] += 1
        d["tickets"] += r["tickets"]
        if r["won"]:
            d["wins"] += 1
            d["payout"] += r["hit_odds"]

    cum = 0
    total_r = total_t = total_w = 0
    total_p = 0.0
    win_days = 0

    for date in sorted(daily.keys()):
        d = daily[date]
        pl = d["payout"] - d["tickets"]
        cum += pl
        total_r += d["races"]
        total_t += d["tickets"]
        total_w += d["wins"]
        total_p += d["payout"]
        if pl > 0:
            win_days += 1
        marker = "+" if pl > 0 else "-"
        roi = d["payout"] / d["tickets"] if d["tickets"] > 0 else 0
        print(
            f"  {date}: {d['races']:>2}R {d['tickets']:>3}tkt "
            f"{d['wins']}W P/L {pl:>+7.0f} cum {cum:>+8.0f} {marker}"
        )

    days = len(daily)
    if total_t > 0:
        roi = total_p / total_t
        print(f"\n  {days} days, {total_r}R, {total_t}tkt, {total_w}W")
        print(f"  ROI: {roi:.0%}, P/L: {total_p - total_t:+.0f}")
        print(f"  R/d: {total_r/days:.1f}, Win days: {win_days}/{days} ({win_days/days:.0%})")

    return {
        "days": days,
        "races": total_r,
        "tickets": total_t,
        "wins": total_w,
        "payout": total_p,
        "roi": total_p / total_t if total_t > 0 else 0,
    }


def run_period(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map):
    """Period backtest with daily breakdown."""
    train_df = df[df["race_date"] < args.from_date]
    test_df = df[
        (df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)
    ]

    dates = sorted(train_df["race_date"].unique())
    val_start = dates[max(0, len(dates) - 60)]
    train_early = train_df[train_df["race_date"] < val_start]
    train_late = train_df[train_df["race_date"] >= val_start]

    print(f"Training on data before {args.from_date}...", file=sys.stderr)
    b1_model, rank_model = train_models(train_early, train_late)

    results = evaluate_period(
        b1_model, rank_model, test_df,
        trifecta_odds, tri_win_prob, finish_map, race_date_map,
        b1_threshold=args.b1_threshold,
        ev_threshold=args.ev_threshold,
    )

    summary = print_daily(
        results,
        f"X-noB1-noB1, b1<{args.b1_threshold:.0%}, EV>={args.ev_threshold:.0%} "
        f"({args.from_date} ~ {args.to_date})",
    )

    if args.json:
        json.dump(
            {"params": vars(args), "summary": summary, "races": results},
            sys.stdout,
            ensure_ascii=False,
            default=str,
        )


def run_wfcv(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map):
    """Walk-forward CV."""
    X_rank, y_rank, meta_rank = prepare_feature_matrix(df)
    folds = walk_forward_splits(
        X_rank, y_rank, meta_rank,
        n_folds=args.n_folds, fold_months=args.fold_months,
    )

    print(f"WF-CV: {len(folds)} folds, b1<{args.b1_threshold:.0%}, EV>={args.ev_threshold:.0%}")
    fold_rois = []

    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]

        train_fold = df[df["race_date"] < test_from]
        test_fold = df[
            (df["race_date"] >= test_from) & (df["race_date"] < test_to)
        ]

        dates = sorted(train_fold["race_date"].unique())
        val_start = dates[max(0, len(dates) - 60)]

        print(f"\nFold {i+1}: {test_from} ~ {test_to}", file=sys.stderr)
        b1_model, rank_model = train_models(
            train_fold[train_fold["race_date"] < val_start],
            train_fold[train_fold["race_date"] >= val_start],
        )

        results = evaluate_period(
            b1_model, rank_model, test_fold,
            trifecta_odds, tri_win_prob, finish_map, race_date_map,
            b1_threshold=args.b1_threshold,
            ev_threshold=args.ev_threshold,
        )

        total_t = sum(r["tickets"] for r in results)
        total_p = sum(r["hit_odds"] for r in results if r["won"])
        roi = total_p / total_t if total_t > 0 else 0
        wins = sum(1 for r in results if r["won"])
        fold_rois.append(roi)

        print(f"  Fold {i+1}: {len(results)}R, {total_t}tkt, {wins}W, ROI {roi:.0%}")

    print(f"\n{'='*50}")
    print(f"Mean ROI: {np.mean(fold_rois):.0%} ± {np.std(fold_rois):.0%}")
    print(f"Min: {np.min(fold_rois):.0%}, Max: {np.max(fold_rois):.0%}")
    sharpe = (np.mean(fold_rois) - 1) / np.std(fold_rois) if np.std(fold_rois) > 0 else 0
    print(f"Sharpe: {sharpe:.2f}")


def run_ev_sweep(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map):
    """EV threshold sweep across WF-CV."""
    X_rank, y_rank, meta_rank = prepare_feature_matrix(df)
    folds = walk_forward_splits(
        X_rank, y_rank, meta_rank,
        n_folds=args.n_folds, fold_months=args.fold_months,
    )

    print(f"EV sweep: {len(folds)} folds")

    # Pre-train models
    fold_models = []
    fold_test_dfs = []
    for i, fold in enumerate(folds):
        test_dates = fold["period"]["test"]
        test_from, test_to = [d.strip() for d in test_dates.split("~")]

        train_fold = df[df["race_date"] < test_from]
        test_fold = df[
            (df["race_date"] >= test_from) & (df["race_date"] < test_to)
        ]

        dates = sorted(train_fold["race_date"].unique())
        val_start = dates[max(0, len(dates) - 60)]

        print(f"  Training fold {i+1}: {test_from} ~ {test_to}", file=sys.stderr)
        b1_model, rank_model = train_models(
            train_fold[train_fold["race_date"] < val_start],
            train_fold[train_fold["race_date"] >= val_start],
        )
        fold_models.append((b1_model, rank_model))
        fold_test_dfs.append(test_fold)

    ev_thresholds = [-0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{'EV':>6} | {'F1':>5} {'F2':>5} {'F3':>5} {'F4':>5} | {'Mean':>5} {'Std':>4} {'Min':>5} {'Shrp':>5} {'R/d':>5}")
    print("-" * 75)

    for ev_thr in ev_thresholds:
        fold_rois = []
        fold_rpd = []

        for i, (b1_model, rank_model) in enumerate(fold_models):
            test_dates = folds[i]["period"]["test"]
            test_from, test_to = [d.strip() for d in test_dates.split("~")]
            test_days = (pd.Timestamp(test_to) - pd.Timestamp(test_from)).days

            results = evaluate_period(
                b1_model, rank_model, fold_test_dfs[i],
                trifecta_odds, tri_win_prob, finish_map, race_date_map,
                b1_threshold=args.b1_threshold,
                ev_threshold=ev_thr,
            )

            total_t = sum(r["tickets"] for r in results)
            total_p = sum(r["hit_odds"] for r in results if r["won"])
            roi = total_p / total_t if total_t > 0 else 0
            rpd = len(results) / test_days if test_days > 0 else 0

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
    parser = argparse.ArgumentParser(description="Backtest trifecta X-noB1-noB1")
    parser.add_argument("--from", dest="from_date")
    parser.add_argument("--to", dest="to_date")
    parser.add_argument("--wfcv", action="store_true")
    parser.add_argument("--ev-sweep", action="store_true")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-months", type=int, default=2)
    parser.add_argument("--b1-threshold", type=float, default=0.40)
    parser.add_argument("--ev-threshold", type=float, default=0.10)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading data...", file=sys.stderr)
    df, trifecta_odds, tri_win_prob, finish_map, race_date_map = load_data(
        args.db_path
    )
    print(f"Loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    if args.ev_sweep:
        run_ev_sweep(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map)
    elif args.wfcv:
        run_wfcv(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map)
    elif args.from_date and args.to_date:
        run_period(args, df, trifecta_odds, tri_win_prob, finish_map, race_date_map)
    else:
        parser.error("--from/--to, --wfcv, or --ev-sweep required")

    print(f"\nTotal: {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
