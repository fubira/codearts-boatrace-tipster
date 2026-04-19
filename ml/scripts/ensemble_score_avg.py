"""Score-averaging ensemble: average model.predict outputs, then run P2.

Unlike strict intersection, this creates a single aggregated prediction by
averaging z-normalized scores from each model. The averaged scores are
then fed into a standard single-model scale sweep.

The intent is to reduce per-model variance while preserving signal. Works
best when models are somewhat decorrelated (different HPs, different
features). For highly correlated models the effect is minimal.

Usage:
    uv run python -m scripts.ensemble_score_avg \\
        --models models/p2_v3_y1,models/bk_120_y1 \\
        --from 2025-04-01 --to 2026-04-20 \\
        --scales 1.0,0.85,0.75,0.65,0.60,0.55
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from scripts.scale_sweep import (
    EXCLUDED_STADIUMS,
    FIELD_SIZE,
    _trifecta_prob_scaled,
    load_odds_map,
)


def predict_zscored(model, feature_means: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict and z-score per race so models can be averaged on equal footing."""
    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = X[c].fillna(feature_means.get(c, 0.0))
    scores = model.predict(X)
    n_races = len(df) // FIELD_SIZE
    s2d = scores.reshape(n_races, FIELD_SIZE)
    mu = s2d.mean(axis=1, keepdims=True)
    sd = s2d.std(axis=1, keepdims=True) + 1e-10
    return (s2d - mu) / sd


def eval_scale_from_scores(
    df: pd.DataFrame,
    scores_2d: np.ndarray,
    odds_map: dict,
    scale: float,
    gap12_th: float = 0.04,
    conc_th: float = 0.60,
    gap23_th: float = 0.13,
    ev_th: float = -0.25,
    unit_divisor: int = 200,
    bankroll: int = 70000,
    bet_cap: int = 30000,
    excluded_stadiums: set | None = None,
):
    if excluded_stadiums is None:
        excluded_stadiums = EXCLUDED_STADIUMS
    n_races = scores_2d.shape[0]
    boats_2d = df["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    finish_2d = df["finish_position"].values.reshape(n_races, FIELD_SIZE)
    stadium_2d = df["stadium_id"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    unit = bankroll // unit_divisor
    n_bets = n_tickets = n_hits = 0
    total_stake = total_payout = 0

    for i in range(n_races):
        if int(stadium_2d[i, 0]) in excluded_stadiums:
            continue
        r1 = int(top_boats[i, 0])
        if r1 != 1:
            continue
        p1 = float(probs[i, pred_order[i, 0]])
        p2 = float(probs[i, pred_order[i, 1]])
        p3 = float(probs[i, pred_order[i, 2]])
        if (p1 - p2) < gap12_th:
            continue
        if (p2 + p3) / (1 - p1 + 1e-10) < conc_th:
            continue
        if (p2 - p3) < gap23_th:
            continue

        r2, r3 = int(top_boats[i, 1]), int(top_boats[i, 2])
        actual_order = np.argsort(finish_2d[i])
        a1 = int(boats_2d[i, actual_order[0]])
        a2 = int(boats_2d[i, actual_order[1]])
        a3 = int(boats_2d[i, actual_order[2]])
        hit_combo = f"{a1}-{a2}-{a3}"

        i1, i2, i3 = int(pred_order[i, 0]), int(pred_order[i, 1]), int(pred_order[i, 2])
        rid = int(rids[i])
        candidates = []
        for combo, perm in (
            (f"{r1}-{r2}-{r3}", (i1, i2, i3)),
            (f"{r1}-{r3}-{r2}", (i1, i3, i2)),
        ):
            odds = odds_map.get((rid, combo))
            if not odds or odds <= 0:
                continue
            mp = _trifecta_prob_scaled(probs[i], *perm, scale=scale)
            ev = mp * odds * 0.75 - 1
            if ev < ev_th:
                continue
            candidates.append((combo, odds))

        if not candidates:
            continue

        per_ticket = min(unit, bet_cap // max(len(candidates), 1))
        race_bought = False
        race_stake = 0
        race_payout = 0
        for combo, odds in candidates:
            n_tickets += 1
            race_stake += per_ticket
            race_bought = True
            if combo == hit_combo:
                n_hits += 1
                race_payout += int(per_ticket * odds)
        if race_bought:
            n_bets += 1
            total_stake += race_stake
            total_payout += race_payout

    pnl = total_payout - total_stake
    hit_pct = n_hits / n_tickets * 100 if n_tickets else 0
    roi = pnl / total_stake * 100 if total_stake else 0
    return {
        "scale": scale,
        "n_bets": n_bets,
        "n_tickets": n_tickets,
        "n_hits": n_hits,
        "hit_pct": hit_pct,
        "stake": total_stake,
        "payout": total_payout,
        "pnl": pnl,
        "roi_pct": roi,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="comma-separated model paths")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--scales", default="1.0,0.85,0.75,0.65,0.60,0.55,0.50")
    p.add_argument("--ev-threshold", type=float, default=-0.25)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = p.parse_args()

    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    scales = [float(s) for s in args.scales.split(",")]

    print(f"Period: {args.date_from} 〜 {args.date_to}")
    print(f"EV threshold: {args.ev_threshold}")
    print(f"Models (score averaging z-scored):")
    for m in model_paths:
        print(f"  {m}")
    print()

    print("Building features...")
    df_all = build_features_df(args.db_path)
    df = df_all[(df_all["race_date"] >= args.date_from) & (df_all["race_date"] <= args.date_to)].copy()
    df = df.sort_values(["race_id", "boat_number"]).reset_index(drop=True)
    print(f"  {len(df)} entries ({len(df) // FIELD_SIZE} races)")

    print("Loading odds...")
    odds_map = load_odds_map(args.db_path)
    print(f"  {len(odds_map)} odds")

    print("Predicting each model + z-score...")
    stacked = []
    for mp in model_paths:
        ranking = f"{mp}/ranking"
        model = load_model(ranking)
        meta = load_model_meta(ranking)
        fm = meta["feature_means"] if meta else {}
        z = predict_zscored(model, fm, df)
        stacked.append(z)
    avg_scores = np.mean(np.stack(stacked), axis=0)

    print(f"{'scale':>6}  {'bets':>6} {'tix':>6} {'hits':>5} {'hit%':>6} {'stake':>10} {'payout':>10} {'P/L':>10} {'ROI%':>7}")
    for s in scales:
        r = eval_scale_from_scores(df, avg_scores, odds_map, s, ev_th=args.ev_threshold)
        print(
            f"{s:>6.2f}  {r['n_bets']:>6} {r['n_tickets']:>6} {r['n_hits']:>5} "
            f"{r['hit_pct']:>5.2f}% {r['stake']:>10,} {r['payout']:>10,} {r['pnl']:>+10,} {r['roi_pct']:>+6.2f}%"
        )


if __name__ == "__main__":
    main()
