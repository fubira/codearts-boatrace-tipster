"""Scale sweep for P2 strategy probability calibration.

Applies a multiplicative `prob_scale` factor to per-boat softmax probs
before EV computation. Filter thresholds (gap12 / top3_conc / gap23)
remain computed on raw probs, so only EV is affected.

Usage:
    uv run python -m scripts.scale_sweep \\
        --model-dirs models/p2_v3_y1,models/bk_120_y1,models/bk_151_y1 \\
        --from 2025-04-01 --to 2026-04-20 \\
        --scales 1.0,0.85,0.8,0.75,0.7,0.65,0.6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta

FIELD_SIZE = 6
EXCLUDED_STADIUMS = {3, 9, 21}  # 江戸川, 津, 芦屋 (from p2_v3 strategy)


@dataclass
class ScaleResult:
    model: str
    scale: float
    n_bets: int
    n_tickets: int
    n_hits: int
    stake: int
    payout: int
    pnl: int
    hit_pct: float
    roi_pct: float


def _trifecta_prob_scaled(probs_6: np.ndarray, i1: int, i2: int, i3: int, scale: float) -> float:
    """Scaled trifecta prob = scale × raw trifecta prob.

    Previous session empirically found the P2 filter tail is overconfident
    by 3〜4pt on the final trifecta prob. The correction is multiplicative
    on the JOINT prob, not per-boat (per-boat scale^3 overshoots heavily).
    """
    p1 = probs_6[i1]
    p2 = probs_6[i2]
    p3 = probs_6[i3]
    if p1 >= 1.0 or (p1 + p2) >= 1.0:
        return 0.0
    return scale * p1 * (p2 / (1 - p1)) * (p3 / (1 - p1 - p2))


def load_odds_map(db_path: str) -> dict[tuple[int, str], float]:
    """Load all confirmed 3連単 odds keyed by (race_id, combo)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT race_id, combination, odds FROM db.race_odds "
            "WHERE bet_type='3連単' AND odds IS NOT NULL AND odds > 0"
        ).fetchall()
    finally:
        conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def eval_scale(
    df: pd.DataFrame,
    model,
    feature_means: dict[str, float],
    odds_map: dict[tuple[int, str], float],
    scale: float,
    gap12_th: float = 0.04,
    conc_th: float = 0.60,
    gap23_th: float = 0.13,
    ev_th: float = -0.25,
    unit_divisor: int = 200,
    bankroll: int = 70000,
    bet_cap: int = 30000,
    daily: bool = False,
    by_stadium: bool = False,
    excluded_stadiums: set | None = None,
    kelly_fraction: float = 0.0,
    min_stake: int = 100,
) -> tuple[ScaleResult, list[dict] | None]:
    """Run P2 decision + accounting for one (model, scale).

    When daily=True, returns per-date aggregation as the second return value.
    When by_stadium=True, returns per-stadium aggregation instead.
    """
    if excluded_stadiums is None:
        excluded_stadiums = EXCLUDED_STADIUMS
    df = df.sort_values(["race_id", "boat_number"]).reset_index(drop=True)

    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = X[c].fillna(feature_means.get(c, 0.0))
    scores = model.predict(X)

    n_races = len(df) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = df["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    finish_2d = df["finish_position"].values.reshape(n_races, FIELD_SIZE)
    stadium_2d = df["stadium_id"].values.reshape(n_races, FIELD_SIZE)
    dates = df["race_date"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    unit = bankroll // unit_divisor
    n_bets = 0
    n_tickets = 0
    n_hits = 0
    total_stake = 0
    total_payout = 0
    per_day: dict[str, dict] = {} if daily else None
    per_std: dict[int, dict] = {} if by_stadium else None

    for i in range(n_races):
        stadium = int(stadium_2d[i, 0])
        if stadium in excluded_stadiums:
            continue

        r1, r2, r3 = int(top_boats[i, 0]), int(top_boats[i, 1]), int(top_boats[i, 2])
        if r1 != 1:
            continue

        p1 = float(probs[i, pred_order[i, 0]])
        p2 = float(probs[i, pred_order[i, 1]])
        p3 = float(probs[i, pred_order[i, 2]])
        gap12 = p1 - p2
        conc = (p2 + p3) / (1 - p1 + 1e-10)
        gap23 = p2 - p3

        if gap12 < gap12_th or conc < conc_th or gap23 < gap23_th:
            continue

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
            candidates.append((combo, odds, mp))

        if not candidates:
            continue

        race_bought = False
        race_hits_payout = 0
        race_stake = 0
        for combo, odds, mp in candidates:
            if kelly_fraction > 0:
                b = odds - 1
                q = 1 - mp
                f_star = (b * mp - q) / b if b > 0 else 0.0
                kelly_stake = int(bankroll * max(f_star, 0.0) * kelly_fraction)
                per_ticket = min(kelly_stake, bet_cap)
                if per_ticket < min_stake:
                    continue
            else:
                per_ticket = min(unit, bet_cap // max(len(candidates), 1))
            n_tickets += 1
            race_stake += per_ticket
            race_bought = True
            if combo == hit_combo:
                n_hits += 1
                race_hits_payout += int(per_ticket * odds)

        if race_bought:
            n_bets += 1
            total_stake += race_stake
            total_payout += race_hits_payout
            tix_hits = sum(1 for c, _o, _m in candidates if c == hit_combo)
            if per_day is not None:
                date_key = str(dates[i])
                d = per_day.setdefault(date_key, {
                    "bets": 0, "tix": 0, "hits": 0, "stake": 0, "payout": 0,
                })
                d["bets"] += 1
                d["tix"] += len(candidates)
                d["hits"] += tix_hits
                d["stake"] += race_stake
                d["payout"] += race_hits_payout
            if per_std is not None:
                s = per_std.setdefault(stadium, {
                    "bets": 0, "tix": 0, "hits": 0, "stake": 0, "payout": 0,
                })
                s["bets"] += 1
                s["tix"] += len(candidates)
                s["hits"] += tix_hits
                s["stake"] += race_stake
                s["payout"] += race_hits_payout

    pnl = total_payout - total_stake
    hit_pct = (n_hits / n_tickets * 100) if n_tickets else 0.0
    roi_pct = (pnl / total_stake * 100) if total_stake else 0.0
    result = ScaleResult(
        model="",
        scale=scale,
        n_bets=n_bets,
        n_tickets=n_tickets,
        n_hits=n_hits,
        stake=total_stake,
        payout=total_payout,
        pnl=pnl,
        hit_pct=hit_pct,
        roi_pct=roi_pct,
    )
    breakdown: list[dict] | None = None
    if per_day is not None:
        breakdown = [
            {"date": k, **v, "pnl": v["payout"] - v["stake"]}
            for k, v in sorted(per_day.items())
        ]
    elif per_std is not None:
        breakdown = [
            {"stadium_id": k, **v, "pnl": v["payout"] - v["stake"]}
            for k, v in sorted(per_std.items())
        ]
    return result, breakdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dirs", required=True, help="comma-separated model parent dirs")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--scales", default="1.0,0.85,0.8,0.75,0.7,0.65,0.6")
    p.add_argument("--ev-threshold", type=float, default=-0.25)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--daily", action="store_true",
                   help="show per-date breakdown (one row per race date)")
    p.add_argument("--by-stadium", action="store_true",
                   help="show per-stadium breakdown instead of daily")
    p.add_argument("--excluded-stadiums", default=None,
                   help="comma-separated stadium IDs to exclude "
                        f"(default: {sorted(EXCLUDED_STADIUMS)})")
    p.add_argument("--kelly-fraction", type=float, default=0.0,
                   help="Kelly fraction (0 = fixed unit, 0.25 = 1/4 Kelly typical)")
    p.add_argument("--min-stake", type=int, default=100,
                   help="Skip ticket if Kelly stake below this (円)")
    args = p.parse_args()

    excluded = (
        {int(x) for x in args.excluded_stadiums.split(",") if x.strip()}
        if args.excluded_stadiums is not None
        else EXCLUDED_STADIUMS
    )

    model_dirs = [m.strip() for m in args.model_dirs.split(",") if m.strip()]
    scales = [float(s) for s in args.scales.split(",")]

    print(f"Period: {args.date_from} 〜 {args.date_to}")
    print(f"EV threshold: {args.ev_threshold}")
    print(f"Excluded stadiums: {sorted(excluded)}")
    print(f"Models: {model_dirs}")
    print(f"Scales: {scales}")
    print()

    print("Building features...")
    df_all = build_features_df(args.db_path)
    df = df_all[(df_all["race_date"] >= args.date_from) & (df_all["race_date"] <= args.date_to)].copy()
    print(f"  filtered: {len(df)} entries ({len(df) // FIELD_SIZE} races)")

    print("Loading odds map...")
    odds_map = load_odds_map(args.db_path)
    print(f"  {len(odds_map)} 3連単 odds rows (all races)")
    print()

    results: list[ScaleResult] = []
    for mdir in model_dirs:
        ranking = f"{mdir}/ranking"
        model = load_model(ranking)
        meta = load_model_meta(ranking)
        fm = meta["feature_means"] if meta else {}
        print(f"=== {mdir} ===")
        print(f"{'scale':>6}  {'bets':>6} {'tix':>6} {'hits':>5} {'hit%':>6} {'stake':>10} {'payout':>10} {'P/L':>10} {'ROI%':>7}")
        for s in scales:
            r, breakdown = eval_scale(
                df, model, fm, odds_map, s,
                ev_th=args.ev_threshold,
                daily=args.daily,
                by_stadium=args.by_stadium,
                excluded_stadiums=excluded,
                kelly_fraction=args.kelly_fraction,
                min_stake=args.min_stake,
            )
            r.model = mdir
            results.append(r)
            print(
                f"{s:>6.2f}  {r.n_bets:>6} {r.n_tickets:>6} {r.n_hits:>5} "
                f"{r.hit_pct:>5.2f}% {r.stake:>10,} {r.payout:>10,} {r.pnl:>+10,} {r.roi_pct:>+6.2f}%"
            )
            if breakdown and args.daily:
                print(f"  -- daily (scale={s:.2f}) --")
                print(f"  {'date':<12} {'bets':>5} {'tix':>4} {'hits':>5} {'hit%':>6} {'stake':>8} {'payout':>8} {'P/L':>8}")
                for d in breakdown:
                    hp = (d["hits"] / d["tix"] * 100) if d["tix"] else 0.0
                    print(
                        f"  {d['date']:<12} {d['bets']:>5} {d['tix']:>4} {d['hits']:>5} "
                        f"{hp:>5.2f}% {d['stake']:>8,} {d['payout']:>8,} {d['pnl']:>+8,}"
                    )
            elif breakdown and args.by_stadium:
                print(f"  -- by stadium (scale={s:.2f}) sorted by ROI --")
                print(f"  {'std':>3} {'bets':>5} {'tix':>4} {'hits':>5} {'hit%':>6} {'stake':>8} {'payout':>8} {'P/L':>8} {'ROI%':>7}")
                for d in sorted(breakdown, key=lambda x: x["pnl"] / max(x["stake"], 1)):
                    hp = (d["hits"] / d["tix"] * 100) if d["tix"] else 0.0
                    roi = (d["pnl"] / d["stake"] * 100) if d["stake"] else 0.0
                    print(
                        f"  {d['stadium_id']:>3} {d['bets']:>5} {d['tix']:>4} {d['hits']:>5} "
                        f"{hp:>5.2f}% {d['stake']:>8,} {d['payout']:>8,} {d['pnl']:>+8,} {roi:>+6.2f}%"
                    )
        print()

    print("=== summary (sorted by P/L desc) ===")
    for r in sorted(results, key=lambda x: x.pnl, reverse=True):
        print(
            f"{r.model:<30} scale={r.scale:.2f}  P/L {r.pnl:>+10,}  "
            f"hit {r.hit_pct:>5.2f}%  ROI {r.roi_pct:>+6.2f}%  tix {r.n_tickets}"
        )


if __name__ == "__main__":
    main()
