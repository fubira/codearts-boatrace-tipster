"""Ensemble evaluation via strict intersection of P2 buy sets.

Given N models, each with its own peak scale, a race is eligible for buy
if ALL models pass the structural filter (gap12/conc/gap23) AND a specific
ticket combo meets the ev threshold for ALL models.

This cuts noise by requiring independent agreement across models. The
intersection is strict: model A's top-3 must match model B's top-3 and
both must price the combo as EV+ at their respective scales.

Usage:
    uv run python -m scripts.ensemble_sweep \\
        --models "models/p2_v3_y1@0.70,models/bk_120_y1@0.60" \\
        --from 2025-04-01 --to 2026-04-20 \\
        --ev-threshold -0.25
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from scripts.scale_sweep import (
    EXCLUDED_STADIUMS,
    ScaleResult,
    _trifecta_prob_scaled,
    load_odds_map,
)

FIELD_SIZE = 6


@dataclass
class ModelEntry:
    name: str
    model: Any
    feature_means: dict[str, float]
    scale: float


def parse_models(spec: str) -> list[tuple[str, float]]:
    """Parse 'path1@scale1,path2@scale2,...' into [(path, scale), ...]."""
    entries: list[tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "@" not in part:
            raise ValueError(f"missing scale for model '{part}' (use path@scale)")
        path, scale_str = part.rsplit("@", 1)
        entries.append((path.strip(), float(scale_str)))
    return entries


def load_models(entries: list[tuple[str, float]]) -> list[ModelEntry]:
    loaded: list[ModelEntry] = []
    for path, scale in entries:
        ranking = f"{path}/ranking"
        m = load_model(ranking)
        meta = load_model_meta(ranking)
        fm = meta["feature_means"] if meta else {}
        loaded.append(ModelEntry(name=path, model=m, feature_means=fm, scale=scale))
    return loaded


def _model_per_race(
    df: pd.DataFrame,
    entry: ModelEntry,
    gap12_th: float,
    conc_th: float,
    gap23_th: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-race (passes_filter, top_boats_3, probs_6) for one model.

    df is pre-sorted by (race_id, boat_number). passes_filter is bool array
    of length n_races. top_boats_3 is (n_races, 3) int array. probs_6 is
    (n_races, 6) float array of softmax probabilities.
    """
    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = X[c].fillna(entry.feature_means.get(c, 0.0))
    scores = entry.model.predict(X)

    n_races = len(df) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = df["boat_number"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    # Sort probs into pred_order (so probs_sorted[i, 0] is top-1 prob etc.)
    probs_sorted = np.take_along_axis(probs, pred_order, axis=1)
    p1 = probs_sorted[:, 0]
    p2 = probs_sorted[:, 1]
    p3 = probs_sorted[:, 2]
    gap12 = p1 - p2
    conc = (p2 + p3) / (1 - p1 + 1e-10)
    gap23 = p2 - p3

    passes = (top_boats[:, 0] == 1) & (gap12 >= gap12_th) & (conc >= conc_th) & (gap23 >= gap23_th)

    return passes, top_boats[:, :3], probs


def eval_ensemble(
    df: pd.DataFrame,
    entries: list[ModelEntry],
    odds_map: dict[tuple[int, str], float],
    gap12_th: float = 0.04,
    conc_th: float = 0.60,
    gap23_th: float = 0.13,
    ev_th: float = -0.25,
    unit_divisor: int = 200,
    bankroll: int = 70000,
    bet_cap: int = 30000,
    excluded_stadiums: set | None = None,
    by_stadium: bool = False,
) -> tuple[ScaleResult, list[dict] | None]:
    """Run strict-intersection ensemble: buy iff ALL models price it EV+.

    A combo qualifies when:
      1. All models independently pass structural filters for that race.
      2. All models' top-3 sets (as a set) are identical for that race.
      3. Model A's top-1 == model B's top-1 == 1 (enforced in pass filter).
      4. For each candidate combo c, every model's EV(c) >= ev_th.
    """
    if excluded_stadiums is None:
        excluded_stadiums = EXCLUDED_STADIUMS

    df = df.sort_values(["race_id", "boat_number"]).reset_index(drop=True)
    n_races = len(df) // FIELD_SIZE
    rids = df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    stadium_2d = df["stadium_id"].values.reshape(n_races, FIELD_SIZE)
    finish_2d = df["finish_position"].values.reshape(n_races, FIELD_SIZE)
    boats_2d = df["boat_number"].values.reshape(n_races, FIELD_SIZE)

    per_model: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for e in entries:
        per_model.append(_model_per_race(df, e, gap12_th, conc_th, gap23_th))

    unit = bankroll // unit_divisor
    n_bets = 0
    n_tickets = 0
    n_hits = 0
    total_stake = 0
    total_payout = 0
    per_std: dict[int, dict] = {} if by_stadium else None

    for i in range(n_races):
        stadium = int(stadium_2d[i, 0])
        if stadium in excluded_stadiums:
            continue
        if not all(pm[0][i] for pm in per_model):
            continue

        # Require all models to agree on top-3 SET (unordered).
        # Ordering differences are allowed — combos are enumerated below.
        tops_3 = [tuple(sorted(pm[1][i])) for pm in per_model]
        if len(set(tops_3)) > 1:
            continue

        actual_order = np.argsort(finish_2d[i])
        a1 = int(boats_2d[i, actual_order[0]])
        a2 = int(boats_2d[i, actual_order[1]])
        a3 = int(boats_2d[i, actual_order[2]])
        hit_combo = f"{a1}-{a2}-{a3}"

        # Enumerate the 2 P2 candidate combos using model 0's ordering
        # (top-1/2/3). Since tops_3 sets match, model j has the same 3
        # boats but possibly different ordering — we re-locate them.
        r1, r2, r3 = int(per_model[0][1][i, 0]), int(per_model[0][1][i, 1]), int(per_model[0][1][i, 2])
        rid = int(rids[i])

        candidates = []
        for combo_boats in [(r1, r2, r3), (r1, r3, r2)]:
            combo_str = f"{combo_boats[0]}-{combo_boats[1]}-{combo_boats[2]}"
            odds = odds_map.get((rid, combo_str))
            if not odds or odds <= 0:
                continue
            all_pass = True
            for j, entry in enumerate(entries):
                top3 = per_model[j][1][i]
                probs = per_model[j][2][i]
                # Map combo_boats to indices in this model's boats array
                ij_top = [int(np.where(boats_2d[i] == b)[0][0]) for b in combo_boats]
                mp = _trifecta_prob_scaled(probs, *ij_top, scale=entry.scale)
                ev_j = mp * odds * 0.75 - 1
                if ev_j < ev_th:
                    all_pass = False
                    break
            if all_pass:
                candidates.append((combo_str, odds))

        if not candidates:
            continue

        per_ticket = min(unit, bet_cap // max(len(candidates), 1))
        race_bought = False
        race_hits_payout = 0
        race_stake = 0
        for combo, odds in candidates:
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
            tix_hits = sum(1 for c, _ in candidates if c == hit_combo)
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
        model="+".join(e.name for e in entries),
        scale=0.0,  # ensemble uses per-model scales
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
    if per_std is not None:
        breakdown = [
            {"stadium_id": k, **v, "pnl": v["payout"] - v["stake"]}
            for k, v in sorted(per_std.items())
        ]
    return result, breakdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True,
                   help="comma-separated 'path@scale' entries (e.g. models/a@0.7,models/b@0.6)")
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--ev-threshold", type=float, default=-0.25)
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--by-stadium", action="store_true")
    p.add_argument("--excluded-stadiums", default=None)
    args = p.parse_args()

    entries_spec = parse_models(args.models)

    excluded = (
        {int(x) for x in args.excluded_stadiums.split(",") if x.strip()}
        if args.excluded_stadiums is not None
        else EXCLUDED_STADIUMS
    )

    print(f"Period: {args.date_from} 〜 {args.date_to}")
    print(f"EV threshold: {args.ev_threshold}")
    print(f"Excluded stadiums: {sorted(excluded)}")
    print("Models:")
    for path, scale in entries_spec:
        print(f"  {path} @ scale={scale}")
    print()

    print("Loading models...")
    entries = load_models(entries_spec)

    print("Building features...")
    df_all = build_features_df(args.db_path)
    df = df_all[(df_all["race_date"] >= args.date_from) & (df_all["race_date"] <= args.date_to)].copy()
    print(f"  filtered: {len(df)} entries ({len(df) // FIELD_SIZE} races)")

    print("Loading odds map...")
    odds_map = load_odds_map(args.db_path)
    print(f"  {len(odds_map)} 3連単 odds rows")
    print()

    r, breakdown = eval_ensemble(
        df, entries, odds_map,
        ev_th=args.ev_threshold,
        by_stadium=args.by_stadium,
        excluded_stadiums=excluded,
    )
    print(f"=== ensemble (intersection) ===")
    print(f"{'bets':>6} {'tix':>6} {'hits':>5} {'hit%':>6} {'stake':>10} {'payout':>10} {'P/L':>10} {'ROI%':>7}")
    print(
        f"{r.n_bets:>6} {r.n_tickets:>6} {r.n_hits:>5} "
        f"{r.hit_pct:>5.2f}% {r.stake:>10,} {r.payout:>10,} {r.pnl:>+10,} {r.roi_pct:>+6.2f}%"
    )
    if breakdown:
        print()
        print(f"-- by stadium sorted by ROI --")
        print(f"{'std':>3} {'bets':>5} {'tix':>4} {'hits':>5} {'hit%':>6} {'stake':>8} {'payout':>8} {'P/L':>8} {'ROI%':>7}")
        for d in sorted(breakdown, key=lambda x: x["pnl"] / max(x["stake"], 1)):
            hp = (d["hits"] / d["tix"] * 100) if d["tix"] else 0.0
            roi = (d["pnl"] / d["stake"] * 100) if d["stake"] else 0.0
            print(
                f"{d['stadium_id']:>3} {d['bets']:>5} {d['tix']:>4} {d['hits']:>5} "
                f"{hp:>5.2f}% {d['stake']:>8,} {d['payout']:>8,} {d['pnl']:>+8,} {roi:>+6.2f}%"
            )


if __name__ == "__main__":
    main()
