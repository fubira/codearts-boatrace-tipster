"""Shared P2 decision helpers for ops diagnostic scripts.

analyze_decisions.py and compare_miss_patterns.py both load a P2 model,
run predictions, softmax to probs, and build 2-ticket P2 combinations
(1-r2-r3, 1-r3-r2) with per-ticket EV against 3連単 odds. This module
consolidates that "model → per-race decision record" pipeline. Each
caller layers its own filter chain and output formatting on top.

Callers apply their own ticket filtering: analyze_decisions keeps every
ticket (unconditional EV display), while compare_miss_patterns drops
tickets with EV < ev_threshold. So `any_hit` is intentionally NOT stored
on RaceDecision — callers derive it from whatever ticket subset matches
their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import get_connection
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.model import load_model, load_model_meta
from scripts.tune_p2 import _trifecta_prob

FIELD_SIZE = 6


@dataclass
class Ticket:
    combo: str
    odds: float
    ev: float
    hit: bool


@dataclass
class RaceDecision:
    rid: int
    r1: int
    r2: int
    r3: int
    p1: float
    p2: float
    p3: float
    gap12: float
    conc: float
    gap23: float
    max_ev: float
    tickets: list[Ticket]
    hit_combo: str
    a1: int
    a2: int
    a3: int
    # Boat 1 actual finish (1-6). 0 is a sentinel for NaN finish_position
    # (フライング / 欠場 / 失格). int(np.nan) raises, so callers must read
    # this field instead of casting finish_position directly.
    boat1_finish: int


def load_model_and_strategy(model_dir: str) -> tuple[Any, dict, dict[str, float]]:
    """Load a saved P2 ranking model and its strategy + feature_means.

    `model_dir` is the parent directory (e.g. `models/p2_v2`); the actual
    artifacts live in `<model_dir>/ranking/` per the project layout.
    Delegates to `boatrace_tipster_ml.model` so path conventions stay
    centralized.
    """
    ranking_dir = f"{model_dir}/ranking"
    model = load_model(ranking_dir)
    meta = load_model_meta(ranking_dir)
    if meta is None:
        raise FileNotFoundError(f"model_meta.json not found in {ranking_dir}")
    return model, meta["strategy"], meta["feature_means"]


def latest_race_date(db_path: str) -> str:
    """Return the latest race_date in the DB as a 'YYYY-MM-DD' string."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT MAX(race_date) FROM db.races").fetchone()
    finally:
        conn.close()
    return str(row[0])


def compute_race_decisions(
    df: pd.DataFrame,
    model: Any,
    feature_means: dict[str, float],
    odds_map: dict[tuple[int, str], float],
) -> list[RaceDecision]:
    """Run the model over df and build one RaceDecision per race.

    df must contain exactly 6 rows per race (all boats). Rows are
    sorted by (race_id, boat_number) internally before reshaping, so
    callers don't need to pre-sort.

    Tickets are returned unconditionally — no EV threshold is applied
    here. `max_ev` defaults to -999.0 if no tickets exist (missing odds),
    matching the legacy analyze_decisions sentinel used for ev_low
    classification.
    """
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

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    decisions: list[RaceDecision] = []
    for i in range(n_races):
        rid = int(rids[i])
        r1 = int(top_boats[i, 0])
        r2 = int(top_boats[i, 1])
        r3 = int(top_boats[i, 2])
        p1 = float(model_probs[i, pred_order[i, 0]])
        p2 = float(model_probs[i, pred_order[i, 1]])
        p3 = float(model_probs[i, pred_order[i, 2]])
        gap12 = p1 - p2
        conc = (p2 + p3) / (1 - p1 + 1e-10)
        gap23 = p2 - p3

        actual_order = np.argsort(finish_2d[i])
        a1 = int(boats_2d[i, actual_order[0]])
        a2 = int(boats_2d[i, actual_order[1]])
        a3 = int(boats_2d[i, actual_order[2]])
        hit_combo = f"{a1}-{a2}-{a3}"

        boat1_idx = int(np.where(boats_2d[i] == 1)[0][0])
        boat1_raw = finish_2d[i, boat1_idx]
        boat1_finish = 0 if pd.isna(boat1_raw) else int(boat1_raw)

        i1 = int(pred_order[i, 0])
        i2 = int(pred_order[i, 1])
        i3 = int(pred_order[i, 2])
        tickets: list[Ticket] = []
        max_ev = -999.0
        for combo, perm in (
            (f"{r1}-{r2}-{r3}", (i1, i2, i3)),
            (f"{r1}-{r3}-{r2}", (i1, i3, i2)),
        ):
            odds = odds_map.get((rid, combo))
            if not odds or odds <= 0:
                continue
            mp = _trifecta_prob(model_probs[i], *perm)
            ev = float(mp / (1 / odds) * 0.75 - 1)
            tickets.append(Ticket(combo=combo, odds=float(odds), ev=ev, hit=combo == hit_combo))
            if ev > max_ev:
                max_ev = ev

        decisions.append(RaceDecision(
            rid=rid, r1=r1, r2=r2, r3=r3,
            p1=p1, p2=p2, p3=p3,
            gap12=gap12, conc=conc, gap23=gap23,
            max_ev=max_ev, tickets=tickets,
            hit_combo=hit_combo, a1=a1, a2=a2, a3=a3,
            boat1_finish=boat1_finish,
        ))
    return decisions
