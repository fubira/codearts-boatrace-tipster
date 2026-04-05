"""Predict trifecta X-noB1-noB1 strategy (12-ticket fixed 1st place).

Flow:
  1. Boat1 binary model: predict b1_prob for each race
  2. Filter: b1_prob < b1_threshold (boat 1 likely to lose)
  3. LambdaRank: pick winner X (top non-boat-1)
  4. Compute trifecta-implied EV = softmax_prob / market_prob * 0.75 - 1
  5. Filter: EV >= ev_threshold
  6. Generate 12 tickets: X-{non-1, non-X}-{non-1, non-X}

Usage:
    uv run --directory ml python -m scripts.predict_trifecta --date 2026-04-04
    uv run --directory ml python -m scripts.predict_trifecta --date 2026-04-04 --ev-threshold 0.33
"""

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import load_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from boatrace_tipster_ml.snapshot_features import build_features_from_snapshot

FIELD_SIZE = 6
DEFAULT_B1_THRESHOLD = 0.482
DEFAULT_EV_THRESHOLD = 0.10


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, name FROM db.stadiums").fetchall()
        return {int(r[0]): r[1] for r in rows}
    finally:
        conn.close()


def _load_trifecta_odds(db_path: str, date: str, next_day: str) -> tuple[
    dict[tuple[int, str], float],
    dict[tuple[int, int], float],
]:
    """Load trifecta odds and compute market-implied win probabilities.

    Returns:
        trifecta_odds: {(race_id, combo_str): odds}
        tri_win_prob: {(race_id, first_boat): sum(0.75/odds)}
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT o.race_id, o.combination, o.odds
        FROM db.race_odds o
        JOIN db.races r ON r.id = o.race_id
        WHERE o.bet_type = '3連単'
          AND r.race_date >= ? AND r.race_date < ?
        """,
        [date, next_day],
    ).fetchall()
    conn.close()

    trifecta_odds: dict[tuple[int, str], float] = {}
    tri_win_prob: dict[tuple[int, int], float] = defaultdict(float)

    for r in rows:
        rid, combo, odds = int(r[0]), r[1], float(r[2])
        trifecta_odds[(rid, combo)] = odds
        if odds > 0:
            tri_win_prob[(rid, int(combo.split("-")[0]))] += 0.75 / odds

    return trifecta_odds, dict(tri_win_prob)


def predict_trifecta(
    date: str,
    model_dir: str,
    db_path: str,
    b1_threshold: float | None = None,
    ev_threshold: float | None = None,
    snapshot_path: str | None = None,
    race_ids: list[int] | None = None,
) -> dict:
    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    # Build features (fast path with snapshot, or full pipeline)
    with contextlib.redirect_stdout(sys.stderr):
        if snapshot_path:
            print(f"Using snapshot: {snapshot_path}")
            df = build_features_from_snapshot(db_path, snapshot_path, date)
        else:
            df = build_features_df(db_path, start_date=date, end_date=next_day)

    if len(df) == 0:
        return {
            "date": date,
            "model_dir": model_dir,
            "n_races": 0,
            "predictions": [],
            "evaluated_race_ids": [],
            "error": f"No race data for {date}",
        }

    # Filter to specific races if requested (runner mode)
    race_id_set = set(race_ids) if race_ids else None
    if race_id_set:
        df = df[df["race_id"].isin(race_id_set)].reset_index(drop=True)
        if len(df) == 0:
            return {
                "date": date,
                "model_dir": model_dir,
                "n_races": 0,
                "predictions": [],
                "evaluated_race_ids": [],
            }

    # --- Boat1 binary model ---
    b1_dir = f"{model_dir}/boat1"
    b1_model = load_boat1_model(b1_dir)
    b1_meta = load_model_meta(b1_dir)

    b1_rows = df[df["boat_number"] == 1][
        ["race_id", "stadium_id", "race_number", "race_date"]
    ].reset_index(drop=True)

    X_b1, _, meta_b1 = reshape_to_boat1(df)

    # Fill NaN with training means
    X_b1 = X_b1.astype("float64")
    nan_cols = [c for c in X_b1.columns if X_b1[c].isna().any()]
    if nan_cols and b1_meta and b1_meta.get("feature_means"):
        for c in nan_cols:
            if c in b1_meta["feature_means"]:
                X_b1[c] = X_b1[c].fillna(b1_meta["feature_means"][c])
        n_filled = (~meta_b1["has_exhibition"]).sum()
        if n_filled > 0:
            print(
                f"Filled NaN for {n_filled} race(s) without exhibition: {nan_cols}",
                file=sys.stderr,
            )

    b1_probs = b1_model.predict_proba(X_b1)[:, 1]
    b1_map = {int(meta_b1["race_id"].values[i]): i for i in range(len(b1_probs))}

    # --- LambdaRank model ---
    rank_dir = f"{model_dir}/ranking"
    rank_model = load_model(rank_dir)
    rank_meta = load_model_meta(rank_dir)

    # Resolve strategy params from model_meta if not explicitly provided
    strategy = rank_meta.get("strategy", {}) if rank_meta else {}
    if b1_threshold is None:
        b1_threshold = strategy.get("b1_threshold", DEFAULT_B1_THRESHOLD)
    if ev_threshold is None:
        meta_ev = strategy.get("ev_threshold")
        ev_threshold = float(meta_ev) if isinstance(meta_ev, (int, float)) else DEFAULT_EV_THRESHOLD
    print(f"Strategy: b1<{b1_threshold:.3f} EV>={ev_threshold:.2f}", file=sys.stderr)

    X_rank, _, meta_rank = prepare_feature_matrix(df)

    # Fill NaN for ranking features
    X_rank = X_rank.astype("float64")
    rank_nan_cols = [c for c in X_rank.columns if X_rank[c].isna().any()]
    if rank_nan_cols and rank_meta and rank_meta.get("feature_means"):
        for c in rank_nan_cols:
            if c in rank_meta["feature_means"]:
                X_rank[c] = X_rank[c].fillna(rank_meta["feature_means"][c])

    rank_scores = rank_model.predict(X_rank)

    n_races = len(X_rank) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_rank["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids = meta_rank["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    # Softmax probabilities
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    # Top boats
    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    # Race ID → rank index
    rank_rid_map = {int(race_ids[i]): i for i in range(n_races)}

    # --- Trifecta odds ---
    trifecta_odds, tri_win_prob = _load_trifecta_odds(db_path, date, next_day)

    # --- Stadium names ---
    stadium_names = _load_stadium_names(db_path)

    # --- Build predictions ---
    predictions = []
    evaluated_race_ids = []
    skipped: dict[int, dict] = {}  # race_id → {b1_prob, ev?, reason}
    n_total = len(b1_rows)
    n_b1_pass = 0
    n_has_odds = 0
    n_ev_pass = 0

    for i in range(n_total):
        rid = int(b1_rows["race_id"].values[i])
        bi = b1_map.get(rid)
        ri = rank_rid_map.get(rid)
        if bi is None or ri is None:
            continue

        b1p = float(b1_probs[bi])
        if b1p >= b1_threshold:
            skipped[rid] = {"b1_prob": round(b1p, 4), "reason": "b1_win"}
            continue
        n_b1_pass += 1
        evaluated_race_ids.append(rid)

        # Winner pick: top non-boat-1
        wp = int(top_boats[ri, 0])
        if wp == 1:
            wp = int(top_boats[ri, 1])

        # Winner's softmax probability
        bidx = np.where(boats_2d[ri] == wp)[0]
        if len(bidx) == 0:
            skipped[rid] = {"b1_prob": round(b1p, 4), "reason": "no_winner"}
            continue
        wprob = float(rank_probs[ri, bidx[0]])

        # Trifecta-implied EV
        mkt_prob = tri_win_prob.get((rid, wp), 0)
        if mkt_prob <= 0:
            skipped[rid] = {"b1_prob": round(b1p, 4), "reason": "no_odds"}
            continue
        n_has_odds += 1

        ev = wprob / mkt_prob * 0.75 - 1
        if ev < ev_threshold:
            skipped[rid] = {"b1_prob": round(b1p, 4), "ev": round(ev, 4), "reason": "ev_low"}
            continue
        n_ev_pass += 1

        # Build X-noB1-noB1 tickets (12 combinations)
        excluded = {wp, 1}
        flow = [int(b) for b in boats_2d[ri] if int(b) not in excluded]
        tickets = []
        for b2 in flow:
            for b3 in flow:
                if b2 != b3:
                    combo = f"{wp}-{b2}-{b3}"
                    if (rid, combo) in trifecta_odds:
                        tickets.append(combo)

        if not tickets:
            continue

        sid = int(b1_rows["stadium_id"].values[i])
        predictions.append({
            "race_id": rid,
            "race_date": str(b1_rows["race_date"].values[i]),
            "stadium_id": sid,
            "stadium_name": stadium_names.get(sid, f"場{sid}"),
            "race_number": int(b1_rows["race_number"].values[i]),
            "winner_pick": wp,
            "b1_prob": round(b1p, 4),
            "winner_prob": round(wprob, 4),
            "ev": round(ev, 4),
            "tickets": tickets,
            "has_exhibition": bool(meta_b1["has_exhibition"].values[bi]),
        })

    return {
        "date": date,
        "model_dir": model_dir,
        "b1_threshold": b1_threshold,
        "ev_threshold": ev_threshold,
        "n_races": len(predictions),
        "predictions": predictions,
        "evaluated_race_ids": evaluated_race_ids,
        "skipped": skipped,
        "stats": {
            "total": n_total,
            "b1_pass": n_b1_pass,
            "has_odds": n_has_odds,
            "ev_pass": n_ev_pass,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict trifecta X-noB1-noB1 strategy"
    )
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--model-dir", default="models/trifecta_v1")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--b1-threshold", type=float, default=None,
                        help="Override b1_threshold (default: from model_meta)")
    parser.add_argument("--ev-threshold", type=float, default=None,
                        help="Override ev_threshold (default: from model_meta)")
    parser.add_argument("--snapshot", default=None,
                        help="Stats snapshot path for fast inference")
    parser.add_argument("--race-ids", default=None,
                        help="Comma-separated race IDs to predict (runner mode)")
    args = parser.parse_args()

    rid_list = None
    if args.race_ids:
        rid_list = [int(x) for x in args.race_ids.split(",")]

    result = predict_trifecta(
        args.date, args.model_dir, args.db_path,
        b1_threshold=args.b1_threshold,
        ev_threshold=args.ev_threshold,
        snapshot_path=args.snapshot,
        race_ids=rid_list,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
