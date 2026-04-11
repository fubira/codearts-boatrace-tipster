"""Predict P2 trifecta strategy (B1 top-1, adaptive 1-2 tickets).

Flow:
  1. LambdaRank (Non-odds 21 features): rank boats per race
  2. Filter: top-1 must be boat 1
  3. Filter: top3_concentration >= threshold
  4. Filter: gap23 >= threshold
  5. P2 tickets: 1-(rank2,rank3)-(rank2,rank3), per-ticket EV check
  6. Return predictions with per-ticket model_prob, market_odds, ev

Usage:
    uv run --directory ml python -m scripts.predict_p2 --date 2026-04-04
    uv run --directory ml python -m scripts.predict_p2 --date 2026-04-04 --use-snapshots
"""

import argparse
import contextlib
import json
import sys
from datetime import datetime, timedelta

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
from boatrace_tipster_ml.snapshot_features import build_features_from_snapshot
from scripts.tune_p2 import _trifecta_prob

FIELD_SIZE = 6

# Defaults (overridden by model_meta.json strategy section)
DEFAULT_GAP23_THRESHOLD = 0.13
DEFAULT_TOP3_CONC_THRESHOLD = 0.0
DEFAULT_EV_THRESHOLD = 0.0


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, name FROM db.stadiums").fetchall()
        return {int(r[0]): r[1] for r in rows}
    finally:
        conn.close()


def _load_trifecta_odds(
    db_path: str, date: str, next_day: str, *, use_snapshots: bool = False,
) -> dict[tuple[int, str], float]:
    """Load trifecta odds. Returns {(race_id, combo_str): odds}."""
    conn = get_connection(db_path)
    if use_snapshots:
        rows = conn.execute(
            """
            SELECT s.race_id, s.combination, s.odds
            FROM db.race_odds_snapshots s
            JOIN db.races r ON r.id = s.race_id
            WHERE s.bet_type = '3連単'
              AND r.race_date >= ? AND r.race_date < ?
              AND s.id IN (
                SELECT MAX(s2.id)
                FROM db.race_odds_snapshots s2
                WHERE s2.race_id = s.race_id
                  AND s2.bet_type = s.bet_type
                  AND s2.combination = s.combination
              )
            """,
            [date, next_day],
        ).fetchall()
    else:
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
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def predict_p2(
    date: str,
    model_dir: str,
    db_path: str,
    snapshot_path: str | None = None,
    race_ids: list[int] | None = None,
    use_snapshots: bool = False,
) -> dict:
    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    # Build features
    with contextlib.redirect_stdout(sys.stderr):
        if snapshot_path:
            print(f"Using snapshot: {snapshot_path}")
            df = build_features_from_snapshot(db_path, snapshot_path, date)
        else:
            df = build_features_df(db_path, start_date=date, end_date=next_day)

    if len(df) == 0:
        return {
            "date": date,
            "n_races": 0,
            "predictions": [],
            "evaluated_race_ids": [],
            "skipped": {},
        }

    # Filter to specific races if requested (runner mode)
    if race_ids:
        race_id_set = set(race_ids)
        df = df[df["race_id"].isin(race_id_set)].reset_index(drop=True)
        if len(df) == 0:
            return {
                "date": date,
                "n_races": 0,
                "predictions": [],
                "evaluated_race_ids": [],
                "skipped": {},
            }

    # Load ranking model
    rank_dir = f"{model_dir}/ranking"
    rank_model = load_model(rank_dir)
    rank_meta = load_model_meta(rank_dir)

    # Resolve strategy thresholds from model_meta
    strategy = rank_meta.get("strategy", {}) if rank_meta else {}
    gap23_threshold = strategy.get("gap23_threshold", DEFAULT_GAP23_THRESHOLD)
    top3_conc_threshold = strategy.get("top3_conc_threshold", DEFAULT_TOP3_CONC_THRESHOLD)
    ev_threshold = strategy.get("ev_threshold", DEFAULT_EV_THRESHOLD)
    print(
        f"P2 Strategy: gap23>={gap23_threshold:.3f} conc>={top3_conc_threshold:.3f} ev>={ev_threshold:.3f}",
        file=sys.stderr,
    )

    # Prepare feature matrix using model_meta's feature_columns
    feature_cols = rank_meta.get("feature_columns") if rank_meta else None
    if not feature_cols:
        print("ERROR: model_meta.json missing feature_columns", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing features in data: {missing}", file=sys.stderr)
        sys.exit(1)

    # Build meta for race-level info
    meta_cols = ["race_id", "boat_number", "stadium_id", "race_number", "race_date"]
    available_meta = [c for c in meta_cols if c in df.columns]
    meta_df = df[available_meta].copy()

    X = df[feature_cols].copy()
    fill_nan_with_means(X, rank_meta)
    rank_scores = rank_model.predict(X)

    n_races = len(X) // FIELD_SIZE
    scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta_df["boat_number"].values.reshape(n_races, FIELD_SIZE)
    race_ids_arr = meta_df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]

    # Softmax probabilities
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)

    # Load trifecta odds and stadium names
    trifecta_odds = _load_trifecta_odds(
        db_path, date, next_day, use_snapshots=use_snapshots
    )
    stadium_names = _load_stadium_names(db_path)

    # Race-level metadata (boat_number == 1 rows for stadium_id, race_number)
    b1_rows = df[df["boat_number"] == 1][available_meta].reset_index(drop=True)
    b1_rid_map = {int(b1_rows["race_id"].values[i]): i for i in range(len(b1_rows))}

    # Build predictions
    predictions = []
    evaluated_race_ids = []
    skipped: dict[int, dict] = {}
    n_total = n_races
    n_b1_top = 0
    n_conc_pass = 0
    n_gap23_pass = 0
    n_predicted = 0

    for i in range(n_races):
        rid = int(race_ids_arr[i])
        po = pred_order[i]
        probs = model_probs[i]

        # Filter 1: top-1 must be boat 1
        if top_boats[i, 0] != 1:
            skipped[rid] = {"reason": "not_b1_top", "top1_boat": int(top_boats[i, 0])}
            continue
        n_b1_top += 1
        evaluated_race_ids.append(rid)

        p1 = float(probs[po[0]])
        p2 = float(probs[po[1]])
        p3 = float(probs[po[2]])

        # Filter 2: top3_concentration
        top3_conc = (p2 + p3) / (1 - p1 + 1e-10)
        if top3_conc < top3_conc_threshold:
            skipped[rid] = {"reason": "top3_conc_low", "top3_conc": round(top3_conc, 4)}
            continue
        n_conc_pass += 1

        # Filter 3: gap23
        gap23 = p2 - p3
        if gap23 < gap23_threshold:
            skipped[rid] = {"reason": "gap23_low", "gap23": round(gap23, 4)}
            continue
        n_gap23_pass += 1

        # P2 tickets
        r2, r3 = int(top_boats[i, 1]), int(top_boats[i, 2])
        i1, i2, i3 = po[0], po[1], po[2]

        tickets = []
        for combo, mp_fn in [
            (f"1-{r2}-{r3}", lambda: _trifecta_prob(probs, i1, i2, i3)),
            (f"1-{r3}-{r2}", lambda: _trifecta_prob(probs, i1, i3, i2)),
        ]:
            mkt_odds = trifecta_odds.get((rid, combo))
            if not mkt_odds or mkt_odds <= 0:
                continue
            mp = mp_fn()
            ev = mp / (1.0 / mkt_odds) * 0.75 - 1
            if ev >= ev_threshold:
                tickets.append({
                    "combo": combo,
                    "model_prob": round(mp, 6),
                    "market_odds": round(mkt_odds, 1),
                    "ev": round(ev, 4),
                })

        if not tickets:
            skipped[rid] = {"reason": "no_ev_tickets", "gap23": round(gap23, 4),
                            "top3_conc": round(top3_conc, 4)}
            continue
        n_predicted += 1

        # Race metadata
        bi = b1_rid_map.get(rid)
        sid = int(b1_rows["stadium_id"].values[bi]) if bi is not None else 0
        rnum = int(b1_rows["race_number"].values[bi]) if bi is not None else 0
        rdate = str(b1_rows["race_date"].values[bi]) if bi is not None else date

        predictions.append({
            "race_id": rid,
            "race_date": rdate,
            "stadium_id": sid,
            "stadium_name": stadium_names.get(sid, f"場{sid}"),
            "race_number": rnum,
            "top3_conc": round(top3_conc, 4),
            "gap23": round(gap23, 4),
            "tickets": tickets,
            "has_exhibition": True,  # always True for non-odds features
        })

    return {
        "date": date,
        "model_dir": model_dir,
        "strategy": "P2",
        "gap23_threshold": gap23_threshold,
        "top3_conc_threshold": top3_conc_threshold,
        "ev_threshold": ev_threshold,
        "n_races": len(predictions),
        "predictions": predictions,
        "evaluated_race_ids": evaluated_race_ids,
        "skipped": skipped,
        "stats": {
            "total": n_total,
            "b1_top": n_b1_top,
            "conc_pass": n_conc_pass,
            "gap23_pass": n_gap23_pass,
            "predicted": n_predicted,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="P2 trifecta prediction")
    parser.add_argument("--date", required=True)
    parser.add_argument("--model-dir", default="models/trifecta_v1")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--race-ids", default=None,
                        help="Comma-separated race IDs (runner mode)")
    parser.add_argument("--use-snapshots", action="store_true",
                        help="Read odds from race_odds_snapshots instead of race_odds")
    parser.add_argument("--json", action="store_true", default=True)
    args = parser.parse_args()

    race_ids = [int(x) for x in args.race_ids.split(",")] if args.race_ids else None

    result = predict_p2(
        date=args.date,
        model_dir=args.model_dir,
        db_path=args.db_path,
        snapshot_path=args.snapshot,
        race_ids=race_ids,
        use_snapshots=args.use_snapshots,
    )

    json.dump(result, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
