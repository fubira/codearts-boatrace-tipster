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
from boatrace_tipster_ml.registry import get_active_model_dir
from boatrace_tipster_ml.snapshot_features import build_features_from_snapshot
from scripts.tune_p2 import _trifecta_prob

FIELD_SIZE = 6

# Defaults (overridden by model_meta.json strategy section)
DEFAULT_GAP23_THRESHOLD = 0.13
DEFAULT_TOP3_CONC_THRESHOLD = 0.0
DEFAULT_EV_THRESHOLD = 0.0
DEFAULT_GAP12_MIN_THRESHOLD = 0.0


def build_p2_would_be_tickets(
    probs_6,
    i1: int,
    i2: int,
    i3: int,
    r2: int,
    r3: int,
    trifecta_odds: dict,
    rid: int,
) -> list[dict]:
    """Build would-be P2 ticket records for the two 1-head orderings.

    Returns 2 dicts (combos 1-r2-r3 and 1-r3-r2), each with:
      combo: str
      model_prob: rounded Plackett-Luce probability (always present)
      market_odds: rounded float, or None if odds missing / non-positive
      ev: rounded float, or None if odds missing (no threshold applied)

    Caller applies the EV threshold to pick the final buy list.
    """
    tickets: list[dict] = []
    for combo, ia, ib, ic in [
        (f"1-{r2}-{r3}", i1, i2, i3),
        (f"1-{r3}-{r2}", i1, i3, i2),
    ]:
        mp = _trifecta_prob(probs_6, ia, ib, ic)
        mkt_odds = trifecta_odds.get((rid, combo))
        if mkt_odds and mkt_odds > 0:
            ev = mp / (1.0 / mkt_odds) * 0.75 - 1
            tickets.append({
                "combo": combo,
                "model_prob": round(mp, 6),
                "market_odds": round(mkt_odds, 1),
                "ev": round(ev, 4),
            })
        else:
            tickets.append({
                "combo": combo,
                "model_prob": round(mp, 6),
                "market_odds": None,
                "ev": None,
            })
    return tickets


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
        # Use T-5 snapshot to mirror runner's initial prediction step.
        # T-1 drift check is a separate runner stage, not part of predict.
        rows = conn.execute(
            """
            SELECT s.race_id, s.combination, s.odds
            FROM db.race_odds_snapshots s
            JOIN db.races r ON r.id = s.race_id
            WHERE s.bet_type = '3連単'
              AND s.timing = 'T-5'
              AND r.race_date >= ? AND r.race_date < ?
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
    gap12_min_threshold = strategy.get("gap12_min_threshold", DEFAULT_GAP12_MIN_THRESHOLD)
    excluded_stadiums = set(strategy.get("excluded_stadiums") or [])
    print(
        f"P2 Strategy: gap12>={gap12_min_threshold:.3f} gap23>={gap23_threshold:.3f} "
        f"conc>={top3_conc_threshold:.3f} ev>={ev_threshold:.3f} "
        f"excluded_stadiums={sorted(excluded_stadiums) or 'none'}",
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

    # bc_* (oriten exhibition) data is timing-sensitive: scraper fetches at T-7
    # but the runner predicts at T-5, so a race occasionally lands with NULL
    # bc_*. fill_nan_with_means then collapses bc_*_zscore to 0, which silently
    # weakens the prediction. Surface this per-race so the log can flag it.
    bc_cols = [c for c in
               ("bc_lap_time", "bc_turn_time", "bc_straight_time", "bc_slit_diff")
               if c in df.columns]
    if bc_cols:
        bc_full_per_boat = df[bc_cols].notna().all(axis=1).astype(int)
        bc_count_per_race = bc_full_per_boat.groupby(df["race_id"]).sum().to_dict()
    else:
        bc_count_per_race = {}

    def _bc_status(rid: int) -> str:
        n = int(bc_count_per_race.get(rid, 0))
        if n == FIELD_SIZE:
            return "full"
        if n == 0:
            return "missing"
        return f"partial:{n}/{FIELD_SIZE}"

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
    n_gap12_pass = 0
    n_conc_pass = 0
    n_gap23_pass = 0
    n_predicted = 0

    for i in range(n_races):
        rid = int(race_ids_arr[i])
        po = pred_order[i]
        probs = model_probs[i]
        bc_status = _bc_status(rid)

        # Filter 0: stadium exclusion (structurally unprofitable venues)
        if excluded_stadiums:
            b1_idx = b1_rid_map.get(rid)
            if b1_idx is not None:
                sid = int(b1_rows["stadium_id"].values[b1_idx])
                if sid in excluded_stadiums:
                    skipped[rid] = {
                        "reason": "stadium_excluded",
                        "stadium_id": sid,
                        "bc_status": bc_status,
                    }
                    continue

        # Filter 1: top-1 must be boat 1
        if top_boats[i, 0] != 1:
            skipped[rid] = {
                "reason": "not_b1_top",
                "top1_boat": int(top_boats[i, 0]),
                "bc_status": bc_status,
            }
            continue
        n_b1_top += 1
        evaluated_race_ids.append(rid)

        p1 = float(probs[po[0]])
        p2 = float(probs[po[1]])
        p3 = float(probs[po[2]])

        # Compute would-be P2 tickets for ALL races that have boat1 on top,
        # regardless of whether the race ultimately passes the later filters —
        # SKIP logs show "which tickets were cut" for post-hoc threshold review.
        r2, r3 = int(top_boats[i, 1]), int(top_boats[i, 2])
        i1, i2, i3 = po[0], po[1], po[2]
        would_be_tickets = build_p2_would_be_tickets(
            probs, i1, i2, i3, r2, r3, trifecta_odds, rid,
        )

        # Filter 2: gap12 (model must show meaningful confidence gap between 1 and 2)
        gap12 = p1 - p2
        if gap12 < gap12_min_threshold:
            skipped[rid] = {
                "reason": "gap12_low",
                "gap12": round(gap12, 4),
                "would_be_tickets": would_be_tickets,
                "bc_status": bc_status,
            }
            continue
        n_gap12_pass += 1

        # Filter 3: top3_concentration
        top3_conc = (p2 + p3) / (1 - p1 + 1e-10)
        if top3_conc < top3_conc_threshold:
            skipped[rid] = {
                "reason": "top3_conc_low",
                "top3_conc": round(top3_conc, 4),
                "would_be_tickets": would_be_tickets,
                "bc_status": bc_status,
            }
            continue
        n_conc_pass += 1

        # Filter 4: gap23
        gap23 = p2 - p3
        if gap23 < gap23_threshold:
            skipped[rid] = {
                "reason": "gap23_low",
                "gap23": round(gap23, 4),
                "would_be_tickets": would_be_tickets,
                "bc_status": bc_status,
            }
            continue
        n_gap23_pass += 1

        # Apply per-ticket EV filter to would_be_tickets to finalize the buy list
        tickets = [
            t for t in would_be_tickets
            if t["market_odds"] is not None and t["ev"] is not None and t["ev"] >= ev_threshold
        ]

        if not tickets:
            skipped[rid] = {
                "reason": "no_ev_tickets",
                "gap23": round(gap23, 4),
                "top3_conc": round(top3_conc, 4),
                "would_be_tickets": would_be_tickets,
                "bc_status": bc_status,
            }
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
            "gap12": round(gap12, 4),
            "tickets": tickets,
            "has_exhibition": True,  # always True for non-odds features
            "bc_status": bc_status,
        })

    return {
        "date": date,
        "model_dir": model_dir,
        "strategy": "P2",
        "gap23_threshold": gap23_threshold,
        "top3_conc_threshold": top3_conc_threshold,
        "gap12_min_threshold": gap12_min_threshold,
        "ev_threshold": ev_threshold,
        "n_races": len(predictions),
        "predictions": predictions,
        "evaluated_race_ids": evaluated_race_ids,
        "skipped": skipped,
        "stats": {
            "total": n_total,
            "b1_top": n_b1_top,
            "gap12_pass": n_gap12_pass,
            "conc_pass": n_conc_pass,
            "gap23_pass": n_gap23_pass,
            "predicted": n_predicted,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="P2 trifecta prediction")
    parser.add_argument("--date", required=True)
    parser.add_argument("--model-dir", default=get_active_model_dir())
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
