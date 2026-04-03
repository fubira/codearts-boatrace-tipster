"""Predict boat 1 win probability and anti-favorite recommendation.

Outputs two types of predictions per race:
  - boat1: probability of boat 1 winning, tansho EV
  - anti_favorite: when boat1 prob < threshold, LambdaRank top non-boat1 pick

Usage:
    uv run --directory ml python -m scripts.predict_boat1 --date 2026-04-02
"""

import argparse
import contextlib
import json
import sys
from datetime import datetime, timedelta

import numpy as np

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import load_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import FEATURE_COLS, prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, name FROM db.stadiums").fetchall()
        return {int(r[0]): r[1] for r in rows}
    finally:
        conn.close()


FIELD_SIZE = 6
ANTI_FAVORITE_THRESHOLD = 0.40


def predict_boat1(date: str, model_dir: str, db_path: str, ranking_model_dir: str | None = None) -> dict:
    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # Build features (progress logs go to stderr)
    with contextlib.redirect_stdout(sys.stderr):
        df = build_features_df(db_path, start_date=date, end_date=next_day)

    if len(df) == 0:
        return {"date": date, "model_dir": model_dir, "model_meta": None,
                "n_races": 0, "predictions": [], "error": f"No race data for {date}"}

    # Extract race info before reshape
    b1_rows = df[df["boat_number"] == 1][["race_id", "stadium_id", "race_number", "race_date"]].reset_index(drop=True)

    X, y, meta = reshape_to_boat1(df)
    assert len(b1_rows) == len(X), f"Row mismatch: b1_rows={len(b1_rows)} vs X={len(X)}"

    # Load boat1 binary model
    model = load_boat1_model(model_dir)
    model_meta = load_model_meta(model_dir)

    # Fill NaN features with training data means for robustness
    X = X.astype("float64")
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    if nan_cols:
        feature_means = model_meta.get("feature_means") if model_meta else None
        if feature_means:
            for c in nan_cols:
                if c in feature_means:
                    X[c] = X[c].fillna(feature_means[c])
        n_no_exh = (~meta["has_exhibition"]).sum()
        if n_no_exh > 0:
            print(f"Filled NaN features for {n_no_exh} race(s) without exhibition data: {nan_cols}", file=sys.stderr)

    # Predict boat1 probabilities
    probs = model.predict_proba(X)[:, 1]
    odds = meta["b1_tansho_odds"].values

    # --- LambdaRank for anti-favorite strategy ---
    rank_data = None
    if ranking_model_dir:
        try:
            ranking_model = load_model(ranking_model_dir)
            ranking_meta = load_model_meta(ranking_model_dir)

            X_rank, _, meta_rank = prepare_feature_matrix(df)

            # Fill NaN for ranking features
            X_rank = X_rank.astype("float64")
            rank_nan_cols = [c for c in X_rank.columns if X_rank[c].isna().any()]
            if rank_nan_cols and ranking_meta and ranking_meta.get("feature_means"):
                for c in rank_nan_cols:
                    if c in ranking_meta["feature_means"]:
                        X_rank[c] = X_rank[c].fillna(ranking_meta["feature_means"][c])

            rank_scores = ranking_model.predict(X_rank)

            n_races = len(X_rank) // FIELD_SIZE
            scores_2d = rank_scores.reshape(n_races, FIELD_SIZE)
            boats_2d = meta_rank["boat_number"].values.reshape(n_races, FIELD_SIZE)
            race_ids_2d = meta_rank["race_id"].values.reshape(n_races, FIELD_SIZE)

            # Softmax probabilities
            exp_scores = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
            rank_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Top boats by predicted rank
            pred_order = np.argsort(-scores_2d, axis=1)
            top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)
            top_probs = np.take_along_axis(rank_probs, pred_order, axis=1)

            # Build lookup: race_id → (top_boats, top_probs)
            rank_data = {}
            for i in range(n_races):
                rid = int(race_ids_2d[i, 0])
                rank_data[rid] = {
                    "top_boats": top_boats[i].tolist(),
                    "top_probs": top_probs[i].tolist(),
                }
            print(f"LambdaRank loaded: {n_races} races ranked", file=sys.stderr)
        except Exception as e:
            print(f"LambdaRank failed: {e}", file=sys.stderr)

    # Stadium names
    stadium_names = _load_stadium_names(db_path)

    # Build predictions
    predictions = []
    for i in range(len(probs)):
        p = float(probs[i])
        o = float(odds[i]) if not np.isnan(odds[i]) else None
        ev = round((p * o - 1) * 100, 1) if o is not None else None

        sid = int(b1_rows["stadium_id"].values[i])
        rid = int(b1_rows["race_id"].values[i])

        pred = {
            "race_id": rid,
            "race_date": str(b1_rows["race_date"].values[i]),
            "stadium_id": sid,
            "stadium_name": stadium_names.get(sid, f"場{sid}"),
            "race_number": int(b1_rows["race_number"].values[i]),
            "prob": round(p, 4),
            "tansho_odds": o,
            "ev": ev,
            "recommend": ev is not None and ev > 0,
            "has_exhibition": bool(meta["has_exhibition"].values[i]),
        }

        # Anti-favorite recommendation
        if rank_data and p < ANTI_FAVORITE_THRESHOLD:
            rd = rank_data.get(rid)
            if rd:
                # Pick top non-boat-1 boat
                for rank_idx in range(FIELD_SIZE):
                    boat = rd["top_boats"][rank_idx]
                    if boat != 1:
                        pred["anti_favorite"] = {
                            "boat_number": boat,
                            "rank_prob": round(rd["top_probs"][rank_idx], 4),
                        }
                        break

        predictions.append(pred)

    return {
        "date": date,
        "model_dir": model_dir,
        "model_meta": model_meta,
        "n_races": len(predictions),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict boat 1 win probability")
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--model-dir", default="models/boat1")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--ranking-model-dir", default="models/ranking")
    args = parser.parse_args()

    result = predict_boat1(args.date, args.model_dir, args.db_path, args.ranking_model_dir)
    json.dump(result, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
