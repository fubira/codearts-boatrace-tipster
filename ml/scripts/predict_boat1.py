"""Predict boat 1 win probability and EV for a target date.

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
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model_meta


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, name FROM db.stadiums").fetchall()
        return {int(r[0]): r[1] for r in rows}
    finally:
        conn.close()


def predict_boat1(date: str, model_dir: str, db_path: str) -> dict:
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

    # Load model
    model = load_boat1_model(model_dir)
    model_meta = load_model_meta(model_dir)

    # Fill NaN features with training data means for robustness
    # (exhibition/weather data may not be available yet for upcoming races)
    X = X.astype("float64")
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    if nan_cols:
        # Build reference means from recent training data
        ref_end = date
        ref_start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
        with contextlib.redirect_stdout(sys.stderr):
            ref_df = build_features_df(db_path, start_date=ref_start, end_date=ref_end)
        if len(ref_df) > 0:
            ref_X, _, _ = reshape_to_boat1(ref_df)
            ref_X = ref_X.astype("float64")
            for c in nan_cols:
                fill_val = ref_X[c].mean()
                X[c] = X[c].fillna(fill_val)
            n_no_exh = (~meta["has_exhibition"]).sum()
            print(f"Filled NaN features for {n_no_exh} race(s) without exhibition data: {nan_cols}", file=sys.stderr)

    # Predict
    probs = model.predict_proba(X)[:, 1]
    odds = meta["b1_tansho_odds"].values

    # Stadium names
    stadium_names = _load_stadium_names(db_path)

    # Build predictions
    predictions = []
    for i in range(len(probs)):
        p = float(probs[i])
        o = float(odds[i]) if not np.isnan(odds[i]) else None
        ev = round((p * o - 1) * 100, 1) if o is not None else None
        recommend = ev is not None and ev > 0

        sid = int(b1_rows["stadium_id"].values[i])
        predictions.append({
            "race_id": int(b1_rows["race_id"].values[i]),
            "race_date": str(b1_rows["race_date"].values[i]),
            "stadium_id": sid,
            "stadium_name": stadium_names.get(sid, f"場{sid}"),
            "race_number": int(b1_rows["race_number"].values[i]),
            "prob": round(p, 4),
            "tansho_odds": o,
            "ev": ev,
            "recommend": recommend,
            "has_exhibition": bool(meta["has_exhibition"].values[i]),
        })

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
    args = parser.parse_args()

    result = predict_boat1(args.date, args.model_dir, args.db_path)
    json.dump(result, sys.stdout, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
