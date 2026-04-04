"""Daily trifecta analysis using saved models.

Retroactively evaluates trifecta X-noB1-noB1 strategy on a given date (or range)
using the production saved models. No retraining — fast execution.

Shows:
  1. Per-race breakdown: stadium, R#, pick, b1_prob, EV, result, payout
  2. EV threshold sweep: ROI / hit rate / P/L at multiple thresholds
  3. Summary statistics

Usage:
    uv run --directory ml python -m scripts.daily_trifecta --date 2026-04-04
    uv run --directory ml python -m scripts.daily_trifecta --from 2026-04-01 --to 2026-04-05
    uv run --directory ml python -m scripts.daily_trifecta --date 2026-04-04 --json
"""

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from boatrace_tipster_ml.boat1_features import reshape_to_boat1
from boatrace_tipster_ml.boat1_model import load_boat1_model
from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.feature_config import prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta

FIELD_SIZE = 6
DEFAULT_MODEL_DIR = "models/trifecta_v1"


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, name FROM db.stadiums").fetchall()
        return {int(r[0]): r[1] for r in rows}
    finally:
        conn.close()


def _load_trifecta_odds(
    db_path: str, start_date: str, end_date: str
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, int], float]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT o.race_id, o.combination, o.odds
        FROM db.race_odds o
        JOIN db.races r ON r.id = o.race_id
        WHERE o.bet_type = '3連単'
          AND r.race_date >= ? AND r.race_date < ?
        """,
        [start_date, end_date],
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


def _load_finish_data(
    db_path: str, start_date: str, end_date: str
) -> tuple[dict[tuple[int, int], int], dict[int, str]]:
    """Load finish positions and trifecta payouts."""
    conn = get_connection(db_path)

    # Finish positions
    rows = conn.execute(
        """
        SELECT e.race_id, e.boat_number, e.finish_position
        FROM db.race_entries e
        JOIN db.races r ON r.id = e.race_id
        WHERE r.race_date >= ? AND r.race_date < ?
          AND e.finish_position IS NOT NULL
        """,
        [start_date, end_date],
    ).fetchall()
    finish_map = {(int(r[0]), int(r[1])): int(r[2]) for r in rows}

    # Trifecta payouts
    payout_rows = conn.execute(
        """
        SELECT p.race_id, p.combination, p.payout
        FROM db.race_payouts p
        JOIN db.races r ON r.id = p.race_id
        WHERE r.race_date >= ? AND r.race_date < ?
          AND p.bet_type = '3連単'
        """,
        [start_date, end_date],
    ).fetchall()
    payout_map = {(int(r[0]), r[1]): int(r[2]) for r in payout_rows}

    conn.close()
    return finish_map, payout_map


def analyze(
    start_date: str,
    end_date: str,
    model_dir: str,
    db_path: str,
) -> list[dict]:
    """Analyze all races in the date range. Returns per-race results at EV>=0."""
    # Build features
    with contextlib.redirect_stdout(sys.stderr):
        df = build_features_df(db_path, start_date=start_date, end_date=end_date)

    if len(df) == 0:
        print(f"No race data for {start_date} ~ {end_date}", file=sys.stderr)
        return []

    # Load saved models
    b1_model = load_boat1_model(f"{model_dir}/boat1")
    b1_meta = load_model_meta(f"{model_dir}/boat1")
    rank_model = load_model(f"{model_dir}/ranking")
    rank_meta = load_model_meta(f"{model_dir}/ranking")

    strategy = rank_meta.get("strategy", {}) if rank_meta else {}
    b1_threshold = strategy.get("b1_threshold", 0.482)

    # Boat1 prediction
    b1_rows = df[df["boat_number"] == 1][
        ["race_id", "stadium_id", "race_number", "race_date"]
    ].reset_index(drop=True)

    X_b1, _, meta_b1 = reshape_to_boat1(df)
    X_b1 = X_b1.astype("float64")
    nan_cols = [c for c in X_b1.columns if X_b1[c].isna().any()]
    if nan_cols and b1_meta and b1_meta.get("feature_means"):
        for c in nan_cols:
            if c in b1_meta["feature_means"]:
                X_b1[c] = X_b1[c].fillna(b1_meta["feature_means"][c])

    b1_probs = b1_model.predict_proba(X_b1)[:, 1]
    b1_map = {int(meta_b1["race_id"].values[i]): i for i in range(len(b1_probs))}

    # LambdaRank prediction
    X_rank, _, meta_rank = prepare_feature_matrix(df)
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

    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)
    rank_rid_map = {int(race_ids[i]): i for i in range(n_races)}

    # Load odds and results
    trifecta_odds, tri_win_prob = _load_trifecta_odds(db_path, start_date, end_date)
    finish_map, payout_map = _load_finish_data(db_path, start_date, end_date)
    stadium_names = _load_stadium_names(db_path)

    # Evaluate each race (EV >= 0, no exhibition filter here — flag it instead)
    results = []
    for i in range(len(b1_rows)):
        rid = int(b1_rows["race_id"].values[i])
        bi = b1_map.get(rid)
        ri = rank_rid_map.get(rid)
        if bi is None or ri is None:
            continue

        b1p = float(b1_probs[bi])
        if b1p >= b1_threshold:
            continue

        has_exh = bool(meta_b1["has_exhibition"].values[bi])

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
        if ev < 0:
            continue

        # Build tickets
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

        # Check actual result
        actual = {}
        for b in range(1, 7):
            fp = finish_map.get((rid, b))
            if fp is not None:
                actual[fp] = b

        actual_combo = None
        if 1 in actual and 2 in actual and 3 in actual:
            actual_combo = f"{actual[1]}-{actual[2]}-{actual[3]}"

        won = actual_combo is not None and actual_combo in tickets
        hit_payout = payout_map.get((rid, actual_combo), 0) if actual_combo else 0

        sid = int(b1_rows["stadium_id"].values[i])
        rno = int(b1_rows["race_number"].values[i])
        rdate = str(b1_rows["race_date"].values[i])

        results.append({
            "race_id": rid,
            "date": rdate,
            "stadium_id": sid,
            "stadium_name": stadium_names.get(sid, f"場{sid}"),
            "race_number": rno,
            "winner_pick": wp,
            "b1_prob": round(b1p, 4),
            "winner_prob": round(wprob, 4),
            "ev": round(ev, 4),
            "tickets": len(tickets),
            "has_exhibition": has_exh,
            "actual_combo": actual_combo,
            "won": won,
            "hit_payout": hit_payout,
            "pick_1st": actual.get(1) == wp if 1 in actual else None,
        })

    results.sort(key=lambda r: (r["date"], r["stadium_id"], r["race_number"]))
    return results


def print_report(results: list[dict], start_date: str, end_date: str):
    """Print human-readable report to stderr."""
    if not results:
        print("No qualifying races found.", file=sys.stderr)
        return

    # Count races with/without results
    has_result = sum(1 for r in results if r["actual_combo"] is not None)
    no_result = len(results) - has_result

    print(f"\n{'='*80}", file=sys.stderr)
    print(f"3連単 Daily Analysis: {start_date} ~ {end_date}", file=sys.stderr)
    print(f"Candidates (EV>=0%): {len(results)} races ({has_result} with results, {no_result} pending)", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)

    # Per-race detail
    print(f"\n{'Date':>10} {'Venue':>4} {'R':>2} {'Pick':>2} {'b1%':>5} {'EV':>6} {'Exh':>3} {'Result':>7} {'Hit':>5} {'Payout':>8}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_date[r["date"]].append(r)

    for date in sorted(by_date.keys()):
        for r in by_date[date]:
            exh = "o" if r["has_exhibition"] else "x"
            if r["actual_combo"] is None:
                result_str = "---"
                hit_str = ""
                payout_str = ""
            elif r["won"]:
                result_str = r["actual_combo"]
                hit_str = "WIN"
                payout_str = f"¥{r['hit_payout']:,}"
            else:
                result_str = r["actual_combo"]
                hit_str = "---"
                payout_str = ""

            pick1st = ""
            if r["pick_1st"] is not None:
                pick1st = "1着o" if r["pick_1st"] else "1着x"

            print(
                f"{r['date']:>10} {r['stadium_name']:>4} R{r['race_number']:>2} "
                f"{r['winner_pick']:>2}号 {r['b1_prob']*100:>4.1f}% {r['ev']*100:>+5.1f}% "
                f" {exh}  {result_str:>7} {hit_str:>5} {payout_str:>8}  {pick1st}",
                file=sys.stderr,
            )

    # EV threshold sweep
    ev_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50]
    exh_results = [r for r in results if r["has_exhibition"] and r["actual_combo"] is not None]

    if not exh_results:
        print("\nNo races with exhibition + results for threshold analysis.", file=sys.stderr)
        return

    print(f"\n{'='*80}", file=sys.stderr)
    print("EV Threshold Sweep (exhibition-only, results-available)", file=sys.stderr)
    print(f"{'EV':>6} {'Races':>5} {'Tkt':>5} {'Wins':>4} {'Hit%':>5} {'ROI':>6} {'P/L(¥100)':>10}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    for thr in ev_thresholds:
        filtered = [r for r in exh_results if r["ev"] >= thr]
        if not filtered:
            print(f"{thr*100:>+5.0f}%  {'---':>5}", file=sys.stderr)
            continue

        n_races = len(filtered)
        n_tickets = sum(r["tickets"] for r in filtered)
        n_wins = sum(1 for r in filtered if r["won"])
        total_payout = sum(r["hit_payout"] / 100 for r in filtered if r["won"])
        roi = total_payout / n_tickets if n_tickets > 0 else 0
        pl = total_payout - n_tickets

        marker = " **" if roi >= 1.0 else ""
        print(
            f"{thr*100:>+5.0f}% {n_races:>5} {n_tickets:>5} {n_wins:>4} "
            f"{n_wins/n_races:>4.0%} {roi:>5.0%} {pl:>+10.0f}{marker}",
            file=sys.stderr,
        )

    # Daily summary (with default threshold from model)
    print(f"\n{'='*80}", file=sys.stderr)
    print("Daily Summary (EV>=33%, exhibition-only)", file=sys.stderr)
    print(f"{'Date':>10} {'Races':>5} {'Tkt':>5} {'Wins':>4} {'P/L':>8} {'Cum':>8}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    cum = 0.0
    for date in sorted(by_date.keys()):
        day_races = [
            r for r in by_date[date]
            if r["has_exhibition"] and r["actual_combo"] is not None and r["ev"] >= 0.33
        ]
        if not day_races:
            continue

        tkt = sum(r["tickets"] for r in day_races)
        wins = sum(1 for r in day_races if r["won"])
        payout = sum(r["hit_payout"] / 100 for r in day_races if r["won"])
        pl = payout - tkt
        cum += pl
        marker = "+" if pl > 0 else "-"
        print(
            f"{date:>10} {len(day_races):>5} {tkt:>5} {wins:>4} {pl:>+7.0f} {cum:>+7.0f} {marker}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Daily trifecta analysis (saved models)"
    )
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument("--from", dest="from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", help="End date exclusive (YYYY-MM-DD)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    if args.date:
        start_date = args.date
        end_date = (
            datetime.strptime(args.date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
    elif args.from_date and args.to_date:
        start_date = args.from_date
        end_date = args.to_date
    else:
        parser.error("--date or --from/--to required")

    results = analyze(start_date, end_date, args.model_dir, args.db_path)
    print_report(results, start_date, end_date)

    if args.json:
        json.dump(
            {
                "start_date": start_date,
                "end_date": end_date,
                "model_dir": args.model_dir,
                "n_candidates": len(results),
                "races": results,
            },
            sys.stdout,
            ensure_ascii=False,
            default=str,
        )


if __name__ == "__main__":
    main()
