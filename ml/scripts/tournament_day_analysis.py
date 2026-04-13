"""Stratify baseline P2 performance by tournament day (開催日 1-6).

Tournament day is computed per stadium as consecutive-date runs: the first
race_date of a run is day 1, the next is day 2, etc. Any gap > 1 day starts
a new tournament.

Hypothesis: prediction accuracy may vary across the tournament because:
  - Day 1 has no in-tournament info about current form / motor condition
  - Day 5-6 is 準優勝戦 / 優勝戦 where field is filtered to strong racers

Usage:
    cd ml && uv run python scripts/tournament_day_analysis.py \\
        --from 2026-01-01 --to 2026-04-13
"""

import argparse
import contextlib
import io
import sys

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
from boatrace_tipster_ml.registry import get_active_model_dir

FIELD_SIZE = 6
UNIT_COST = 100


def _trifecta_prob(probs_6, i1, i2, i3):
    p1, p2, p3 = probs_6[i1], probs_6[i2], probs_6[i3]
    if p1 >= 1.0 or (p1 + p2) >= 1.0:
        return 0.0
    return p1 * (p2 / (1 - p1)) * (p3 / (1 - p1 - p2))


def _load_confirmed_odds(db_path, from_date, to_date):
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT o.race_id, o.combination, o.odds
        FROM db.race_odds o
        JOIN db.races r ON r.id = o.race_id
        WHERE o.bet_type = '3連単'
          AND r.race_date >= ? AND r.race_date < ?
          AND o.odds > 0
        """,
        [from_date, to_date],
    ).fetchall()
    conn.close()
    return {(int(r[0]), r[1]): float(r[2]) for r in rows}


def _load_tournament_day_map(db_path, from_date, to_date):
    """Return {race_id: day_of_tournament (1-indexed)} for races in the window.

    Tournament day is derived from consecutive-date runs per stadium. A new
    tournament starts whenever the gap from the previous stadium-race-date
    exceeds 1 day. Uses a look-back of 10 days before from_date so the first
    few days of the window are correctly labeled mid-tournament.
    """
    conn = get_connection(db_path)
    # Look back 10 days so tournaments crossing the from_date boundary are
    # labeled correctly.
    rows = conn.execute(
        """
        WITH base AS (
            SELECT id, stadium_id, race_date
            FROM db.races
            WHERE race_date >= CAST(CAST(? AS DATE) - INTERVAL 10 DAY AS VARCHAR)
              AND race_date < ?
        ),
        distinct_dates AS (
            SELECT DISTINCT stadium_id, race_date FROM base
        ),
        with_prev AS (
            SELECT
                stadium_id, race_date,
                LAG(race_date) OVER (PARTITION BY stadium_id ORDER BY race_date) AS prev_date
            FROM distinct_dates
        ),
        with_new_flag AS (
            SELECT
                stadium_id, race_date,
                CASE
                    WHEN prev_date IS NULL
                         OR (CAST(race_date AS DATE) - CAST(prev_date AS DATE)) > 1
                    THEN 1 ELSE 0
                END AS is_new_tournament
            FROM with_prev
        ),
        with_tour_id AS (
            SELECT
                stadium_id, race_date,
                SUM(is_new_tournament) OVER (PARTITION BY stadium_id ORDER BY race_date) AS tour_id
            FROM with_new_flag
        ),
        with_day AS (
            SELECT
                stadium_id, race_date, tour_id,
                ROW_NUMBER() OVER (PARTITION BY stadium_id, tour_id ORDER BY race_date) AS day_of_tour
            FROM with_tour_id
        )
        SELECT b.id, w.day_of_tour
        FROM base b
        JOIN with_day w
          ON w.stadium_id = b.stadium_id AND w.race_date = b.race_date
        WHERE b.race_date >= ?
        """,
        [from_date, to_date, from_date],
    ).fetchall()
    conn.close()
    return {int(r[0]): int(r[1]) for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--model-dir", default=get_active_model_dir())
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    print("Loading features...", file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    rank_dir = f"{args.model_dir}/ranking"
    rank_model = load_model(rank_dir)
    rank_meta = load_model_meta(rank_dir)
    feature_cols = rank_meta["feature_columns"]
    strategy = rank_meta["strategy"]
    gap12_th = strategy.get("gap12_min_threshold", 0.0)
    conc_th = strategy.get("top3_conc_threshold", 0.0)
    gap23_th = strategy.get("gap23_threshold", 0.0)
    ev_th = strategy.get("ev_threshold", 0.0)

    print(
        f"Strategy: gap12≥{gap12_th} conc≥{conc_th} gap23≥{gap23_th} ev≥{ev_th}",
        file=sys.stderr,
    )

    odds = _load_confirmed_odds(args.db_path, args.from_date, args.to_date)
    tour_day_map = _load_tournament_day_map(args.db_path, args.from_date, args.to_date)
    print(f"Tournament day mapping loaded: {len(tour_day_map)} races", file=sys.stderr)

    test_df = df[(df["race_date"] >= args.from_date) & (df["race_date"] < args.to_date)]
    X = test_df[feature_cols].copy()
    fill_nan_with_means(X, rank_meta)
    meta = test_df[["race_id", "boat_number", "finish_position"]].copy()
    scores = rank_model.predict(X)

    n_races = len(X) // FIELD_SIZE
    scores_2d = scores.reshape(n_races, FIELD_SIZE)
    boats_2d = meta["boat_number"].values.reshape(n_races, FIELD_SIZE)
    rids = meta["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    race_dates = test_df["race_date"].values.reshape(n_races, FIELD_SIZE)[:, 0]
    y_2d = meta["finish_position"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1).astype(int)
    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    model_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    actual_order = np.argsort(y_2d, axis=1)
    actual_boats = np.take_along_axis(boats_2d, actual_order, axis=1).astype(int)

    records = []
    for i in range(n_races):
        if top_boats[i, 0] != 1:
            continue
        po = pred_order[i]
        probs = model_probs[i]
        p1, p2, p3 = float(probs[po[0]]), float(probs[po[1]]), float(probs[po[2]])

        if (p1 - p2) < gap12_th:
            continue
        if (p2 + p3) / (1 - p1 + 1e-10) < conc_th:
            continue
        if (p2 - p3) < gap23_th:
            continue

        rid = int(rids[i])
        day = tour_day_map.get(rid)
        if day is None:
            continue  # race without tournament day (edge case)

        r2, r3 = int(top_boats[i, 1]), int(top_boats[i, 2])
        i1, i2, i3 = po[0], po[1], po[2]
        a1, a2, a3 = int(actual_boats[i, 0]), int(actual_boats[i, 1]), int(actual_boats[i, 2])
        hit_combo = f"{a1}-{a2}-{a3}"

        tickets = []
        for combo, ia, ib, ic in [
            (f"1-{r2}-{r3}", i1, i2, i3),
            (f"1-{r3}-{r2}", i1, i3, i2),
        ]:
            mkt = odds.get((rid, combo))
            if not mkt or mkt <= 0:
                continue
            mp = _trifecta_prob(probs, ia, ib, ic)
            ev = mp / (1.0 / mkt) * 0.75 - 1
            if ev >= ev_th:
                tickets.append({"combo": combo, "odds": mkt})

        if not tickets:
            continue

        payout = 0.0
        won = False
        for t in tickets:
            if t["combo"] == hit_combo:
                payout = t["odds"] * UNIT_COST
                won = True
                break

        records.append({
            "day": day,
            "race_date": str(race_dates[i]),
            "n_tickets": len(tickets),
            "cost": len(tickets) * UNIT_COST,
            "payout": payout,
            "won": won,
        })

    n = len(records)
    if n == 0:
        print("No records after filters.")
        return

    total_cost = sum(r["cost"] for r in records)
    total_payout = sum(r["payout"] for r in records)
    total_wins = sum(1 for r in records if r["won"])
    total_tickets = sum(r["n_tickets"] for r in records)

    print(f"\n=== Baseline candidate races: {n} ===")
    print(
        f"Total: {n}R / {total_tickets}T / {total_wins}W "
        f"hit={100*total_wins/n:.1f}% ROI={100*total_payout/total_cost:.0f}% "
        f"P/L={total_payout-total_cost:+,.0f}"
    )

    # Stratify by day
    print(f"\n=== 開催日別（OOS 2026-01-01〜{args.to_date}）===")
    print(
        f"{'開催日':>6} {'レース':>6} {'券':>4} {'勝':>4} {'hit%':>6} "
        f"{'平均配当':>10} {'ROI%':>6} {'P/L':>10}"
    )
    print("-" * 70)

    days_sorted = sorted(set(r["day"] for r in records))
    for day in days_sorted:
        bin_recs = [r for r in records if r["day"] == day]
        bn = len(bin_recs)
        bt = sum(r["n_tickets"] for r in bin_recs)
        bw = sum(1 for r in bin_recs if r["won"])
        bc = sum(r["cost"] for r in bin_recs)
        bp = sum(r["payout"] for r in bin_recs)
        avg_payout = (bp / bw) if bw > 0 else 0
        print(
            f"{day:>5}日 {bn:>6} {bt:>4} {bw:>4} {100*bw/bn:>5.1f}% "
            f"{avg_payout:>10,.0f} {100*bp/bc if bc>0 else 0:>5.0f}% {bp-bc:>+10,.0f}"
        )

    # 2-group comparison: early (days 1-2) vs late (days 5-6)
    print(f"\n=== 2 グループ比較 ===")
    early = [r for r in records if r["day"] <= 2]
    mid = [r for r in records if 3 <= r["day"] <= 4]
    late = [r for r in records if r["day"] >= 5]
    for label, group in [("初日-2日目", early), ("3日目-4日目", mid), ("5日目以降", late)]:
        if not group:
            continue
        gn = len(group)
        gw = sum(1 for r in group if r["won"])
        gc = sum(r["cost"] for r in group)
        gp = sum(r["payout"] for r in group)
        print(
            f"  {label}: {gn}R / {gw}W / hit={100*gw/gn:.1f}% / "
            f"ROI={100*gp/gc if gc>0 else 0:.0f}% / P/L={gp-gc:+,.0f}"
        )

    # Month-by-month day 1 vs day 2+ reproducibility check
    print(f"\n=== 月別 day 1 vs day 2+ 再現性 ===")
    print(
        f"{'月':>8} {'day':>4} {'レース':>6} {'勝':>4} {'hit%':>6} "
        f"{'ROI%':>6} {'P/L':>10}"
    )
    print("-" * 60)
    months = sorted(set(r["race_date"][:7] for r in records))
    for month in months:
        d1 = [r for r in records if r["race_date"].startswith(month) and r["day"] == 1]
        d2p = [r for r in records if r["race_date"].startswith(month) and r["day"] >= 2]
        for label, group in [("day 1", d1), ("day 2+", d2p)]:
            if not group:
                continue
            gn = len(group)
            gw = sum(1 for r in group if r["won"])
            gc = sum(r["cost"] for r in group)
            gp = sum(r["payout"] for r in group)
            print(
                f"{month:>8} {label:>4} {gn:>6} {gw:>4} {100*gw/gn:>5.1f}% "
                f"{100*gp/gc if gc>0 else 0:>5.0f}% {gp-gc:>+10,.0f}"
            )
        print()

    # Optional: race_number breakdown for late-tournament races (finals)
    # This doesn't require race_number — we approximate finals by day 5 (準優) / day 6 (優勝戦)
    print(f"\n=== 準優 (day 5) / 優勝戦 (day 6) の hit 率確認 ===")
    for label, day in [("day 5 (準優想定)", 5), ("day 6 (優勝戦想定)", 6)]:
        group = [r for r in records if r["day"] == day]
        if not group:
            continue
        gn = len(group)
        gw = sum(1 for r in group if r["won"])
        gc = sum(r["cost"] for r in group)
        gp = sum(r["payout"] for r in group)
        print(
            f"  {label}: {gn}R / {gw}W / hit={100*gw/gn:.1f}% / "
            f"ROI={100*gp/gc if gc>0 else 0:.0f}% / P/L={gp-gc:+,.0f}"
        )


if __name__ == "__main__":
    main()
