"""Analyze odds drift between T-3/T-1 and confirmed odds.

Computes market probability shifts and EV threshold accuracy for
T-3+T-1 extrapolation model.

Usage:
    uv run --directory ml python -m scripts.analyze_odds_drift
"""

import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection


def main():
    conn = get_connection(DEFAULT_DB_PATH)

    # Load all snapshots
    rows = conn.execute("""
        SELECT s.race_id, s.timing, s.combination, s.odds, r.race_date
        FROM db.race_odds_snapshots s
        JOIN db.races r ON r.id = s.race_id
        WHERE s.bet_type = '3連単' AND s.odds > 0
        ORDER BY s.race_id, s.timing
    """).fetchall()
    conn.close()

    # Build market probability per (race_id, boat, timing)
    # mp = sum(0.75 / odds) for all combos starting with that boat
    mp: dict[tuple[int, str, int], float] = defaultdict(float)
    dates: dict[int, str] = {}
    for race_id, timing, combo, odds, race_date in rows:
        first_boat = int(combo.split("-")[0])
        mp[(race_id, timing, first_boat)] += 0.75 / odds
        dates[race_id] = race_date

    # Find races that have all 3 timings (T-3, T-1, final)
    race_boats: dict[int, set[int]] = defaultdict(set)
    for (race_id, timing, boat) in mp:
        race_boats[race_id].add(boat)

    valid_triples = []
    for race_id in sorted(set(r[0] for r in rows)):
        for boat in range(1, 7):
            t3 = mp.get((race_id, "T-3", boat))
            t1 = mp.get((race_id, "T-1", boat))
            final = mp.get((race_id, "final", boat))
            if t3 is not None and t1 is not None and final is not None:
                valid_triples.append({
                    "race_id": race_id,
                    "boat": boat,
                    "date": dates[race_id],
                    "mp_t3": t3,
                    "mp_t1": t1,
                    "mp_final": final,
                })

    print(f"Valid triples (T-3 + T-1 + final): {len(valid_triples)}")
    print(f"Races: {len(set(v['race_id'] for v in valid_triples))}")
    print(f"Dates: {sorted(set(v['date'] for v in valid_triples))}")

    if not valid_triples:
        print("No data to analyze")
        sys.exit(0)

    # --- Market probability drift ---
    t3_mps = np.array([v["mp_t3"] for v in valid_triples])
    t1_mps = np.array([v["mp_t1"] for v in valid_triples])
    final_mps = np.array([v["mp_final"] for v in valid_triples])

    t3_shift = np.abs(final_mps - t3_mps)
    t1_shift = np.abs(final_mps - t1_mps)

    print(f"\n--- Market Probability Drift ---")
    print(f"T-3 vs final: mean shift {t3_shift.mean()*100:.1f}%, median {np.median(t3_shift)*100:.1f}%")
    print(f"T-1 vs final: mean shift {t1_shift.mean()*100:.1f}%, median {np.median(t1_shift)*100:.1f}%")

    # --- EV shift ---
    # EV = model_prob / market_prob * 0.75 - 1
    # Using mp as proxy for "what the model would see"
    # EV shift = (1/mp_final - 1/mp_t3) * model_prob * 0.75
    # Simplified: relative change in 1/mp
    ev_shift_t3 = (1/final_mps - 1/t3_mps) / (1/t3_mps) * 100  # percentage points
    ev_shift_t1 = (1/final_mps - 1/t1_mps) / (1/t1_mps) * 100

    print(f"\nEV shift (confirmed vs T-3): mean {ev_shift_t3.mean():+.1f}pt, median {np.median(ev_shift_t3):+.1f}pt")
    print(f"EV shift (confirmed vs T-1): mean {ev_shift_t1.mean():+.1f}pt, median {np.median(ev_shift_t1):+.1f}pt")

    # --- T-3 + T-1 extrapolation model ---
    # Predict final_mp from T-3 and T-1
    X = np.column_stack([t3_mps, t1_mps])
    y = final_mps

    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    residual = y - pred
    r2 = reg.score(X, y)

    print(f"\n--- T-3 + T-1 Extrapolation Model ---")
    print(f"final_mp = {reg.coef_[0]:.3f}*T3 + {reg.coef_[1]:.3f}*T1 + {reg.intercept_:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Residual: mean {residual.mean()*100:.2f}%, std {residual.std()*100:.2f}%")

    # --- Threshold accuracy for various EV thresholds ---
    print(f"\n--- EV Threshold Accuracy ---")
    print(f"{'Threshold':>10} | {'T-3 only':>10} | {'T-3+T-1':>10} | {'T-3 miss':>10} | {'Extrap miss':>10} | {'Extrap FP':>10}")

    for thr in [0.20, 0.30, 0.36, 0.40, 0.48]:
        # Filter to non-favorite boats (mp_t3 in reasonable range)
        boundary = [v for v in valid_triples if 0.01 < v["mp_t3"] < 0.50]
        if not boundary:
            continue

        # For each boat, simulate model_prob values that place EV_final near
        # the threshold (delta offsets around thr). Check whether T-3 alone
        # and T-3+T-1 extrapolation agree with the confirmed-odds decision.
        correct_t3 = 0
        correct_extrap = 0
        miss_t3 = 0
        miss_extrap = 0
        fp_extrap = 0
        total_near = 0

        for v in boundary:
            mp_pred = reg.coef_[0] * v["mp_t3"] + reg.coef_[1] * v["mp_t1"] + reg.intercept_
            if mp_pred <= 0:
                continue

            for delta in [-0.05, 0.0, 0.05, 0.10, 0.20, 0.40]:
                ev_target = thr + delta
                p = (1 + ev_target) * v["mp_final"] / 0.75
                if p <= 0:
                    continue

                ev_t3_val = p / v["mp_t3"] * 0.75 - 1
                ev_final_val = p / v["mp_final"] * 0.75 - 1
                ev_extrap_val = p / mp_pred * 0.75 - 1

                buy_final = ev_final_val >= thr
                buy_t3 = ev_t3_val >= thr
                buy_extrap = ev_extrap_val >= thr

                total_near += 1
                if buy_t3 == buy_final:
                    correct_t3 += 1
                if buy_extrap == buy_final:
                    correct_extrap += 1
                if buy_final and not buy_t3:
                    miss_t3 += 1
                if buy_final and not buy_extrap:
                    miss_extrap += 1
                if not buy_final and buy_extrap:
                    fp_extrap += 1

        if total_near > 0:
            acc_t3 = correct_t3 / total_near * 100
            acc_ext = correct_extrap / total_near * 100
            print(f"  ev>{thr:+.0%}   | {acc_t3:>8.1f}%  | {acc_ext:>8.1f}%  | {miss_t3:>10} | {miss_extrap:>10} | {fp_extrap:>10}")

    # --- Per-date breakdown ---
    print(f"\n--- Per-Date Drift ---")
    for d in sorted(set(v["date"] for v in valid_triples)):
        day_data = [v for v in valid_triples if v["date"] == d]
        day_t3_shift = np.mean([abs(v["mp_final"] - v["mp_t3"]) for v in day_data])
        day_t1_shift = np.mean([abs(v["mp_final"] - v["mp_t1"]) for v in day_data])
        n_races = len(set(v["race_id"] for v in day_data))
        print(f"  {d}: {n_races:>3}R | T-3 shift {day_t3_shift*100:.1f}% | T-1 shift {day_t1_shift*100:.1f}%")

    print(f"\nTotal time: done")


if __name__ == "__main__":
    main()
