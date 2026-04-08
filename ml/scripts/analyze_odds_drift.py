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
        # For each (race, boat), check if EV > threshold at confirmed odds
        # Use a "model_prob" proxy: just use final_mp itself as a reference
        # What we really want: given a fixed model_prob, does the market prob change
        # enough to flip the EV decision?
        #
        # EV_timing = model_prob / mp_timing * 0.75 - 1
        # Since model_prob is the same, EV_timing > thr iff mp_timing < model_prob * 0.75 / (1 + thr)
        # Equivalently, we compare mp_t3 vs mp_final: if mp drops, EV rises.
        #
        # Simpler: use 1/mp as proxy for EV direction
        # EV ∝ 1/mp, so threshold on EV maps to threshold on mp
        # At EV boundary: model_prob / mp * 0.75 = 1 + thr
        # We want races where EV is near threshold at T-3

        # Focus on EV boundary zone: T-3 EV in [-10%, +80%]
        # Using 1/mp ratio: ev_t3 ≈ C/mp_t3 for some constant C
        # We approximate by looking at how mp changes affect threshold crossing

        # Direct approach: for each boat, compute
        # "would a model with prob = mp_final * (1+thr) / 0.75 cross the threshold?"
        # That's circular. Instead, compute the fraction of cases where
        # sign(EV_t3 - thr) != sign(EV_final - thr)

        # Use relative EV: EV = 1/mp * constant - 1
        # threshold crossing flips when 1/mp crosses a boundary
        # T-3: ev_t3 = k/mp_t3 - 1 vs thr → k = (1+thr)*mp_t3
        # Final: ev_final = k/mp_final - 1 = (1+thr)*mp_t3/mp_final - 1
        # ev_final > thr iff mp_t3/mp_final > 1 (i.e., mp dropped)

        # More precise: for points near threshold at T-3,
        # check if the extrapolated mp gives the right final decision

        # Filter to boundary zone: T-3 EV within [-10%, +80%] of threshold
        # Using mp_t3 as reference, boundary means mp_t3 in a range
        boundary = []
        for v in valid_triples:
            # Proxy EV at T-3: relative to threshold
            # ev_t3 ~ C/mp_t3, we want ev_t3 near thr
            # Filter: |mp_t3 - mp_final| / mp_t3 within reasonable range
            # AND mp_t3 in a range that could be near threshold
            # Since we don't have model_prob, use a simpler filter:
            # only look at boats where mp_t3 is not too extreme
            if 0.01 < v["mp_t3"] < 0.50:  # non-favorite boats
                boundary.append(v)

        if not boundary:
            continue

        # For boundary boats: check threshold crossing consistency
        # "Would the decision be the same at T-3 vs final?"
        # Using the ratio test: if mp moved by >threshold-relevant amount
        correct_t3 = 0
        correct_extrap = 0
        total_b = 0
        miss_t3 = 0  # final says buy, T-3 says skip
        miss_extrap = 0
        fp_extrap = 0  # extrap says buy, final says skip

        for v in boundary:
            # Ground truth: final mp
            # "Buy" at final if mp is low enough → EV high enough
            # Without model_prob, check if the ordering is preserved
            # Use: does T-3 correctly predict whether final mp went up or down?
            mp_pred_extrap = reg.coef_[0] * v["mp_t3"] + reg.coef_[1] * v["mp_t1"] + reg.intercept_

            # Check sign of (mp_t3 - mp_final): if positive, EV went up (good for buying)
            # T-3 decision: use mp_t3 (as-is)
            # Extrap decision: use mp_pred_extrap
            # Final truth: use mp_final

            # Threshold crossing: for a given model_prob p,
            # EV = p/mp * 0.75 - 1 > thr iff mp < p*0.75/(1+thr)
            # Decision agrees if mp_t3 and mp_final are on the same side of the cutoff
            # Since p is unknown, use the ratio: mp_t3/mp_final
            # If ratio > 1: T-3 overestimates mp → underestimates EV → might miss a buy
            # If ratio < 1: T-3 underestimates mp → overestimates EV → might false positive

            # Simpler metric: does the ordering match?
            # "buy at final" = mp_final < mp_t3 * (1 - margin)
            # But without model_prob this is hard to define precisely

            # Let's use a direct test: assume model_prob is such that
            # EV_final is exactly at threshold. Then check if T-3 and extrap agree.
            # model_prob = (1+thr) * mp_final / 0.75
            p = (1 + thr) * v["mp_final"] / 0.75
            ev_t3 = p / v["mp_t3"] * 0.75 - 1
            ev_final = p / v["mp_final"] * 0.75 - 1  # = thr by construction
            ev_extrap = p / mp_pred_extrap * 0.75 - 1 if mp_pred_extrap > 0 else -1

            # Now check: at slightly above/below threshold
            # For boats where final EV > thr (buy), check if T-3/extrap also says buy
            # For boats where final EV < thr (skip), check if T-3/extrap also says skip
            # Since ev_final = thr exactly, jitter by looking at actual nearby points

        # Better approach: use actual EV relative shifts
        # For each boat, assume some model_prob, compute EV at T-3, T-1, final
        # and check threshold crossing.
        # Use model_prob such that EV at final is in [-10%, +80%] range near threshold.
        miss_t3 = 0
        miss_extrap = 0
        fp_extrap = 0
        total_near = 0

        for v in boundary:
            mp_pred = reg.coef_[0] * v["mp_t3"] + reg.coef_[1] * v["mp_t1"] + reg.intercept_
            if mp_pred <= 0:
                continue

            # Test with model_prob that puts EV_final near threshold
            # p = (1+thr+delta) * mp_final / 0.75 for delta in [-0.1, +0.8]
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
