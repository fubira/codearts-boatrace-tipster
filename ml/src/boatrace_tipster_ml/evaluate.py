"""Evaluation metrics for boat race prediction.

Includes ranking metrics and actual payout-based recovery rate.
Primary metric: 2連単 (exacta) recovery rate.

All computations are vectorized using numpy reshape to (n_races, 6).
No Python per-race loops.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from .db import get_connection
from .feature_config import neutralize_leaked_features

# Permutation importance default settings
PERM_N_REPEATS = 5

FIELD_SIZE = 6
BET_UNIT = 100


def evaluate_model(
    model: LGBMRanker,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    db_path: str | None = None,
    payouts_cache: dict | None = None,
    skip_confidence: bool = False,
) -> dict:
    """Evaluate model on a dataset, computing per-race metrics.

    Automatically neutralizes leaked features (gate_bias, upset_rate) so
    the model cannot use intraday leakage for within-race discrimination.

    Args:
        payouts_cache: Pre-loaded payouts dict to avoid DB queries.
        skip_confidence: Skip confidence analysis (faster for HPO).
    """
    X = neutralize_leaked_features(X, meta)
    scores = model.predict(X)

    df = meta[["race_id", "boat_number"]].copy()
    df["score"] = scores
    df["actual_pos"] = y.values

    # Filter to races with a valid winner (exclude all-DNF races)
    actual_pos = df["actual_pos"].values.reshape(-1, FIELD_SIZE)
    has_winner = (actual_pos == 1).any(axis=1)
    keep_mask = np.repeat(has_winner, FIELD_SIZE)
    df = df[keep_mask].reset_index(drop=True)

    # Pre-compute predicted ranking per race (vectorized, assumes 6 entries/race)
    pred_ranks, actual_ranks = _compute_rankings(df)

    results = {
        "topNAccuracy": _top_n_accuracy(pred_ranks, actual_ranks),
        "avgNDCG": _average_ndcg(pred_ranks, actual_ranks),
        "multiHitRates": _multi_bet_hit_rates(pred_ranks, actual_ranks),
    }

    if db_path or payouts_cache is not None:
        race_ids = df["race_id"].unique()
        boat_by_pred = _boat_numbers_by_pred_rank(df)

        payouts_db = payouts_cache if payouts_cache is not None else load_payouts(db_path, race_ids)

        all_stats = _simulate_all_bets(race_ids, boat_by_pred, payouts_db)
        payout_roi: dict[str, dict[str, float]] = {}
        for name, (b, p, h) in all_stats.items():
            s = _bet_stats(b, p, h)
            if s:
                payout_roi[name] = s
        if payout_roi:
            results["payoutROI"] = payout_roi

        if not skip_confidence:
            confidence = _race_confidence(df)
            confidence_analysis = _confidence_analysis_with_payouts(
                race_ids, boat_by_pred, confidence, payouts_db
            )
            if confidence_analysis:
                results["confidenceAnalysis"] = confidence_analysis

    return results


# ---------------------------------------------------------------------------
# Vectorized ranking computation
# ---------------------------------------------------------------------------


def _compute_rankings(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute predicted and actual rankings per race.

    Returns:
        pred_ranks: (n_races, 6) — predicted rank (1-6) for each boat position
        actual_ranks: (n_races, 6) — actual rank (1-6) for each boat position
    """
    n_races = len(df) // FIELD_SIZE
    scores = df["score"].values.reshape(n_races, FIELD_SIZE)
    actual = df["actual_pos"].values.reshape(n_races, FIELD_SIZE)

    # Predicted rank: argsort descending scores, then rank
    pred_order = np.argsort(-scores, axis=1)
    pred_ranks = np.empty_like(pred_order)
    rows = np.arange(n_races)[:, None]
    pred_ranks[rows, pred_order] = np.arange(1, FIELD_SIZE + 1)

    return pred_ranks, actual.astype(int)


def _boat_numbers_by_pred_rank(df: pd.DataFrame) -> np.ndarray:
    """Get boat numbers sorted by predicted rank per race.

    Returns: (n_races, 6) where [i, 0] is the predicted 1st boat number.
    """
    n_races = len(df) // FIELD_SIZE
    scores = df["score"].values.reshape(n_races, FIELD_SIZE)
    boats = df["boat_number"].values.reshape(n_races, FIELD_SIZE)

    pred_order = np.argsort(-scores, axis=1)
    return np.take_along_axis(boats, pred_order, axis=1).astype(int)


def _race_confidence(df: pd.DataFrame) -> np.ndarray:
    """Confidence = gap between 1st and 2nd predicted scores."""
    n_races = len(df) // FIELD_SIZE
    scores = df["score"].values.reshape(n_races, FIELD_SIZE)
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    return sorted_scores[:, 0] - sorted_scores[:, 1]


# ---------------------------------------------------------------------------
# Ranking metrics (vectorized)
# ---------------------------------------------------------------------------


def _top_n_accuracy(
    pred_ranks: np.ndarray, actual_ranks: np.ndarray
) -> dict[str, float]:
    """For each race, check if actual winner is in model's top-N predictions."""
    # Find which boat position has actual_pos == 1
    winner_mask = actual_ranks == 1  # (n_races, 6)
    # Get the predicted rank of the actual winner
    winner_pred_rank = (pred_ranks * winner_mask).sum(axis=1)  # (n_races,)

    n = len(winner_pred_rank)
    return {
        str(k): float((winner_pred_rank <= k).sum() / n) for k in [1, 2, 3]
    }


def _multi_bet_hit_rates(
    pred_ranks: np.ndarray, actual_ranks: np.ndarray
) -> dict[str, float]:
    """Compute hit rates for multi-boat bet types."""
    n = pred_ranks.shape[0]

    # Get boat indices (column positions) sorted by predicted/actual rank
    pred_top2 = np.argsort(pred_ranks, axis=1)[:, :2]
    pred_top3 = np.argsort(pred_ranks, axis=1)[:, :3]
    actual_top2 = np.argsort(actual_ranks, axis=1)[:, :2]
    actual_top3 = np.argsort(actual_ranks, axis=1)[:, :3]

    # 2連単: exact order of top2
    exacta = np.all(pred_top2 == actual_top2, axis=1).sum()
    # 3連単: exact order of top3
    trifecta = np.all(pred_top3 == actual_top3, axis=1).sum()

    return {
        "2連単": exacta / n,
        "3連単": trifecta / n,
    }


def _ndcg(pred_ranks: np.ndarray, actual_ranks: np.ndarray, k: int = 3) -> float:
    """Average nDCG@k across all races (vectorized)."""
    n_races = pred_ranks.shape[0]
    relevance = FIELD_SIZE + 1 - actual_ranks  # 1st→6, 6th→1

    # Sort relevance by predicted rank
    pred_order = np.argsort(pred_ranks, axis=1)
    sorted_rel = np.take_along_axis(relevance, pred_order, axis=1)[:, :k]

    # DCG
    discounts = np.log2(np.arange(2, k + 2))
    dcg = (sorted_rel / discounts).sum(axis=1)

    # Ideal DCG
    ideal_rel = np.sort(relevance, axis=1)[:, ::-1][:, :k]
    idcg = (ideal_rel / discounts).sum(axis=1)

    ndcg = np.where(idcg > 0, dcg / idcg, 0.0)
    return float(ndcg.mean())


def _average_ndcg(
    pred_ranks: np.ndarray, actual_ranks: np.ndarray
) -> float:
    return _ndcg(pred_ranks, actual_ranks, k=3)


# ---------------------------------------------------------------------------
# Payout-based recovery rate
# ---------------------------------------------------------------------------


def load_payouts(
    db_path: str,
    race_ids: np.ndarray,
) -> dict[int, dict[str, dict[str, int]]]:
    """Load payouts from DB: {race_id: {bet_type: {combination: payout}}}"""
    if len(race_ids) == 0:
        return {}
    conn = get_connection(db_path)
    id_list = ",".join(str(int(r)) for r in race_ids)
    rows = conn.execute(
        f"SELECT race_id, bet_type, combination, payout "
        f"FROM db.race_payouts WHERE race_id IN ({id_list})",
    ).fetchall()
    conn.close()

    result: dict[int, dict[str, dict[str, int]]] = {}
    for race_id, bet_type, combination, payout in rows:
        result.setdefault(race_id, {}).setdefault(bet_type, {})[combination] = payout
    return result


def _combo(nums: list | np.ndarray, ordered: bool = False) -> str:
    """Build combination string matching DB format.

    DB format: ordered uses '-' (e.g., 1-2), unordered uses '=' (e.g., 1=2).
    """
    if ordered:
        return "-".join(str(int(n)) for n in nums)
    return "=".join(str(int(n)) for n in sorted(nums))


# Betting strategy generators
def _exacta_bets(nums: np.ndarray) -> list[str]:
    return [_combo([nums[0], s], ordered=True) for s in nums[1:3]]


def _trifecta_bets(nums: np.ndarray) -> list[str]:
    first, second, third = int(nums[0]), nums[1:3], nums[1:4]
    bets = []
    for s in second:
        for t in third:
            if int(s) != int(t):
                bets.append(_combo([first, s, t], ordered=True))
    return bets


_STRATEGIES: list[tuple[str, str, callable]] = [
    ("2連単", "2連単", _exacta_bets),
    ("3連単", "3連単", _trifecta_bets),
]


def _simulate_all_bets(
    race_ids: np.ndarray,
    boat_by_pred: np.ndarray,
    payouts_db: dict[int, dict[str, dict[str, int]]],
) -> dict[str, tuple[int, int, int]]:
    """Simulate all betting strategies. Returns {name: (bet, payout, hits)}."""
    stats: dict[str, list[int]] = {name: [0, 0, 0] for name, _, _ in _STRATEGIES}

    for i, race_id in enumerate(race_ids):
        rid = int(race_id)
        rp = payouts_db.get(rid)
        if not rp:
            continue
        nums = boat_by_pred[i]
        for name, db_key, bet_fn in _STRATEGIES:
            type_payouts = rp.get(db_key)
            if not type_payouts:
                continue
            combos = bet_fn(nums)
            for c in combos:
                stats[name][0] += BET_UNIT
                p = type_payouts.get(c)
                if p:
                    stats[name][1] += p
                    stats[name][2] += 1

    return {name: tuple(v) for name, v in stats.items()}


def _bet_stats(bet: int, payout: int, hits: int) -> dict[str, float] | None:
    if bet == 0:
        return None
    n = bet // BET_UNIT
    return {
        "recoveryRate": payout / bet,
        "hitRate": hits / n,
        "avgPayout": payout / hits if hits > 0 else 0,
        "betCount": n,
    }


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------


def _confidence_analysis_with_payouts(
    race_ids: np.ndarray,
    boat_by_pred: np.ndarray,
    confidence: np.ndarray,
    payouts_db: dict,
) -> list[dict]:
    """Analyze recovery rate at various confidence thresholds."""
    if not payouts_db:
        return []

    results = []
    for pct in [0, 25, 50, 66, 75]:
        threshold = float(np.percentile(confidence, pct)) if pct > 0 else 0.0
        mask = confidence >= threshold
        selected_ids = race_ids[mask]
        selected_boats = boat_by_pred[mask]

        all_stats = _simulate_all_bets(selected_ids, selected_boats, payouts_db)

        for name, (b, p, h) in all_stats.items():
            s = _bet_stats(b, p, h)
            if s:
                results.append({
                    "percentile": pct,
                    "threshold": round(threshold, 4),
                    "betType": name,
                    "recoveryRate": s["recoveryRate"],
                    "hitRate": s["hitRate"],
                    "betCount": s["betCount"],
                })

    return results


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance(
    model: LGBMRanker,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    db_path: str | None = None,
    n_repeats: int = PERM_N_REPEATS,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute permutation importance for all features.

    Measures the drop in nDCG and 2連単 ROI when each feature is shuffled.
    Shuffles across all entries (breaking within-race signal).

    Returns: {feature_name: {ndcg_drop, roi_drop, ndcg_drop_std, roi_drop_std}}
    """
    rng = np.random.default_rng(seed)

    # Baseline scores
    baseline = evaluate_model(model, X, y, meta, db_path=db_path)
    base_ndcg = baseline["avgNDCG"]
    base_roi = baseline.get("payoutROI", {}).get("2連単", {}).get("recoveryRate", 0.0)

    results: dict[str, dict[str, float]] = {}

    for col in X.columns:
        ndcg_drops = []
        roi_drops = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)

            perm_result = evaluate_model(model, X_perm, y, meta, db_path=db_path)
            perm_ndcg = perm_result["avgNDCG"]
            perm_roi = perm_result.get("payoutROI", {}).get("2連単", {}).get("recoveryRate", 0.0)

            ndcg_drops.append(base_ndcg - perm_ndcg)
            roi_drops.append(base_roi - perm_roi)

        results[col] = {
            "ndcg_drop": float(np.mean(ndcg_drops)),
            "ndcg_drop_std": float(np.std(ndcg_drops)),
            "roi_drop": float(np.mean(roi_drops)),
            "roi_drop_std": float(np.std(roi_drops)),
        }

    return results


# ---------------------------------------------------------------------------
# Trifecta X-allflow strategy evaluation
# ---------------------------------------------------------------------------

TRIFECTA_TICKETS_PER_BET = 20


def evaluate_trifecta_strategy(
    b1_probs: np.ndarray,
    meta_b1: pd.DataFrame,
    rank_scores: np.ndarray,
    meta_rank: pd.DataFrame,
    finish_map: dict[tuple[int, int], int],
    trifecta_odds: dict[tuple[int, str], float],
    tri_win_prob: dict[tuple[int, int], float],
    b1_threshold: float,
    ev_threshold: float,
    *,
    r2_ev_threshold: float | None = None,
    race_date_map: dict[int, str] | None = None,
    exacta_odds: dict[tuple[int, str], float] | None = None,
    per_race: bool = False,
) -> dict | list[dict]:
    """Evaluate X-allflow trifecta strategy on test data.

    Used by tune, backtest, and MC simulation.

    Args:
        ev_threshold: Rank-1 EV threshold. Bet on rank-1 pick if EV >= this.
        r2_ev_threshold: Rank-2 EV threshold. When rank-1 EV is below
            ev_threshold, try rank-2 pick if its EV >= this value.
            None disables rank-2 fallback (default, backward compatible).
        per_race: If True, return list of per-race dicts (for MC/backtest daily).
                  If False, return summary dict with races/cost/wins/payout/roi.
    """
    field_size = 6

    n_races = len(rank_scores) // field_size
    scores_2d = rank_scores.reshape(n_races, field_size)
    boats_2d = meta_rank["boat_number"].values.reshape(n_races, field_size)
    race_ids = meta_rank["race_id"].values.reshape(n_races, field_size)[:, 0]

    pred_order = np.argsort(-scores_2d, axis=1)
    top_boats = np.take_along_axis(boats_2d, pred_order, axis=1)

    exp_s = np.exp(scores_2d - scores_2d.max(axis=1, keepdims=True))
    rank_probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    b1_map = {rid: i for i, rid in enumerate(meta_b1["race_id"].values)}

    total_races = 0
    total_cost = 0.0
    total_wins = 0
    total_payout = 0.0
    r2_races = 0
    r2_wins = 0
    results: list[dict] = []

    for ri in range(n_races):
        rid = int(race_ids[ri])
        bi = b1_map.get(rid)
        if bi is None:
            continue
        b1p = float(b1_probs[bi])
        if b1p >= b1_threshold:
            continue

        # --- Rank-1 pick: top non-boat-1 ---
        wp1 = int(top_boats[ri, 0])
        if wp1 == 1:
            wp1 = int(top_boats[ri, 1])

        bidx1 = np.where(boats_2d[ri] == wp1)[0]
        if len(bidx1) == 0:
            continue
        wp1_prob = float(rank_probs[ri, bidx1[0]])

        mkt_prob1 = tri_win_prob.get((rid, wp1), 0)
        if mkt_prob1 <= 0:
            continue
        ev1 = wp1_prob / mkt_prob1 * 0.75 - 1

        # --- Decide: rank-1, rank-2 fallback, or skip ---
        if ev1 >= ev_threshold:
            wp, ev, wprob, rank_used = wp1, ev1, wp1_prob, 1
        elif r2_ev_threshold is not None:
            # Rank-2: 2nd non-boat-1
            non_b1_top = [int(top_boats[ri, k]) for k in range(field_size)
                          if int(top_boats[ri, k]) != 1]
            if len(non_b1_top) < 2:
                continue
            wp2 = non_b1_top[1]

            bidx2 = np.where(boats_2d[ri] == wp2)[0]
            if len(bidx2) == 0:
                continue
            wp2_prob = float(rank_probs[ri, bidx2[0]])

            mkt_prob2 = tri_win_prob.get((rid, wp2), 0)
            if mkt_prob2 <= 0:
                continue
            ev2 = wp2_prob / mkt_prob2 * 0.75 - 1

            if ev2 >= r2_ev_threshold:
                wp, ev, wprob, rank_used = wp2, ev2, wp2_prob, 2
            else:
                continue
        else:
            continue

        total_races += 1
        total_cost += TRIFECTA_TICKETS_PER_BET
        if rank_used == 2:
            r2_races += 1

        pick_1st = finish_map.get((rid, wp)) == 1
        allflow_odds = 0.0
        exacta_hit_odds = 0.0

        if pick_1st:
            a2 = a3 = None
            for b in range(1, 7):
                fp = finish_map.get((rid, b))
                if fp == 2:
                    a2 = b
                if fp == 3:
                    a3 = b

            if a2 and a3:
                hc = f"{wp}-{a2}-{a3}"
                ho = trifecta_odds.get((rid, hc))
                if ho and ho > 0:
                    allflow_odds = ho
                    total_wins += 1
                    total_payout += ho
                    if rank_used == 2:
                        r2_wins += 1
                if exacta_odds is not None:
                    ec = f"{wp}-{a2}"
                    eo = exacta_odds.get((rid, ec))
                    if eo and eo > 0:
                        exacta_hit_odds = eo

        if per_race:
            results.append({
                "race_id": rid,
                "date": race_date_map.get(rid, "") if race_date_map else "",
                "winner_pick": wp,
                "b1_prob": round(b1p, 3),
                "winner_prob": round(wprob, 3),
                "ev": round(ev, 3),
                "pick_1st": pick_1st,
                "allflow_odds": round(allflow_odds, 1),
                "exacta_hit_odds": round(exacta_hit_odds, 1),
                "rank_used": rank_used,
            })

    if per_race:
        return results

    roi = total_payout / total_cost if total_cost > 0 else 0
    return {
        "races": total_races,
        "cost": total_cost,
        "wins": total_wins,
        "payout": total_payout,
        "roi": roi,
        "r2_races": r2_races,
        "r2_wins": r2_wins,
    }
