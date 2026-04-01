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

FIELD_SIZE = 6
BET_UNIT = 100


def evaluate_model(
    model: LGBMRanker,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    db_path: str | None = None,
) -> dict:
    """Evaluate model on a dataset, computing per-race metrics."""
    scores = model.predict(X)

    df = meta[["race_id", "boat_number"]].copy()
    df["score"] = scores
    df["actual_pos"] = y.values

    # Filter to complete races (6 entries) for vectorized computation
    race_counts = df.groupby("race_id").size()
    complete_races = race_counts[race_counts == FIELD_SIZE].index
    df = df[df["race_id"].isin(complete_races)].reset_index(drop=True)

    # Pre-compute predicted ranking per race (vectorized)
    pred_ranks, actual_ranks = _compute_rankings(df)

    results = {
        "topNAccuracy": _top_n_accuracy(pred_ranks, actual_ranks),
        "avgNDCG": _average_ndcg(pred_ranks, actual_ranks),
        "multiHitRates": _multi_bet_hit_rates(pred_ranks, actual_ranks),
    }

    if db_path:
        race_ids = df["race_id"].unique()
        boat_by_pred = _boat_numbers_by_pred_rank(df)

        payout_roi = _payout_recovery(race_ids, boat_by_pred, db_path)
        if payout_roi:
            results["payoutROI"] = payout_roi

        confidence = _race_confidence(df)
        confidence_analysis = _confidence_analysis(
            race_ids, boat_by_pred, confidence, db_path
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


def _load_payouts(
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
def _tansho_bets(nums: np.ndarray) -> list[str]:
    return [str(int(nums[0]))]


def _fukusho_bets(nums: np.ndarray) -> list[str]:
    return [str(int(nums[0]))]


def _exacta_bets(nums: np.ndarray) -> list[str]:
    return [_combo([nums[0], s], ordered=True) for s in nums[1:3]]


def _quinella_bets(nums: np.ndarray) -> list[str]:
    return [_combo(nums[:2])]


def _trifecta_bets(nums: np.ndarray) -> list[str]:
    first, second, third = int(nums[0]), nums[1:3], nums[1:4]
    bets = []
    for s in second:
        for t in third:
            if int(s) != int(t):
                bets.append(_combo([first, s, t], ordered=True))
    return bets


def _trio_bets(nums: np.ndarray) -> list[str]:
    return [_combo([nums[0], nums[1], t]) for t in nums[2:4]]


_STRATEGIES: list[tuple[str, str, callable]] = [
    ("単勝", "単勝", _tansho_bets),
    ("複勝", "複勝", _fukusho_bets),
    ("2連単", "2連単", _exacta_bets),
    ("3連単", "3連単", _trifecta_bets),
]

PRIMARY_STRATEGY = "2連単"
BETTING_STRATEGIES = ["2連単", "3連単"]


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


def _payout_recovery(
    race_ids: np.ndarray,
    boat_by_pred: np.ndarray,
    db_path: str,
) -> dict[str, dict[str, float]]:
    """Compute actual payout-based recovery rate for all bet types."""
    payouts_db = _load_payouts(db_path, race_ids)
    if not payouts_db:
        return {}

    all_stats = _simulate_all_bets(race_ids, boat_by_pred, payouts_db)

    result: dict[str, dict[str, float]] = {}
    for name, (b, p, h) in all_stats.items():
        s = _bet_stats(b, p, h)
        if s:
            result[name] = s
    return result


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------


def _confidence_analysis(
    race_ids: np.ndarray,
    boat_by_pred: np.ndarray,
    confidence: np.ndarray,
    db_path: str,
) -> list[dict]:
    """Analyze recovery rate at various confidence thresholds."""
    payouts_db = _load_payouts(db_path, race_ids)
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
