"""Evaluation metrics for boat race prediction.

Includes ranking metrics and actual payout-based recovery rate.
Primary metric: 2連単 (exacta) recovery rate.
"""

import numpy as np
import pandas as pd

from .db import get_connection
from lightgbm import LGBMRanker


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

    results = {
        "topNAccuracy": _top_n_accuracy(df),
        "avgNDCG": _average_ndcg(df),
        "multiHitRates": _multi_bet_hit_rates(df),
    }

    if db_path:
        payout_roi = _payout_recovery(df, db_path)
        if payout_roi:
            results["payoutROI"] = payout_roi

        confidence_analysis = _confidence_analysis(df, db_path)
        if confidence_analysis:
            results["confidenceAnalysis"] = confidence_analysis

    return results


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def _top_n_accuracy(df: pd.DataFrame) -> dict[str, float]:
    """For each race, check if actual winner is in model's top-N predictions."""
    counts = {1: 0, 2: 0, 3: 0}
    total = 0
    for _race_id, group in df.groupby("race_id"):
        group = group.sort_values("score", ascending=False)
        top3_pos = group.head(3)["actual_pos"].values
        total += 1
        for n in [1, 2, 3]:
            if 1 in top3_pos[:n]:
                counts[n] += 1
    return {str(n): counts[n] / total if total > 0 else 0.0 for n in [1, 2, 3]}


def _multi_bet_hit_rates(df: pd.DataFrame) -> dict[str, float]:
    """Compute hit rates for multi-boat bet types (index-based)."""
    exacta_hit = quinella_hit = trifecta_hit = trio_hit = total = 0

    for _race_id, group in df.groupby("race_id"):
        pred = group.sort_values("score", ascending=False)
        actual = group.sort_values("actual_pos")

        pred_nums = pred["boat_number"].values
        actual_nums = actual["boat_number"].values

        total += 1

        # 2連単: top2 in exact order
        if list(pred_nums[:2]) == list(actual_nums[:2]):
            exacta_hit += 1
        # 2連複: top2 regardless of order
        if set(pred_nums[:2]) == set(actual_nums[:2]):
            quinella_hit += 1
        # 3連単: top3 in exact order
        if list(pred_nums[:3]) == list(actual_nums[:3]):
            trifecta_hit += 1
        # 3連複: top3 regardless of order
        if set(pred_nums[:3]) == set(actual_nums[:3]):
            trio_hit += 1

    n = total or 1
    return {
        "2連単": exacta_hit / n,
        "2連複": quinella_hit / n,
        "3連単": trifecta_hit / n,
        "3連複": trio_hit / n,
    }


def _ndcg_at_k(relevance: np.ndarray, k: int = 6) -> float:
    """Compute nDCG@k for a single query (race)."""
    relevance = relevance[:k]
    if len(relevance) == 0:
        return 0.0
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    ideal = np.sort(relevance)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _average_ndcg(df: pd.DataFrame, k: int = 5) -> float:
    """Average nDCG@k across all races."""
    ndcgs = []
    for _race_id, group in df.groupby("race_id"):
        group = group.sort_values("score", ascending=False)
        field_size = len(group)
        relevance = np.maximum(0, field_size - group["actual_pos"].values + 1)
        ndcgs.append(_ndcg_at_k(relevance, k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ---------------------------------------------------------------------------
# Payout-based recovery rate
# ---------------------------------------------------------------------------

BET_UNIT = 100

CONFIDENCE_THRESHOLDS: list[tuple[float, int]] = [
    (1.0, 5),
    (0.7, 4),
    (0.45, 3),
    (0.25, 2),
]


def _load_payouts(
    db_path: str, race_ids: list[int],
) -> dict[int, dict[str, dict[str, int]]]:
    """Load payouts from DB: {race_id: {bet_type: {combination: payout}}}"""
    if not race_ids:
        return {}
    conn = get_connection(db_path)
    placeholders = ",".join("?" * len(race_ids))
    rows = conn.execute(
        f"SELECT race_id, bet_type, combination, payout "
        f"FROM db.race_payouts WHERE race_id IN ({placeholders})",
        race_ids,
    ).fetchall()
    conn.close()

    result: dict[int, dict[str, dict[str, int]]] = {}
    for race_id, bet_type, combination, payout in rows:
        result.setdefault(race_id, {}).setdefault(bet_type, {})[combination] = payout
    return result


def _combo(nums: list[int], ordered: bool = False) -> str:
    """Build combination string matching DB format."""
    if ordered:
        return "-".join(str(n) for n in nums)
    return "-".join(str(n) for n in sorted(nums))


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
# Betting strategies (boat racing specific)
# ---------------------------------------------------------------------------


def _tansho_bets(nums: list[int]) -> list[str]:
    """単勝: top1 (1点)"""
    return [str(nums[0])]


def _fukusho_bets(nums: list[int]) -> list[str]:
    """複勝: top1 (1点) — 2着以内に入れば的中"""
    return [str(nums[0])]


def _exacta_bets(nums: list[int]) -> list[str]:
    """2連単: top1→1着, top2,3→2着 (2点)"""
    if len(nums) < 2:
        return []
    return [_combo([nums[0], s], ordered=True) for s in nums[1:3]]


def _quinella_bets(nums: list[int]) -> list[str]:
    """2連複: top1-top2 (1点)"""
    if len(nums) < 2:
        return []
    return [_combo(nums[:2])]


def _trifecta_bets(nums: list[int]) -> list[str]:
    """3連単: ◎固定1着, 2着○▲, 3着○▲△(2着除く) (最大6点)"""
    if len(nums) < 3:
        return []
    first = nums[0]
    second = nums[1:3]    # ○▲
    third = nums[1:4]     # ○▲△
    bets = []
    for s in second:
        for t in third:
            if s != t:
                bets.append(_combo([first, s, t], ordered=True))
    return bets


def _trio_bets(nums: list[int]) -> list[str]:
    """3連複: top1,2軸 → 3頭目をtop3,4 (最大2点)"""
    if len(nums) < 3:
        return []
    return [_combo([nums[0], nums[1], t]) for t in nums[2:4]]


# Strategy registry: (display_name, db_key, bet_generator)
# All strategies are evaluated for metrics, but only BETTING_STRATEGIES are
# used for actual betting (2連単 + 3連単).
_STRATEGIES: list[tuple[str, str, callable]] = [
    ("単勝", "単勝", _tansho_bets),
    ("複勝", "複勝", _fukusho_bets),
    ("2連単", "2連単", _exacta_bets),
    ("2連複", "2連複", _quinella_bets),
    ("3連単", "3連単", _trifecta_bets),
    ("3連複", "3連複", _trio_bets),
]

# Primary evaluation metric
PRIMARY_STRATEGY = "2連単"

# Actual betting targets (2連単: main, 3連単: sub)
BETTING_STRATEGIES = ["2連単", "3連単"]


def _simulate_bets(
    df: pd.DataFrame,
    payouts_db: dict[int, dict[str, dict[str, int]]],
    db_key: str,
    bet_fn: callable,
) -> tuple[int, int, int]:
    """Run a betting strategy across all races, return (total_bet, total_payout, hits)."""
    total_bet = total_payout = hits = 0
    for race_id, group in df.groupby("race_id"):
        rp = payouts_db.get(race_id, {}).get(db_key, {})
        if not rp:
            continue
        pred = group.sort_values("score", ascending=False)
        nums = [int(pred.iloc[i]["boat_number"]) for i in range(min(5, len(pred)))]
        combos = bet_fn(nums)
        for c in combos:
            total_bet += BET_UNIT
            p = rp.get(c)
            if p:
                total_payout += p
                hits += 1
    return total_bet, total_payout, hits


def _compute_race_confidence(group: pd.DataFrame) -> float:
    """Confidence = gap between 1st and 2nd model scores."""
    scores = group.sort_values("score", ascending=False)["score"].values
    if len(scores) < 2:
        return 0.0
    return float(scores[0] - scores[1])


def _payout_recovery(df: pd.DataFrame, db_path: str) -> dict[str, dict[str, float]]:
    """Compute actual payout-based recovery rate for all bet types."""
    race_ids = sorted(df["race_id"].unique().tolist())
    payouts_db = _load_payouts(db_path, race_ids)
    if not payouts_db:
        return {}

    has_payouts = set(payouts_db.keys())
    df_filtered = df[df["race_id"].isin(has_payouts)]

    stats: dict[str, dict[str, float]] = {}
    for name, db_key, bet_fn in _STRATEGIES:
        b, p, h = _simulate_bets(df_filtered, payouts_db, db_key, bet_fn)
        s = _bet_stats(b, p, h)
        if s:
            stats[name] = s

    return stats


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------


def _confidence_analysis(df: pd.DataFrame, db_path: str) -> list[dict]:
    """Analyze recovery rate at various confidence thresholds."""
    race_ids = sorted(df["race_id"].unique().tolist())
    payouts_db = _load_payouts(db_path, race_ids)
    if not payouts_db:
        return []

    races_with_payouts = set(payouts_db.keys())

    race_conf: dict[int, float] = {}
    race_bets: dict[int, dict[str, tuple[int, int, int]]] = {}

    for race_id, group in df.groupby("race_id"):
        if race_id not in races_with_payouts:
            continue
        race_conf[race_id] = _compute_race_confidence(group)

        pred = group.sort_values("score", ascending=False)
        nums = [int(pred.iloc[i]["boat_number"]) for i in range(min(5, len(pred)))]
        race_bets[race_id] = {}
        for name, db_key, bet_fn in _STRATEGIES:
            rp = payouts_db.get(race_id, {}).get(db_key, {})
            if not rp:
                continue
            combos = bet_fn(nums)
            bet = len(combos) * BET_UNIT
            payout = hits = 0
            for c in combos:
                p = rp.get(c)
                if p:
                    payout += p
                    hits += 1
            race_bets[race_id][name] = (bet, payout, hits)

    if not race_conf:
        return []

    score_diffs = np.array(list(race_conf.values()))

    results = []
    for pct in [0, 25, 50, 66, 75]:
        threshold = float(np.percentile(score_diffs, pct)) if pct > 0 else 0.0
        selected = {rid for rid, sd in race_conf.items() if sd >= threshold}
        if not selected:
            continue

        for name, _, _ in _STRATEGIES:
            total_b = total_p = total_h = 0
            for rid in selected:
                rb = race_bets.get(rid, {}).get(name)
                if rb:
                    total_b += rb[0]
                    total_p += rb[1]
                    total_h += rb[2]
            s = _bet_stats(total_b, total_p, total_h)
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
