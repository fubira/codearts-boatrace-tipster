"""Tests for evaluation metrics.

Covers: combo format, betting strategies, ranking computation, accuracy metrics.
"""

import numpy as np
import pytest

from .evaluate import (
    _combo,
    _compute_rankings,
    _exacta_bets,
    _fukusho_bets,
    _multi_bet_hit_rates,
    _ndcg,
    _tansho_bets,
    _top_n_accuracy,
    _trifecta_bets,
)

import pandas as pd


# ---------------------------------------------------------------------------
# _combo: DB format matching
# ---------------------------------------------------------------------------


class TestCombo:
    def test_ordered_uses_hyphen(self):
        assert _combo([1, 2], ordered=True) == "1-2"
        assert _combo([3, 1], ordered=True) == "3-1"

    def test_unordered_uses_equals_and_sorts(self):
        assert _combo([1, 2]) == "1=2"
        assert _combo([3, 1]) == "1=3"

    def test_three_boat_ordered(self):
        assert _combo([2, 4, 5], ordered=True) == "2-4-5"

    def test_three_boat_unordered(self):
        assert _combo([5, 2, 4]) == "2=4=5"

    def test_numpy_input(self):
        assert _combo(np.array([1, 2]), ordered=True) == "1-2"
        assert _combo(np.array([3, 1])) == "1=3"

    def test_float_input_converts_to_int(self):
        assert _combo([1.0, 2.0], ordered=True) == "1-2"


# ---------------------------------------------------------------------------
# Betting strategies: combination generation
# ---------------------------------------------------------------------------


class TestBettingStrategies:
    def test_tansho_1_bet(self):
        nums = np.array([3, 1, 5, 2, 4, 6])
        bets = _tansho_bets(nums)
        assert bets == ["3"]

    def test_fukusho_1_bet(self):
        nums = np.array([3, 1, 5, 2, 4, 6])
        bets = _fukusho_bets(nums)
        assert bets == ["3"]

    def test_exacta_2_bets(self):
        """2連単: 1着固定, 2着に2nd/3rd候補 → 2点"""
        nums = np.array([3, 1, 5, 2, 4, 6])
        bets = _exacta_bets(nums)
        assert len(bets) == 2
        assert "3-1" in bets  # 3→1
        assert "3-5" in bets  # 3→5

    def test_exacta_ordered(self):
        """2連単は順序あり"""
        nums = np.array([1, 3, 5, 2, 4, 6])
        bets = _exacta_bets(nums)
        assert "1-3" in bets
        assert "3-1" not in bets

    def test_trifecta_bets(self):
        """3連単: 1着◎, 2着○▲, 3着○▲△ → 最大4点"""
        nums = np.array([1, 2, 3, 4, 5, 6])
        bets = _trifecta_bets(nums)
        # first=1, second=[2,3], third=[2,3,4]
        # 1-2-3, 1-2-4, 1-3-2, 1-3-4 → 4 bets
        assert len(bets) == 4
        assert "1-2-3" in bets
        assert "1-2-4" in bets
        assert "1-3-2" in bets
        assert "1-3-4" in bets
        # 1-2-2 should NOT exist (s != t check)
        assert "1-2-2" not in bets
        assert "1-3-3" not in bets

    def test_trifecta_no_duplicates(self):
        """3連単で同じ艇番が1着2着に来ない"""
        nums = np.array([5, 1, 3, 2, 4, 6])
        bets = _trifecta_bets(nums)
        for b in bets:
            parts = b.split("-")
            assert len(parts) == len(set(parts)), f"Duplicate in {b}"


# ---------------------------------------------------------------------------
# _compute_rankings: vectorized ranking
# ---------------------------------------------------------------------------


class TestComputeRankings:
    def _make_df(self, scores, actual_pos, boat_numbers=None):
        n = len(scores)
        if boat_numbers is None:
            boat_numbers = list(range(1, 7)) * (n // 6)
        return pd.DataFrame({
            "race_id": [1] * n,
            "boat_number": boat_numbers,
            "score": scores,
            "actual_pos": actual_pos,
        })

    def test_single_race(self):
        # Boat 3 has highest score, boat 1 has lowest
        df = self._make_df(
            scores=[1.0, 2.0, 5.0, 4.0, 3.0, 0.5],
            actual_pos=[3, 2, 1, 4, 5, 6],
        )
        pred_ranks, actual_ranks = _compute_rankings(df)

        assert pred_ranks.shape == (1, 6)
        # Boat 3 (idx 2) has highest score → pred rank 1
        assert pred_ranks[0, 2] == 1
        # Boat 6 (idx 5) has lowest score → pred rank 6
        assert pred_ranks[0, 5] == 6

        # Actual ranks should match input
        assert actual_ranks[0, 0] == 3  # boat 1
        assert actual_ranks[0, 2] == 1  # boat 3

    def test_two_races(self):
        df = pd.DataFrame({
            "race_id": [1]*6 + [2]*6,
            "boat_number": list(range(1, 7)) * 2,
            "score": [6, 5, 4, 3, 2, 1,  1, 2, 3, 4, 5, 6],
            "actual_pos": [1, 2, 3, 4, 5, 6,  6, 5, 4, 3, 2, 1],
        })
        pred_ranks, actual_ranks = _compute_rankings(df)
        assert pred_ranks.shape == (2, 6)
        # Race 1: boat 1 (idx 0) has highest score → rank 1
        assert pred_ranks[0, 0] == 1
        # Race 2: boat 6 (idx 5) has highest score → rank 1
        assert pred_ranks[1, 5] == 1


# ---------------------------------------------------------------------------
# _top_n_accuracy
# ---------------------------------------------------------------------------


class TestTopNAccuracy:
    def test_perfect_prediction(self):
        pred_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        acc = _top_n_accuracy(pred_ranks, actual_ranks)
        assert acc["1"] == 1.0
        assert acc["2"] == 1.0
        assert acc["3"] == 1.0

    def test_winner_predicted_2nd(self):
        # Actual winner is boat 2 (idx 1), predicted rank 2
        pred_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        actual_ranks = np.array([[2, 1, 3, 4, 5, 6]])
        acc = _top_n_accuracy(pred_ranks, actual_ranks)
        assert acc["1"] == 0.0  # not in top1
        assert acc["2"] == 1.0  # in top2
        assert acc["3"] == 1.0

    def test_winner_predicted_last(self):
        pred_ranks = np.array([[6, 5, 4, 3, 2, 1]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        acc = _top_n_accuracy(pred_ranks, actual_ranks)
        assert acc["1"] == 0.0
        assert acc["2"] == 0.0
        assert acc["3"] == 0.0

    def test_multiple_races(self):
        pred_ranks = np.array([
            [1, 2, 3, 4, 5, 6],  # perfect
            [6, 5, 4, 3, 2, 1],  # worst
        ])
        actual_ranks = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
        ])
        acc = _top_n_accuracy(pred_ranks, actual_ranks)
        assert acc["1"] == 0.5  # 1/2 correct


# ---------------------------------------------------------------------------
# _multi_bet_hit_rates
# ---------------------------------------------------------------------------


class TestMultiBetHitRates:
    def test_perfect_prediction(self):
        pred_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        rates = _multi_bet_hit_rates(pred_ranks, actual_ranks)
        assert rates["2連単"] == 1.0
        assert rates["3連単"] == 1.0

    def test_top2_swapped(self):
        """1st/2nd swapped → 2連単 miss"""
        pred_ranks = np.array([[2, 1, 3, 4, 5, 6]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        rates = _multi_bet_hit_rates(pred_ranks, actual_ranks)
        assert rates["2連単"] == 0.0

    def test_wrong_prediction(self):
        pred_ranks = np.array([[6, 5, 4, 3, 2, 1]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        rates = _multi_bet_hit_rates(pred_ranks, actual_ranks)
        assert rates["2連単"] == 0.0
        assert rates["3連単"] == 0.0


# ---------------------------------------------------------------------------
# _ndcg
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self):
        pred_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        assert _ndcg(pred_ranks, actual_ranks, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        pred_ranks = np.array([[6, 5, 4, 3, 2, 1]])
        actual_ranks = np.array([[1, 2, 3, 4, 5, 6]])
        ndcg_val = _ndcg(pred_ranks, actual_ranks, k=3)
        assert ndcg_val < 1.0
        assert ndcg_val > 0.0  # not zero because relevance is non-zero for all
