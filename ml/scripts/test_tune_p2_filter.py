"""Tests for evaluate_p2_strategy filter logic in tune_p2.py.

Focus: boundary-value testing for gap12_min_threshold and its interaction
with the pre-existing top-1, top3_conc, gap23 filters. Fixtures are
synthetic so each test is deterministic — scores are log(p) so that the
softmax inside evaluate_p2_strategy reproduces the intended probability
vector exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.tune_p2 import evaluate_p2_strategy

FIELD_SIZE = 6
RACE_ID = 100


def _make_single_race(
    probs: tuple[float, float, float, float, float, float],
    finish_order: tuple[int, int, int, int, int, int] = (1, 2, 3, 4, 5, 6),
):
    """Build synthetic (rank_scores, meta) reproducing a softmax probability vector.

    - probs: target softmax output for boats 1..6 (must sum to ~1.0)
    - finish_order: finish_position per boat row (1..6)

    softmax invariance under additive shift means scores = log(probs) reproduces
    the intended distribution exactly after evaluate_p2_strategy's softmax step.
    """
    assert abs(sum(probs) - 1.0) < 1e-6, f"probs must sum to 1, got {sum(probs)}"
    scores = np.log(np.array(probs, dtype=np.float64))
    meta = pd.DataFrame({
        "race_id": [RACE_ID] * FIELD_SIZE,
        "boat_number": [1, 2, 3, 4, 5, 6],
        "finish_position": list(finish_order),
    })
    return scores, meta


def _fat_odds() -> dict:
    """Trifecta odds large enough for any ticket to pass EV >= 0 filter."""
    odds = {}
    for a in range(1, 7):
        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c in (a, b):
                    continue
                odds[(RACE_ID, f"{a}-{b}-{c}")] = 100.0
    return odds


# ---------------------------------------------------------------------------
# gap12 boundary tests
# ---------------------------------------------------------------------------


def test_gap12_below_threshold_skips_race():
    """gap12 = 0.039 with threshold 0.04 → race is skipped (races=0)."""
    probs = (0.400, 0.361, 0.15, 0.0297, 0.0297, 0.0296)
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.0,
        gap12_min_threshold=0.04,
    )
    assert result["races"] == 0
    assert result["tickets"] == 0


def test_gap12_above_threshold_passes():
    """gap12 = 0.041 with threshold 0.04 → race passes all filters."""
    probs = (0.400, 0.359, 0.15, 0.0303, 0.0303, 0.0304)
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.0,
        gap12_min_threshold=0.04,
    )
    assert result["races"] == 1
    assert result["tickets"] >= 1


def test_gap12_exactly_at_threshold_passes():
    """gap12 = 0.040 exactly with threshold 0.04 → race passes (>= is inclusive)."""
    probs = (0.400, 0.360, 0.15, 0.03, 0.03, 0.03)
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.0,
        gap12_min_threshold=0.04,
    )
    assert result["races"] == 1


def test_gap12_zero_threshold_accepts_tiny_gap():
    """gap12_min_threshold = 0.0 → any positive gap12 passes."""
    probs = (0.3501, 0.3500, 0.15, 0.05, 0.05, 0.0499)  # gap12 ≈ 0.0001
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.0,
        gap12_min_threshold=0.0,
    )
    assert result["races"] == 1


def test_gap12_default_kwarg_is_backward_compat():
    """Omitting gap12_min_threshold reproduces pre-filter behavior (races=1)."""
    probs = (0.3501, 0.3500, 0.15, 0.05, 0.05, 0.0499)  # gap12 ≈ 0.0001
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.0,
        # gap12_min_threshold intentionally omitted → default 0.0
    )
    assert result["races"] == 1


# ---------------------------------------------------------------------------
# Interaction with earlier/later filters
# ---------------------------------------------------------------------------


def test_not_b1_top_skipped_regardless_of_gap12():
    """If top-1 is not boat 1, race is skipped even if gap12 would pass."""
    # Boat 2 is top predicted (highest prob), so filter 1 skips before gap12.
    probs = (0.10, 0.80, 0.05, 0.02, 0.02, 0.01)
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.0,
        gap12_min_threshold=0.04,
    )
    assert result["races"] == 0


def test_conc_filter_still_enforced_after_gap12():
    """Race passing gap12 but failing top3_conc is still skipped."""
    # p1=0.40, p2=0.30, p3=0.05, p4..6 ≈ 0.0833 each
    # gap12 = 0.10 (passes 0.04)
    # top3_conc = (0.30 + 0.05) / (1 - 0.40) = 0.5833
    probs = (0.40, 0.30, 0.05, 0.0833, 0.0833, 0.0834)
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.0,
        ev_threshold=0.0,
        top3_conc_threshold=0.80,  # 0.5833 fails this
        gap12_min_threshold=0.04,  # passes
    )
    assert result["races"] == 0


def test_gap23_filter_still_enforced_after_gap12():
    """Race passing gap12 and conc but failing gap23 is still skipped."""
    # p1=0.40, p2=0.20, p3=0.15 → gap23 = 0.05
    # gap12 = 0.20 (passes), conc = (0.20+0.15)/0.60 = 0.583 (passes with th=0.5)
    probs = (0.40, 0.20, 0.15, 0.0833, 0.0833, 0.0834)
    scores, meta = _make_single_race(probs)
    result = evaluate_p2_strategy(
        rank_scores=scores,
        meta_rank=meta,
        trifecta_odds=_fat_odds(),
        gap23_threshold=0.10,  # 0.05 fails this
        ev_threshold=0.0,
        top3_conc_threshold=0.50,
        gap12_min_threshold=0.04,
    )
    assert result["races"] == 0
