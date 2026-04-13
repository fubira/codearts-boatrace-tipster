"""Tests for pure helpers in predict_p2.py.

Focus: build_p2_would_be_tickets — ensures the would-be ticket construction
matches downstream TypeScript expectations (null market_odds/ev when odds
missing, rounded floats otherwise) and that model_prob uses the PL formula
on softmax probabilities.
"""

from __future__ import annotations

import numpy as np

from scripts.predict_p2 import (
    build_p2_would_be_tickets,
    build_racing_boats_index,
)


# Softmax probabilities for 6 boats, indexed by boat_number-1 (i.e. probs[0]
# = P(boat 1 wins)). Ordering: boat 1 is top, then boat 2, then boat 3.
PROBS = np.array([0.40, 0.30, 0.20, 0.05, 0.03, 0.02])
# pred_order in the source uses argsort(-scores); for our fixture boats
# 1/2/3 are already top-3 so i1=0, i2=1, i3=2.
I1, I2, I3 = 0, 1, 2
R2, R3 = 2, 3  # boat numbers
RID = 100


def _expected_model_prob(p_1st: float, p_2nd: float, p_3rd: float) -> float:
    return p_1st * (p_2nd / (1 - p_1st)) * (p_3rd / (1 - p_1st - p_2nd))


def test_both_combos_built_with_odds_and_ev():
    odds = {
        (RID, "1-2-3"): 8.0,
        (RID, "1-3-2"): 12.0,
    }
    result = build_p2_would_be_tickets(
        PROBS, I1, I2, I3, R2, R3, odds, RID,
    )
    assert len(result) == 2

    mp_123 = _expected_model_prob(0.40, 0.30, 0.20)  # ~0.2857
    mp_132 = _expected_model_prob(0.40, 0.20, 0.30)  # ~0.2000

    t0 = result[0]
    assert t0["combo"] == "1-2-3"
    assert t0["model_prob"] == round(mp_123, 6)
    assert t0["market_odds"] == 8.0
    # EV = mp / (1/odds) * 0.75 - 1 = mp * odds * 0.75 - 1
    assert t0["ev"] == round(mp_123 * 8.0 * 0.75 - 1, 4)

    t1 = result[1]
    assert t1["combo"] == "1-3-2"
    assert t1["model_prob"] == round(mp_132, 6)
    assert t1["market_odds"] == 12.0
    assert t1["ev"] == round(mp_132 * 12.0 * 0.75 - 1, 4)


def test_missing_odds_yields_none_market_odds_and_ev():
    odds = {(RID, "1-2-3"): 8.0}  # 1-3-2 missing
    result = build_p2_would_be_tickets(
        PROBS, I1, I2, I3, R2, R3, odds, RID,
    )
    assert result[0]["market_odds"] == 8.0
    assert result[0]["ev"] is not None

    assert result[1]["combo"] == "1-3-2"
    assert result[1]["market_odds"] is None
    assert result[1]["ev"] is None
    # model_prob is still computed even when odds are missing
    assert result[1]["model_prob"] > 0


def test_zero_or_negative_odds_treated_as_missing():
    odds = {
        (RID, "1-2-3"): 0.0,
        (RID, "1-3-2"): -1.5,
    }
    result = build_p2_would_be_tickets(
        PROBS, I1, I2, I3, R2, R3, odds, RID,
    )
    for t in result:
        assert t["market_odds"] is None
        assert t["ev"] is None


def test_different_rid_is_cache_miss():
    """Odds keyed on a different rid must not leak into this race."""
    odds = {
        (99, "1-2-3"): 20.0,  # wrong rid
        (RID, "1-3-2"): 12.0,  # correct rid
    }
    result = build_p2_would_be_tickets(
        PROBS, I1, I2, I3, R2, R3, odds, RID,
    )
    assert result[0]["combo"] == "1-2-3"
    assert result[0]["market_odds"] is None  # rid 99 not looked up
    assert result[1]["combo"] == "1-3-2"
    assert result[1]["market_odds"] == 12.0


def test_model_prob_matches_pl_formula_for_reversed_pair():
    """Reversing r2/r3 in the ticket should swap the p2/p3 factors in PL."""
    odds = {(RID, "1-2-3"): 100.0, (RID, "1-3-2"): 100.0}
    result = build_p2_would_be_tickets(
        PROBS, I1, I2, I3, R2, R3, odds, RID,
    )
    # 1-2-3: p1=0.40, then p2 from remaining 0.60, then p3 from remaining 0.30
    mp_123 = 0.40 * (0.30 / 0.60) * (0.20 / 0.30)
    # 1-3-2: p1=0.40, then p3 from remaining 0.60, then p2 from remaining 0.40
    mp_132 = 0.40 * (0.20 / 0.60) * (0.30 / 0.40)
    assert result[0]["model_prob"] == round(mp_123, 6)
    assert result[1]["model_prob"] == round(mp_132, 6)


def test_consistency_with_tune_p2_trifecta_prob():
    """build_p2_would_be_tickets must use the same PL formula as tune_p2."""
    from scripts.tune_p2 import _trifecta_prob

    odds = {(RID, "1-2-3"): 8.0, (RID, "1-3-2"): 12.0}
    result = build_p2_would_be_tickets(
        PROBS, I1, I2, I3, R2, R3, odds, RID,
    )
    assert result[0]["model_prob"] == round(_trifecta_prob(PROBS, I1, I2, I3), 6)
    assert result[1]["model_prob"] == round(_trifecta_prob(PROBS, I1, I3, I2), 6)


# ---------------------------------------------------------------------------
# build_racing_boats_index — withdrawal detection
# ---------------------------------------------------------------------------


def _all_trifecta_odds(racing_boats: list[int], rid: int = RID, odds: float = 10.0) -> dict:
    """Build every 3連単 combination among the given racing boats."""
    out = {}
    for a in racing_boats:
        for b in racing_boats:
            if b == a:
                continue
            for c in racing_boats:
                if c in (a, b):
                    continue
                out[(rid, f"{a}-{b}-{c}")] = odds
    return out


def test_racing_boats_full_race():
    """Full 3連単 odds across all 6 boats → {1..6}, 120 combos."""
    odds = _all_trifecta_odds([1, 2, 3, 4, 5, 6])
    index = build_racing_boats_index(odds)
    assert index[RID]["boats"] == {1, 2, 3, 4, 5, 6}
    assert index[RID]["count"] == 120


def test_racing_boats_with_one_withdrawal():
    """Boat 2 withdrawn → combos involve only {1,3,4,5,6}, 60 combos."""
    odds = _all_trifecta_odds([1, 3, 4, 5, 6])
    index = build_racing_boats_index(odds)
    # Boat 2 must NOT appear in the index (no combo has it in 1st position)
    assert index[RID]["boats"] == {1, 3, 4, 5, 6}
    assert 2 not in index[RID]["boats"]
    assert index[RID]["count"] == 60


def test_racing_boats_with_two_withdrawals():
    """Boats 2 and 5 withdrawn → combos involve only {1,3,4,6}, 24 combos."""
    odds = _all_trifecta_odds([1, 3, 4, 6])
    index = build_racing_boats_index(odds)
    assert index[RID]["boats"] == {1, 3, 4, 6}
    assert index[RID]["count"] == 24  # 4 × 3 × 2


def test_racing_boats_skips_zero_and_negative_odds():
    """Odds <= 0 should not contribute to the racing set or count."""
    odds = {
        (RID, "1-2-3"): 10.0,
        (RID, "2-1-3"): 0.0,   # zero
        (RID, "3-1-2"): -1.0,  # negative
    }
    index = build_racing_boats_index(odds)
    # Only "1-2-3" contributes → {1}, count 1
    assert index[RID]["boats"] == {1}
    assert index[RID]["count"] == 1


def test_racing_boats_ignores_malformed_combo():
    """Non-digit combo heads are silently skipped (defensive parsing)."""
    odds = {
        (RID, "1-2-3"): 10.0,
        (RID, "abc-1-2"): 10.0,  # non-digit head
        (RID, "-1-2"): 10.0,     # empty head
    }
    index = build_racing_boats_index(odds)
    # Only "1-2-3" contributes
    assert index[RID]["boats"] == {1}
    assert index[RID]["count"] == 1


def test_racing_boats_empty_odds_absent_from_index():
    """Races without odds should not appear in the index."""
    index = build_racing_boats_index({})
    assert index == {}


def test_racing_boats_multiple_rids_independent():
    """Different race_ids should be indexed independently."""
    odds = {
        (100, "1-2-3"): 10.0,
        (100, "3-4-5"): 10.0,
        (200, "4-5-6"): 10.0,
    }
    index = build_racing_boats_index(odds)
    assert index[100]["boats"] == {1, 3}
    assert index[100]["count"] == 2
    assert index[200]["boats"] == {4}
    assert index[200]["count"] == 1
