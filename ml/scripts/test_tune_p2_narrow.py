"""Tests for the --narrow search-space helpers in tune_p2.py.

Covers boundary conditions (center at min/max, center outside range,
degenerate ranges) so future tuning of the global ranges or NARROW_*
constants is regression-safe.
"""

from __future__ import annotations

import pytest

from scripts.tune_p2 import (
    NARROW_ABS_DELTA,
    NARROW_LOG_FACTOR,
    NARROW_RATIO,
    _narrow_abs,
    _narrow_float,
    _narrow_int,
    _narrow_log,
)


# ---------------------------------------------------------------------------
# _narrow_int
# ---------------------------------------------------------------------------


def test_narrow_int_typical_center():
    # p2_v2 num_leaves=89, ratio 0.30 -> [62, 115]
    assert _narrow_int(89, 0.30, 15, 127) == (62, 115)


def test_narrow_int_clamps_to_lower_bound():
    # max_depth seed = 3 (= global lo), ratio 0.25 -> [3, 3]
    # 3 * 0.75 = 2 (truncated) -> clamped to 3
    # 3 * 1.25 = 3 (truncated)
    # new_lo (3) >= new_hi (3) -> degenerate to single point
    lo, hi = _narrow_int(3, 0.25, 3, 12)
    assert lo == 3 and hi == 3


def test_narrow_int_clamps_to_upper_bound():
    # max_depth seed = 12 (= global hi), ratio 0.25 -> [9, 12]
    lo, hi = _narrow_int(12, 0.25, 3, 12)
    assert lo == 9 and hi == 12


def test_narrow_int_center_below_global_min():
    # Hypothetical seed n_estimators=50 < global_lo=100
    # int(50*0.7)=35 -> clamped lo=100
    # int(50*1.3)=65 -> clamped hi=65
    # new_lo (100) >= new_hi (65) -> degenerate
    # round(50)=50 -> clamped to 100
    lo, hi = _narrow_int(50, 0.30, 100, 1500)
    assert lo == 100 and hi == 100


def test_narrow_int_center_above_global_max():
    # Hypothetical seed num_leaves=200 > global_hi=127
    # int(200*0.7)=140 -> clamped lo=127 (since lo cannot exceed hi)
    # int(200*1.3)=260 -> clamped hi=127
    # new_lo (127) >= new_hi (127) -> degenerate
    lo, hi = _narrow_int(200, 0.30, 15, 127)
    assert lo == 127 and hi == 127


def test_narrow_int_negative_center_truncation():
    # max_depth=4, ratio=0.25 -> int(4*0.75)=int(3.0)=3, int(4*1.25)=int(5.0)=5
    lo, hi = _narrow_int(4, 0.25, 3, 12)
    assert lo == 3 and hi == 5


# ---------------------------------------------------------------------------
# _narrow_float
# ---------------------------------------------------------------------------


def test_narrow_float_typical_center():
    # subsample seed=0.73, ratio 0.20 -> [0.584, 0.876]
    lo, hi = _narrow_float(0.73, 0.20, 0.4, 1.0)
    assert lo == pytest.approx(0.584)
    assert hi == pytest.approx(0.876)


def test_narrow_float_clamps_to_upper_bound():
    # subsample seed=1.0 (= global hi), ratio 0.20 -> [0.8, 1.0]
    lo, hi = _narrow_float(1.0, 0.20, 0.4, 1.0)
    assert lo == pytest.approx(0.8)
    assert hi == pytest.approx(1.0)


def test_narrow_float_clamps_to_lower_bound():
    # subsample seed=0.4 (= global lo), ratio 0.20 -> [0.4, 0.48]
    lo, hi = _narrow_float(0.4, 0.20, 0.4, 1.0)
    assert lo == pytest.approx(0.4)
    assert hi == pytest.approx(0.48)


# ---------------------------------------------------------------------------
# _narrow_log
# ---------------------------------------------------------------------------


def test_narrow_log_typical_center():
    # learning_rate seed=0.0062, factor=2.0 -> [0.0031, 0.0124]
    # but lower clamped to 0.005
    lo, hi = _narrow_log(0.0062, 2.0, 0.005, 0.2)
    assert lo == pytest.approx(0.005)
    assert hi == pytest.approx(0.0124)


def test_narrow_log_wide_factor():
    # reg_lambda seed=0.099, factor=10 -> [0.0099, 0.99]
    lo, hi = _narrow_log(0.099, 10.0, 1e-8, 10.0)
    assert lo == pytest.approx(0.0099)
    assert hi == pytest.approx(0.99)


def test_narrow_log_underflow_safety():
    # reg_alpha seed=1e-10 (below global lo=1e-8), factor=10
    # 1e-10/10 = 1e-11 < 1e-8 -> clamped to 1e-8
    # 1e-10*10 = 1e-9 < 1e-8 -> clamped to 1e-8
    # new_lo (1e-8) >= new_hi (1e-8) -> degenerate
    lo, hi = _narrow_log(1e-10, 10.0, 1e-8, 10.0)
    assert lo == 1e-8 and hi == 1e-8


def test_narrow_log_clamp_lower_only():
    # learning_rate seed=0.005 (= global lo), factor=2
    # 0.005/2 = 0.0025 -> clamped to 0.005
    # 0.005*2 = 0.01
    lo, hi = _narrow_log(0.005, 2.0, 0.005, 0.2)
    assert lo == pytest.approx(0.005)
    assert hi == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# _narrow_abs
# ---------------------------------------------------------------------------


def test_narrow_abs_zero_center():
    # ev_threshold seed=0.0, delta=0.15, range [-0.3, 0.5] -> [-0.15, 0.15]
    lo, hi = _narrow_abs(0.0, 0.15, -0.3, 0.5)
    assert lo == pytest.approx(-0.15)
    assert hi == pytest.approx(0.15)


def test_narrow_abs_typical_center():
    # top3_conc_threshold seed=0.45, delta=0.15 -> [0.30, 0.60]
    lo, hi = _narrow_abs(0.45, 0.15, 0.0, 0.85)
    assert lo == pytest.approx(0.30)
    assert hi == pytest.approx(0.60)


def test_narrow_abs_clamps_lower():
    # top3_conc_threshold seed=0.05, delta=0.15 -> [0.0, 0.20]
    lo, hi = _narrow_abs(0.05, 0.15, 0.0, 0.85)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.20)


def test_narrow_abs_clamps_upper():
    # gap23_threshold seed=0.23, delta=0.05, range [0.0, 0.25] -> [0.18, 0.25]
    lo, hi = _narrow_abs(0.23, 0.05, 0.0, 0.25)
    assert lo == pytest.approx(0.18)
    assert hi == pytest.approx(0.25)


def test_narrow_abs_center_outside_range():
    # ev_threshold seed=0.6 (above global hi=0.5), delta=0.15
    # new_lo = max(-0.3, 0.45) = 0.45
    # new_hi = min(0.5, 0.75) = 0.5
    lo, hi = _narrow_abs(0.6, 0.15, -0.3, 0.5)
    assert lo == pytest.approx(0.45)
    assert hi == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_narrow_constants_present():
    """All HPs that the objective() narrows must have a constant defined."""
    expected_ratio = {
        "num_leaves", "max_depth", "min_child_samples", "n_estimators",
        "subsample", "colsample_bytree",
    }
    expected_log = {"learning_rate", "reg_alpha", "reg_lambda"}
    expected_abs = {
        "top3_conc_threshold", "gap23_threshold", "ev_threshold",
        "gap12_min_threshold",
    }
    assert set(NARROW_RATIO.keys()) == expected_ratio
    assert set(NARROW_LOG_FACTOR.keys()) == expected_log
    assert set(NARROW_ABS_DELTA.keys()) == expected_abs


def test_narrow_constants_positive():
    for v in NARROW_RATIO.values():
        assert 0 < v < 1.0, f"ratio must be in (0,1), got {v}"
    for v in NARROW_LOG_FACTOR.values():
        assert v > 1.0, f"log factor must be >1, got {v}"
    for v in NARROW_ABS_DELTA.values():
        assert v > 0, f"abs delta must be >0, got {v}"
