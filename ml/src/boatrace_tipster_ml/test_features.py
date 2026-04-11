"""Tests for feature extraction pipeline.

Covers: leakage prevention, determinism, encoding logic, 6-entry integrity.
"""

import numpy as np
import pandas as pd
import pytest

from .feature_config import (
    FEATURE_COLS,
    encode_race_grade,
    encode_racer_class,
    encode_weather,
)
from .features import (
    NON_FINISHER_RANK,
    _cumulative_mean,
    _cumulative_rate,
    _cumulative_std,
    _load_all_data,
    build_features,
)
from .db import get_connection

DB_PATH = "../data/boatrace-tipster.db"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_data():
    """Load full feature set once for all tests."""
    X, y, meta = build_features(DB_PATH)
    return X, y, meta


@pytest.fixture(scope="module")
def raw_data():
    """Load raw data for history-level tests."""
    conn = get_connection(DB_PATH)
    df = _load_all_data(conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# 1. Six-entry integrity
# ---------------------------------------------------------------------------


class TestSixEntryIntegrity:
    def test_all_races_have_6_entries(self, full_data):
        X, y, meta = full_data
        race_counts = meta.groupby("race_id").size()
        assert (race_counts == 6).all(), f"Found races with != 6 entries: {race_counts[race_counts != 6]}"

    def test_entries_divisible_by_6(self, full_data):
        X, y, meta = full_data
        assert len(X) % 6 == 0

    def test_non_finisher_rank_is_7(self, full_data):
        _, y, _ = full_data
        assert y.max() == NON_FINISHER_RANK
        assert y.min() == 1
        assert (y == NON_FINISHER_RANK).sum() > 0

    def test_boat_numbers_1_to_6(self, full_data):
        _, _, meta = full_data
        for _, group in meta.groupby("race_id"):
            boats = sorted(group["boat_number"].values)
            assert boats == [1, 2, 3, 4, 5, 6], f"Bad boat numbers: {boats}"
            break  # Check just one to keep fast; full check below

    def test_boat_numbers_1_to_6_all(self, full_data):
        _, _, meta = full_data
        race_boats = meta.groupby("race_id")["boat_number"].apply(
            lambda x: tuple(sorted(x))
        )
        expected = (1, 2, 3, 4, 5, 6)
        bad = race_boats[race_boats != expected]
        assert len(bad) == 0, f"{len(bad)} races with bad boat numbers"


# ---------------------------------------------------------------------------
# 2. Feature column alignment
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    def test_columns_match_config(self, full_data):
        X, _, _ = full_data
        assert list(X.columns) == FEATURE_COLS

    def test_no_unexpected_nans_in_direct_features(self, full_data):
        """Direct DB features should have near-zero NaN rate."""
        X, _, _ = full_data
        direct_cols = [
            "boat_number",
            "national_win_rate", "national_top2_rate", "national_top3_rate",
        ]
        for col in direct_cols:
            nan_pct = X[col].isna().mean()
            assert nan_pct < 0.001, f"{col} has {nan_pct:.1%} NaN"


# ---------------------------------------------------------------------------
# 3. Encoding logic
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_race_grade_encoding(self):
        assert encode_race_grade("SG") == 5
        assert encode_race_grade("G1") == 4
        assert encode_race_grade("一般") == 1
        assert encode_race_grade(None) == 0
        assert encode_race_grade("") == 0

    def test_racer_class_encoding(self):
        assert encode_racer_class("A1") == 4
        assert encode_racer_class("B2") == 1
        assert encode_racer_class(None) == 0

    def test_weather_encoding(self):
        assert encode_weather("晴") == 1
        assert encode_weather("雨") == 3
        assert encode_weather(None) == 0

    def test_encoded_values_in_features(self, full_data):
        X, _, _ = full_data
        # Encoding columns are used in interaction features (class_x_boat),
        # not directly in FEATURE_COLS. Verify interaction feature exists.
        assert "class_x_boat" in X.columns


# ---------------------------------------------------------------------------
# 4. Leakage prevention
# ---------------------------------------------------------------------------

# Historical features that should be NaN for a racer's first-ever appearance
RACER_HISTORICAL_COLS = [
    "racer_course_win_rate",
    "racer_course_top2_rate",
    "recent_avg_position",
]


class TestLeakagePrevention:
    def test_first_appearance_has_nan_history(self, full_data):
        """Racers appearing for the first time in the dataset should
        have NaN for all historical features."""
        X, _, meta = full_data
        full = pd.concat([meta, X], axis=1)

        # Find racers who first appear after 2024-06 (true first in our data)
        first_dates = full.groupby("racer_id")["race_date"].first()
        late_starters = first_dates[first_dates > "2024-06-01"].index

        if len(late_starters) == 0:
            pytest.skip("No late-starting racers found")

        subset = full[full["racer_id"].isin(late_starters)]
        # Use nth(0) instead of first() — first() skips NaN
        first_entries = subset.groupby("racer_id").nth(0)

        for col in RACER_HISTORICAL_COLS:
            nan_count = first_entries[col].isna().sum()
            assert nan_count == len(first_entries), (
                f"Leakage in {col}: {len(first_entries) - nan_count} "
                f"first-ever entries have non-NaN values"
            )

    def test_cumulative_rate_excludes_current_day(self):
        """cum_all - cum_daily pattern should exclude same-day data."""
        df = pd.DataFrame({
            "racer_id": [1, 1, 1, 1],
            "race_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "race_id": [100, 101, 102, 103],
            "entry_id": [1, 2, 3, 4],
            "_is_win": [1, 0, 1, 0],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        rate = _cumulative_rate(df, ["racer_id"], "_is_win")

        # Row 0 (day1 race1): no prior data → NaN
        assert np.isnan(rate[0])
        # Row 1 (day1 race2): no prior data (same day excluded) → NaN
        assert np.isnan(rate[1])
        # Row 2 (day2 race1): prior = day1 (1 win + 0 win) / 2 = 0.5
        assert rate[2] == pytest.approx(0.5)
        # Row 3 (day2 race2): prior = day1 only (same day excluded) = 0.5
        assert rate[3] == pytest.approx(0.5)

    def test_cumulative_mean_excludes_current_day(self):
        """Cumulative mean should only use prior-day data."""
        df = pd.DataFrame({
            "racer_id": [1, 1, 1],
            "race_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "race_id": [100, 101, 102],
            "entry_id": [1, 2, 3],
            "val": [10.0, 20.0, 30.0],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        mean = _cumulative_mean(df, ["racer_id"], "val")

        assert np.isnan(mean[0])  # No prior
        assert mean[1] == pytest.approx(10.0)  # Only day1
        assert mean[2] == pytest.approx(15.0)  # (10 + 20) / 2

    def test_cumulative_std_excludes_current_day(self):
        """Cumulative std should only use prior-day data."""
        df = pd.DataFrame({
            "racer_id": [1, 1, 1, 1],
            "race_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "race_id": [100, 101, 102, 103],
            "entry_id": [1, 2, 3, 4],
            "val": [10.0, 10.0, 20.0, 20.0],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        std = _cumulative_std(df, ["racer_id"], "val")

        assert np.isnan(std[0])  # No prior
        assert np.isnan(std[1])  # Only 1 prior, need >= 2
        assert std[2] == pytest.approx(0.0)  # std(10, 10) = 0
        assert std[3] == pytest.approx(np.std([10, 10, 20]), abs=0.01)

    def test_racer_course_stats_manual_verification(self):
        """Verify racer_course_win_rate by manual calculation."""
        from .features import _add_racer_course_stats

        df = pd.DataFrame({
            "racer_id": [1, 1, 1, 1, 1],
            "course_number": [1, 1, 1, 1, 1],
            "race_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "race_id": [10, 20, 30, 40, 50],
            "entry_id": [1, 2, 3, 4, 5],
            "finish_position": [1, 3, 1, 2, 1],
            "boat_number": [1, 1, 1, 1, 1],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        _add_racer_course_stats(df)

        # Day1: no prior → NaN
        assert np.isnan(df.iloc[0]["racer_course_win_rate"])
        # Day2: prior = day1 (1 win / 1 race) = 1.0
        assert df.iloc[1]["racer_course_win_rate"] == pytest.approx(1.0)
        # Day3: prior = day1+day2 (1 win / 2 races) = 0.5
        assert df.iloc[2]["racer_course_win_rate"] == pytest.approx(0.5)
        # Day4: prior = day1+2+3 (2 wins / 3 races) = 0.667
        assert df.iloc[3]["racer_course_win_rate"] == pytest.approx(2 / 3)

    def test_course_taking_rate_logic(self):
        """Verify course_taking_rate: rate of taking inner course than boat_number."""
        from .features import _add_course_taking_rate

        df = pd.DataFrame({
            "racer_id": [1, 1, 1, 1],
            "boat_number": [3, 3, 3, 3],
            "actual_course_number": [1, 3, 2, 3],  # inner, same, inner, same
            "race_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "race_id": [10, 20, 30, 40],
            "entry_id": [1, 2, 3, 4],
            "finish_position": [1, 2, 3, 4],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        _add_course_taking_rate(df)

        # Day1: no prior → NaN
        assert np.isnan(df.iloc[0]["course_taking_rate"])
        # Day2: prior = day1 (took inner=1, total=1) = 1.0
        assert df.iloc[1]["course_taking_rate"] == pytest.approx(1.0)
        # Day3: prior = day1+2 (took inner=1, total=2) = 0.5
        assert df.iloc[2]["course_taking_rate"] == pytest.approx(0.5)
        # Day4: prior = day1+2+3 (took inner=2, total=3) = 0.667
        assert df.iloc[3]["course_taking_rate"] == pytest.approx(2 / 3)

        # avg_course_diff: mean of (actual_course - boat_number)
        # diffs = [-2, 0, -1, 0]
        assert np.isnan(df.iloc[0]["avg_course_diff"])
        assert df.iloc[1]["avg_course_diff"] == pytest.approx(-2.0)  # prior: [-2]
        assert df.iloc[2]["avg_course_diff"] == pytest.approx(-1.0)  # prior: [-2, 0]
        assert df.iloc[3]["avg_course_diff"] == pytest.approx(-1.0)  # prior: [-2, 0, -1]

        # course_taking_rate_at_boat: same as course_taking_rate here (all boat 3)
        assert np.isnan(df.iloc[0]["course_taking_rate_at_boat"])
        assert df.iloc[1]["course_taking_rate_at_boat"] == pytest.approx(1.0)
        assert df.iloc[2]["course_taking_rate_at_boat"] == pytest.approx(0.5)
        assert df.iloc[3]["course_taking_rate_at_boat"] == pytest.approx(2 / 3)

    def test_course_taking_rate_at_boat_varies_by_boat(self):
        """course_taking_rate_at_boat tracks per (racer, boat_number) independently."""
        from .features import _add_course_taking_rate

        # Same racer, different boat_numbers across days
        df = pd.DataFrame({
            "racer_id": [1, 1, 1, 1],
            "boat_number": [5, 5, 3, 5],
            "actual_course_number": [2, 5, 1, 3],  # inner, same, inner, inner
            "race_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "race_id": [10, 20, 30, 40],
            "entry_id": [1, 2, 3, 4],
            "finish_position": [1, 2, 3, 4],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        _add_course_taking_rate(df)

        # course_taking_rate (per racer, all boats): day4 prior = [inner, same, inner] = 2/3
        assert df.iloc[3]["course_taking_rate"] == pytest.approx(2 / 3)

        # course_taking_rate_at_boat for boat 5 on day4:
        # prior at boat 5 = [inner(day1), same(day2)] = 1/2
        # (day3 was boat 3, not counted here)
        assert df.iloc[3]["course_taking_rate_at_boat"] == pytest.approx(0.5)

        # course_taking_rate_at_boat for boat 3 on day3:
        # no prior at boat 3 → NaN
        assert np.isnan(df.iloc[2]["course_taking_rate_at_boat"])

    def test_recent_form_uses_cum_all_minus_daily(self):
        """Verify recent_form excludes same-day races (cum_all - cum_daily)."""
        from .features import _add_recent_form

        df = pd.DataFrame({
            "racer_id": [1, 1, 1, 1],
            "race_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "race_id": [10, 20, 30, 40],
            "entry_id": [1, 2, 3, 4],
            "finish_position": [1, 2, 1, 3],
            "boat_number": [1, 2, 1, 2],
            "course_number": [1, 2, 1, 2],
        }).sort_values(["race_date", "race_id", "entry_id"]).reset_index(drop=True)

        _add_recent_form(df)

        # Day1 race1: no prior → NaN
        assert np.isnan(df.iloc[0]["recent_win_rate"])
        # Day1 race2: same day excluded → NaN
        assert np.isnan(df.iloc[1]["recent_win_rate"])
        # Day2 race1: prior = day1 (1 win + 0 win) / 2 = 0.5
        assert df.iloc[2]["recent_win_rate"] == pytest.approx(0.5)
        # Day2 race2: prior = day1 only = 0.5
        assert df.iloc[3]["recent_win_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        """Two sequential builds should produce identical results."""
        X1, y1, meta1 = build_features(DB_PATH, start_date="2025-10-01")
        X2, y2, meta2 = build_features(DB_PATH, start_date="2025-10-01")

        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)
        pd.testing.assert_frame_equal(meta1, meta2)

    def test_sort_order_deterministic(self):
        """Data should always be sorted by (race_date, race_id, entry_id)."""
        X, _, meta = build_features(DB_PATH, start_date="2025-10-01")
        full = pd.concat([meta, X], axis=1)

        # Verify order: race_date should be non-decreasing
        dates = full["race_date"].values
        assert (dates[1:] >= dates[:-1]).all()


# ---------------------------------------------------------------------------
# 6. Raw data integrity
# ---------------------------------------------------------------------------


class TestRawDataIntegrity:
    def test_no_duplicate_entries(self, raw_data):
        """No duplicate (race_id, boat_number) combinations."""
        dupes = raw_data.duplicated(subset=["race_id", "boat_number"], keep=False)
        assert not dupes.any(), f"{dupes.sum()} duplicate entries found"

    def test_race_date_format(self, raw_data):
        """race_date should be YYYY-MM-DD format."""
        sample = raw_data["race_date"].head(100)
        assert sample.str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    def test_boat_number_range(self, raw_data):
        assert raw_data["boat_number"].between(1, 6).all()

    def test_stadium_id_range(self, raw_data):
        assert raw_data["stadium_id"].between(1, 24).all()

    def test_race_number_range(self, raw_data):
        assert raw_data["race_number"].between(1, 12).all()
