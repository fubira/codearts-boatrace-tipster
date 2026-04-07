"""Tests for snapshot pipeline (build_snapshot + build_features_from_snapshot).

The snapshot pipeline is the production inference path. If it produces wrong
features, predictions are wrong. These tests verify structural properties
without depending on the real database.

Covers:
1. build_snapshot creates valid SQLite with expected tables/columns
2. build_features_from_snapshot returns DataFrame with all FEATURE_COLS
3. Output shape (6 entries per race), no unexpected NaN
4. Feature column order matches FEATURE_COLS exactly
5. Z-score properties of rel_* columns
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .feature_config import FEATURE_COLS
from .snapshot import (
    _CUMULATIVE_DEFS,
    _ROLLING_DEFS,
    build_snapshot,
    load_snapshot,
)
from .snapshot_features import build_features_from_snapshot


# ---------------------------------------------------------------------------
# Helpers: Create a synthetic SQLite DB that mimics the real schema
# ---------------------------------------------------------------------------

_RACES_DDL = """
CREATE TABLE races (
    id INTEGER PRIMARY KEY,
    race_date TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    stadium_id INTEGER NOT NULL,
    race_grade TEXT,
    race_title TEXT,
    weather TEXT,
    wind_speed INTEGER,
    wind_direction INTEGER,
    wave_height INTEGER,
    temperature REAL,
    water_temperature REAL
)
"""

_ENTRIES_DDL = """
CREATE TABLE race_entries (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    racer_id INTEGER NOT NULL,
    boat_number INTEGER NOT NULL,
    course_number INTEGER,
    motor_number INTEGER,
    racer_class TEXT,
    racer_weight REAL,
    flying_count INTEGER,
    late_count INTEGER,
    average_st REAL,
    national_win_rate REAL,
    national_top2_rate REAL,
    national_top3_rate REAL,
    local_win_rate REAL,
    local_top2_rate REAL,
    local_top3_rate REAL,
    motor_top2_rate REAL,
    motor_top3_rate REAL,
    boat_top2_rate REAL,
    boat_top3_rate REAL,
    exhibition_time REAL,
    exhibition_st REAL,
    tilt REAL,
    stabilizer INTEGER,
    start_timing REAL,
    finish_position INTEGER,
    bc_lap_time REAL,
    bc_turn_time REAL,
    bc_straight_time REAL,
    bc_slit_diff REAL,
    FOREIGN KEY (race_id) REFERENCES races(id)
)
"""

_ODDS_DDL = """
CREATE TABLE race_odds (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    odds REAL,
    FOREIGN KEY (race_id) REFERENCES races(id)
)
"""


def _create_synthetic_db(db_path: str, n_days: int = 5, races_per_day: int = 2) -> None:
    """Create a small SQLite DB with synthetic race data.

    Generates n_days of data, each with races_per_day races of 6 entries.
    Uses deterministic data so tests are reproducible.
    """
    rng = np.random.RandomState(42)
    conn = sqlite3.connect(db_path)
    conn.execute(_RACES_DDL)
    conn.execute(_ENTRIES_DDL)
    conn.execute(_ODDS_DDL)

    race_id = 1
    entry_id = 1
    odds_id = 1
    # Use a small set of racer_ids so cumulative stats accumulate
    racer_pool = list(range(1001, 1013))  # 12 racers for 6 per race

    for day_idx in range(n_days):
        date_str = f"2025-01-{10 + day_idx:02d}"
        for race_idx in range(races_per_day):
            stadium_id = 4
            conn.execute(
                "INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    race_id, date_str, race_idx + 1, stadium_id,
                    "一般", "テスト開催", "晴",
                    rng.randint(0, 8), rng.randint(0, 18), rng.randint(0, 3),
                    20.0, 18.0,
                ),
            )

            # Pick 6 racers for this race
            racers = rng.choice(racer_pool, size=6, replace=False)
            finish_order = rng.permutation(6) + 1

            for boat in range(1, 7):
                racer_id = int(racers[boat - 1])
                conn.execute(
                    """INSERT INTO race_entries VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry_id, race_id, racer_id, boat,
                        boat,  # course_number = boat_number (no front-taking)
                        100 + boat,  # motor_number
                        "A1",  # racer_class
                        50.0 + rng.rand() * 5,  # racer_weight
                        0, 0,  # flying/late count
                        0.15 + rng.rand() * 0.05,  # average_st
                        5.0 + rng.rand() * 3,  # national_win_rate
                        20.0 + rng.rand() * 10,  # national_top2_rate
                        30.0 + rng.rand() * 10,  # national_top3_rate
                        5.0 + rng.rand() * 3,  # local_win_rate
                        20.0 + rng.rand() * 10,  # local_top2_rate
                        30.0 + rng.rand() * 10,  # local_top3_rate
                        25.0 + rng.rand() * 10,  # motor_top2_rate
                        35.0 + rng.rand() * 10,  # motor_top3_rate
                        25.0 + rng.rand() * 10,  # boat_top2_rate
                        35.0 + rng.rand() * 10,  # boat_top3_rate
                        6.5 + rng.rand() * 0.5,  # exhibition_time
                        0.1 + rng.rand() * 0.2,  # exhibition_st
                        -0.5,  # tilt
                        0,  # stabilizer
                        0.1 + rng.rand() * 0.2,  # start_timing
                        int(finish_order[boat - 1]),  # finish_position
                        37.0 + rng.rand() * 2,  # bc_lap_time
                        5.5 + rng.rand() * 0.5,  # bc_turn_time
                        7.0 + rng.rand() * 0.5,  # bc_straight_time
                        rng.rand() * 3,  # bc_slit_diff
                    ),
                )
                # Add tansho odds
                conn.execute(
                    "INSERT INTO race_odds VALUES (?, ?, ?, ?, ?)",
                    (odds_id, race_id, "単勝", str(boat), 2.0 + rng.rand() * 20),
                )
                odds_id += 1
                entry_id += 1
            race_id += 1

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_db(tmp_path: Path) -> str:
    """Create a synthetic DB and return its path."""
    db_path = str(tmp_path / "test.db")
    _create_synthetic_db(db_path, n_days=5, races_per_day=2)
    return db_path


@pytest.fixture
def snapshot_path(tmp_path: Path) -> str:
    """Return a path for the snapshot file."""
    return str(tmp_path / "snapshot.db")


@pytest.fixture
def built_snapshot(synthetic_db: str, snapshot_path: str) -> str:
    """Build a snapshot and return the snapshot path."""
    # Snapshot through day 4 (2025-01-13), predict on day 5 (2025-01-14)
    build_snapshot(synthetic_db, snapshot_path, through_date="2025-01-13")
    return snapshot_path


# ---------------------------------------------------------------------------
# 1. build_snapshot creates valid SQLite with expected tables
# ---------------------------------------------------------------------------


class TestBuildSnapshotStructure:
    def test_creates_file(self, built_snapshot: str):
        assert Path(built_snapshot).exists()
        assert Path(built_snapshot).stat().st_size > 0

    def test_has_expected_tables(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert tables == {"cumulative_stats", "motor_stats", "rolling_daily", "snapshot_meta"}

    def test_meta_contains_through_date(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        meta = dict(conn.execute("SELECT key, value FROM snapshot_meta").fetchall())
        conn.close()
        assert meta["through_date"] == "2025-01-13"
        assert "built_at" in meta
        assert "entry_count" in meta
        assert int(meta["entry_count"]) > 0


class TestCumulativeStatsTable:
    def test_has_rows(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        count = conn.execute("SELECT COUNT(*) FROM cumulative_stats").fetchone()[0]
        conn.close()
        assert count > 0

    def test_expected_stat_names(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        names = {
            row[0]
            for row in conn.execute("SELECT DISTINCT stat_name FROM cumulative_stats")
        }
        conn.close()
        expected = {d[0] for d in _CUMULATIVE_DEFS}
        assert names == expected

    def test_counts_are_positive(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        min_count = conn.execute(
            "SELECT MIN(total_count) FROM cumulative_stats"
        ).fetchone()[0]
        conn.close()
        assert min_count > 0

    def test_std_stat_has_sum_sq(self, built_snapshot: str):
        """The 'std' aggregation type should have non-null total_sum_sq."""
        conn = sqlite3.connect(built_snapshot)
        rows = conn.execute(
            "SELECT total_sum_sq FROM cumulative_stats WHERE stat_name = 'racer_st_stability'"
        ).fetchall()
        conn.close()
        assert len(rows) > 0
        assert all(row[0] is not None for row in rows)


class TestMotorStatsTable:
    def test_has_rows(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        count = conn.execute("SELECT COUNT(*) FROM motor_stats").fetchone()[0]
        conn.close()
        assert count > 0

    def test_counts_are_positive(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        min_count = conn.execute(
            "SELECT MIN(residual_count) FROM motor_stats"
        ).fetchone()[0]
        conn.close()
        assert min_count > 0


class TestRollingDailyTable:
    def test_has_rows(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        count = conn.execute("SELECT COUNT(*) FROM rolling_daily").fetchone()[0]
        conn.close()
        assert count > 0

    def test_expected_stat_names(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        names = {
            row[0]
            for row in conn.execute("SELECT DISTINCT stat_name FROM rolling_daily")
        }
        conn.close()
        expected = {d[0] for d in _ROLLING_DEFS}
        assert names == expected

    def test_dates_within_through_date(self, built_snapshot: str):
        conn = sqlite3.connect(built_snapshot)
        max_date = conn.execute(
            "SELECT MAX(race_date) FROM rolling_daily"
        ).fetchone()[0]
        conn.close()
        assert max_date <= "2025-01-13"


# ---------------------------------------------------------------------------
# 2. load_snapshot round-trip
# ---------------------------------------------------------------------------


class TestLoadSnapshot:
    def test_returns_dict_with_expected_keys(self, built_snapshot: str):
        snap = load_snapshot(built_snapshot)
        assert set(snap.keys()) == {"meta", "cumulative", "motor", "rolling"}

    def test_meta_through_date(self, built_snapshot: str):
        snap = load_snapshot(built_snapshot)
        assert snap["meta"]["through_date"] == "2025-01-13"

    def test_cumulative_is_dict(self, built_snapshot: str):
        snap = load_snapshot(built_snapshot)
        assert isinstance(snap["cumulative"], dict)
        assert len(snap["cumulative"]) > 0
        # Keys are (stat_name, group_key) tuples
        key = next(iter(snap["cumulative"]))
        assert isinstance(key, tuple) and len(key) == 2

    def test_motor_is_dict(self, built_snapshot: str):
        snap = load_snapshot(built_snapshot)
        assert isinstance(snap["motor"], dict)
        assert len(snap["motor"]) > 0

    def test_rolling_is_dict_of_lists(self, built_snapshot: str):
        snap = load_snapshot(built_snapshot)
        assert isinstance(snap["rolling"], dict)
        assert len(snap["rolling"]) > 0
        # Each value is a list of (date, sum, count) tuples
        val = next(iter(snap["rolling"].values()))
        assert isinstance(val, list) and len(val) > 0
        assert len(val[0]) == 3


# ---------------------------------------------------------------------------
# 3. build_features_from_snapshot output structure
# ---------------------------------------------------------------------------


class TestBuildFeaturesFromSnapshot:
    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_db: str, built_snapshot: str):
        self.db_path = synthetic_db
        self.snapshot_path = built_snapshot
        self.target_date = "2025-01-14"

    def _build(self) -> pd.DataFrame:
        return build_features_from_snapshot(
            self.db_path, self.snapshot_path, self.target_date,
        )

    def test_returns_dataframe(self):
        df = self._build()
        assert isinstance(df, pd.DataFrame)

    def test_six_entries_per_race(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        race_counts = df.groupby("race_id").size()
        assert (race_counts == 6).all(), (
            f"Expected 6 entries per race, got: {race_counts.unique()}"
        )

    def test_all_feature_cols_present(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        assert missing == [], f"Missing FEATURE_COLS: {missing}"

    def test_feature_col_order_matches(self):
        """Feature columns must appear in FEATURE_COLS order when selected."""
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        # Selecting FEATURE_COLS should not raise KeyError
        X = df[FEATURE_COLS]
        assert list(X.columns) == FEATURE_COLS

    def test_boat_number_no_nan(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert not df["boat_number"].isna().any()

    def test_national_win_rate_no_nan(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert not df["national_win_rate"].isna().any()

    def test_exhibition_time_no_nan(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert not df["exhibition_time"].isna().any()

    def test_course_number_range(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert df["course_number"].between(1, 6).all()

    def test_boat_number_range(self):
        df = self._build()
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert df["boat_number"].between(1, 6).all()


# ---------------------------------------------------------------------------
# 4. Z-score properties of rel_* columns
# ---------------------------------------------------------------------------


class TestSnapshotZScoreProperties:
    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_db: str, built_snapshot: str):
        self.df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2025-01-14",
        )

    def test_rel_national_win_rate_zero_mean(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        means = df.groupby("race_id")["rel_national_win_rate"].mean()
        for race_id, m in means.items():
            assert abs(m) < 1e-10, f"Race {race_id}: rel_national_win_rate mean = {m}"

    def test_rel_exhibition_time_zero_mean(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        means = df.groupby("race_id")["rel_exhibition_time"].mean()
        for race_id, m in means.items():
            assert abs(m) < 1e-10, f"Race {race_id}: rel_exhibition_time mean = {m}"

    def test_rel_exhibition_st_zero_mean(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        means = df.groupby("race_id")["rel_exhibition_st"].mean()
        for race_id, m in means.items():
            assert abs(m) < 1e-10, f"Race {race_id}: rel_exhibition_st mean = {m}"


# ---------------------------------------------------------------------------
# 5. Cumulative features have reasonable values
# ---------------------------------------------------------------------------


class TestSnapshotCumulativeFeatures:
    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_db: str, built_snapshot: str):
        self.df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2025-01-14",
        )

    def test_racer_course_win_rate_bounded(self):
        """Win rates should be between 0 and 1."""
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        valid = df["racer_course_win_rate"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_stadium_course_win_rate_bounded(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        valid = df["stadium_course_win_rate"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_st_stability_non_negative(self):
        """Standard deviation is always >= 0."""
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        valid = df["st_stability"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 6. Leaked feature columns present (NaN is OK)
# ---------------------------------------------------------------------------


class TestSnapshotLeakedColumns:
    def test_gate_bias_column_exists(self, synthetic_db: str, built_snapshot: str):
        df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2025-01-14",
        )
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "gate_bias" in df.columns

    def test_upset_rate_column_exists(self, synthetic_db: str, built_snapshot: str):
        df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2025-01-14",
        )
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "upset_rate" in df.columns


# ---------------------------------------------------------------------------
# 7. Encoding columns present and valid
# ---------------------------------------------------------------------------


class TestSnapshotEncodings:
    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_db: str, built_snapshot: str):
        self.df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2025-01-14",
        )

    def test_race_grade_code_present(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "race_grade_code" in df.columns
        assert not df["race_grade_code"].isna().any()

    def test_racer_class_code_present(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "racer_class_code" in df.columns
        assert not df["racer_class_code"].isna().any()

    def test_weather_code_present(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "weather_code" in df.columns
        assert not df["weather_code"].isna().any()


# ---------------------------------------------------------------------------
# 8. Interaction features present
# ---------------------------------------------------------------------------


class TestSnapshotInteractionFeatures:
    @pytest.fixture(autouse=True)
    def _setup(self, synthetic_db: str, built_snapshot: str):
        self.df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2025-01-14",
        )

    def test_class_x_boat_present(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "class_x_boat" in df.columns
        assert not df["class_x_boat"].isna().any()

    def test_kado_x_exhibition_present(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "kado_x_exhibition" in df.columns

    def test_wind_speed_x_boat_present(self):
        df = self.df
        if len(df) == 0:
            pytest.skip("No races on target date")
        assert "wind_speed_x_boat" in df.columns
        assert not df["wind_speed_x_boat"].isna().any()


# ---------------------------------------------------------------------------
# 9. Empty target date returns empty DataFrame
# ---------------------------------------------------------------------------


class TestSnapshotEmptyDate:
    def test_no_races_returns_empty(self, synthetic_db: str, built_snapshot: str):
        df = build_features_from_snapshot(
            synthetic_db, built_snapshot, "2099-12-31",
        )
        assert len(df) == 0
