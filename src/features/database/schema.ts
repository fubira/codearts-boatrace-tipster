/** SQLite schema definitions for boatrace data */

export const SCHEMA_VERSION = 4;

export const CREATE_TABLES_SQL = `
  CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
  );

  -- 会場マスタ（24場固定）
  CREATE TABLE IF NOT EXISTS stadiums (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    prefecture TEXT
  );

  -- 選手マスタ
  CREATE TABLE IF NOT EXISTS racers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    branch TEXT,
    birthplace TEXT,
    birth_date TEXT,
    class TEXT
  );

  -- レース情報
  CREATE TABLE IF NOT EXISTS races (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stadium_id INTEGER NOT NULL REFERENCES stadiums(id),
    race_date TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    race_title TEXT,
    race_grade TEXT,
    distance INTEGER NOT NULL DEFAULT 1800,
    deadline TEXT,
    weather TEXT,
    wind_speed INTEGER,
    wind_direction INTEGER,
    wave_height INTEGER,
    temperature REAL,
    water_temperature REAL,
    technique TEXT,
    UNIQUE(stadium_id, race_date, race_number)
  );

  -- 出走表（6艇/レース）
  CREATE TABLE IF NOT EXISTS race_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL REFERENCES races(id),
    racer_id INTEGER NOT NULL REFERENCES racers(id),
    boat_number INTEGER NOT NULL,
    course_number INTEGER,
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
    motor_number INTEGER,
    motor_top2_rate REAL,
    motor_top3_rate REAL,
    boat_number_assigned INTEGER,
    boat_top2_rate REAL,
    boat_top3_rate REAL,
    exhibition_time REAL,
    tilt REAL,
    exhibition_st REAL,
    stabilizer INTEGER NOT NULL DEFAULT 0,
    parts_replaced TEXT,
    start_timing REAL,
    finish_position INTEGER,
    race_time TEXT,
    bc_lap_time REAL,
    bc_turn_time REAL,
    bc_straight_time REAL,
    bc_course INTEGER,
    bc_st1 REAL,
    bc_st2 REAL,
    bc_is_flying INTEGER,
    bc_slit_diff REAL,
    UNIQUE(race_id, boat_number)
  );

  -- 払戻金
  CREATE TABLE IF NOT EXISTS race_payouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL REFERENCES races(id),
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    payout INTEGER NOT NULL,
    UNIQUE(race_id, bet_type, combination)
  );

  -- 選手コース別成績
  CREATE TABLE IF NOT EXISTS racer_course_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    racer_id INTEGER NOT NULL REFERENCES racers(id),
    course_number INTEGER NOT NULL,
    win_rate REAL,
    top2_rate REAL,
    top3_rate REAL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(racer_id, course_number)
  );

  -- オッズ（締切時確定オッズ）
  CREATE TABLE IF NOT EXISTS race_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL REFERENCES races(id),
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    odds REAL NOT NULL,
    UNIQUE(race_id, bet_type, combination)
  );

  -- インデックス
  CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
  CREATE INDEX IF NOT EXISTS idx_races_stadium_date ON races(stadium_id, race_date);
  CREATE INDEX IF NOT EXISTS idx_race_entries_race ON race_entries(race_id);
  CREATE INDEX IF NOT EXISTS idx_race_entries_racer ON race_entries(racer_id);
  CREATE INDEX IF NOT EXISTS idx_race_payouts_race ON race_payouts(race_id);
  CREATE INDEX IF NOT EXISTS idx_racer_course_stats_racer ON racer_course_stats(racer_id);
  CREATE INDEX IF NOT EXISTS idx_race_odds_race ON race_odds(race_id);
`;

/** Incremental migrations keyed by target version */
export const MIGRATIONS: Record<number, string> = {
  2: `
    CREATE TABLE IF NOT EXISTS purchase_records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      race_id INTEGER NOT NULL REFERENCES races(id),
      stadium_name TEXT NOT NULL,
      race_number INTEGER NOT NULL,
      race_date TEXT NOT NULL,
      boat_number INTEGER NOT NULL,
      bet_type TEXT NOT NULL,
      amount INTEGER NOT NULL,
      dry_run INTEGER NOT NULL DEFAULT 1,
      success INTEGER NOT NULL DEFAULT 0,
      error TEXT,
      screenshot_path TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_purchase_records_date ON purchase_records(race_date);
    CREATE INDEX IF NOT EXISTS idx_purchase_records_race ON purchase_records(race_id);
  `,
  3: `
    CREATE TABLE IF NOT EXISTS race_odds_snapshots (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      race_id INTEGER NOT NULL REFERENCES races(id),
      timing TEXT NOT NULL,
      bet_type TEXT NOT NULL,
      combination TEXT NOT NULL,
      odds REAL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_odds_snapshots_race_timing
      ON race_odds_snapshots(race_id, timing);
  `,
  4: `
    ALTER TABLE race_entries ADD COLUMN bc_lap_time REAL;
    ALTER TABLE race_entries ADD COLUMN bc_turn_time REAL;
    ALTER TABLE race_entries ADD COLUMN bc_straight_time REAL;
    ALTER TABLE race_entries ADD COLUMN bc_course INTEGER;
    ALTER TABLE race_entries ADD COLUMN bc_st1 REAL;
    ALTER TABLE race_entries ADD COLUMN bc_st2 REAL;
    ALTER TABLE race_entries ADD COLUMN bc_is_flying INTEGER;
    ALTER TABLE race_entries ADD COLUMN bc_slit_diff REAL;
  `,
};
