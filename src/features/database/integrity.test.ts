import { Database } from "bun:sqlite";
import { describe, expect, test } from "bun:test";
import type { CheckResult } from "./integrity";
import { checkIntegrity } from "./integrity";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createTestDb(): Database {
  const db = new Database(":memory:");
  db.run(`CREATE TABLE races (
    id INTEGER PRIMARY KEY,
    race_date TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    stadium_id INTEGER NOT NULL,
    race_grade TEXT, race_title TEXT, weather TEXT,
    wind_speed INTEGER, wind_direction INTEGER, wave_height INTEGER,
    temperature REAL, water_temperature REAL
  )`);
  db.run(`CREATE TABLE race_entries (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    racer_id INTEGER NOT NULL,
    boat_number INTEGER NOT NULL,
    course_number INTEGER, motor_number INTEGER, racer_class TEXT,
    racer_weight REAL, flying_count INTEGER, late_count INTEGER,
    average_st REAL, national_win_rate REAL, national_top2_rate REAL,
    national_top3_rate REAL, local_win_rate REAL, local_top2_rate REAL,
    local_top3_rate REAL, motor_top2_rate REAL, motor_top3_rate REAL,
    boat_top2_rate REAL, boat_top3_rate REAL,
    exhibition_time REAL, exhibition_st REAL, tilt REAL,
    stabilizer INTEGER, start_timing REAL, finish_position INTEGER,
    FOREIGN KEY (race_id) REFERENCES races(id)
  )`);
  db.run(`CREATE TABLE race_payouts (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    payout INTEGER NOT NULL,
    FOREIGN KEY (race_id) REFERENCES races(id)
  )`);
  db.run(`CREATE TABLE race_odds (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    bet_type TEXT NOT NULL,
    combination TEXT NOT NULL,
    odds REAL,
    FOREIGN KEY (race_id) REFERENCES races(id)
  )`);
  return db;
}

/** Insert a complete race with 6 entries, result, and payout */
function insertRace(
  db: Database,
  raceId: number,
  opts: {
    date?: string;
    raceNumber?: number;
    stadiumId?: number;
    weather?: string | null;
    entryCount?: number;
    withPayout?: boolean;
    boatNumberOverride?: number[];
    finishPositions?: (number | null)[];
  } = {},
) {
  const {
    date = "2025-01-10",
    raceNumber = 1,
    stadiumId = 4,
    weather = "晴",
    entryCount = 6,
    withPayout = true,
    boatNumberOverride,
    finishPositions,
  } = opts;

  db.run(
    "INSERT INTO races VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
    raceId,
    date,
    raceNumber,
    stadiumId,
    "一般",
    "テスト",
    weather,
    3,
    5,
    1,
    20,
    18,
  );

  for (let i = 0; i < entryCount; i++) {
    const boat = boatNumberOverride ? boatNumberOverride[i] : i + 1;
    const finish = finishPositions ? finishPositions[i] : i + 1;
    db.run(
      "INSERT INTO race_entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
      raceId * 100 + i,
      raceId,
      1000 + i,
      boat,
      boat,
      100 + boat,
      "A1",
      52,
      0,
      0,
      0.15,
      5.0,
      30.0,
      50.0,
      5.0,
      30.0,
      50.0,
      30.0,
      45.0,
      30.0,
      45.0,
      6.8,
      0.15,
      -0.5,
      0,
      0.15,
      finish,
    );
  }

  if (withPayout) {
    db.run(
      "INSERT INTO race_payouts VALUES (?,?,?,?,?)",
      raceId * 10,
      raceId,
      "3連単",
      "1-2-3",
      12345,
    );
  }
}

function findCheck(results: CheckResult[], name: string): CheckResult {
  const r = results.find((c) => c.name === name);
  if (!r) throw new Error(`Check '${name}' not found in results`);
  return r;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("checkIntegrity", () => {
  describe("clean database", () => {
    test("all checks pass on valid data", () => {
      const db = createTestDb();
      // Insert 3 months of data for completeness check
      let raceId = 1;
      for (const month of ["2025-01", "2025-02", "2025-03"]) {
        for (let day = 1; day <= 20; day++) {
          for (let rn = 1; rn <= 12; rn++) {
            const date = `${month}-${String(day).padStart(2, "0")}`;
            insertRace(db, raceId, { date, raceNumber: rn, stadiumId: 4 });
            raceId++;
          }
        }
      }
      const results = checkIntegrity(db);
      const errors = results.filter((r) => r.status === "error");
      expect(errors).toHaveLength(0);
    });
  });

  describe("orphan entries", () => {
    test("detects entries referencing non-existent race", () => {
      const db = createTestDb();
      insertRace(db, 1);
      // Insert orphan entry with race_id=999 (doesn't exist)
      db.run(
        `INSERT INTO race_entries VALUES (999,999,1000,1,1,100,'A1',52,0,0,0.15,5,30,50,5,30,50,30,45,30,45,6.8,0.15,-0.5,0,0.15,1)`,
      );
      const results = checkIntegrity(db);
      const check = findCheck(results, "Orphan entries");
      expect(check.status).toBe("error");
      expect(check.detail).toContain("1");
    });
  });

  describe("orphan payouts", () => {
    test("detects payouts referencing non-existent race", () => {
      const db = createTestDb();
      insertRace(db, 1);
      db.run("INSERT INTO race_payouts VALUES (999,999,'3連単','1-2-3',5000)");
      const results = checkIntegrity(db);
      const check = findCheck(results, "Orphan payouts");
      expect(check.status).toBe("error");
    });
  });

  describe("duplicate races", () => {
    test("detects duplicate race keys", () => {
      const db = createTestDb();
      insertRace(db, 1, { date: "2025-01-10", raceNumber: 1, stadiumId: 4 });
      // Same stadium/date/raceNumber but different id
      insertRace(db, 2, { date: "2025-01-10", raceNumber: 1, stadiumId: 4 });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Duplicate races");
      expect(check.status).toBe("error");
    });

    test("ok for same race number at different stadiums", () => {
      const db = createTestDb();
      insertRace(db, 1, { stadiumId: 4 });
      insertRace(db, 2, { stadiumId: 5 });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Duplicate races");
      expect(check.status).toBe("ok");
    });
  });

  describe("duplicate entries", () => {
    test("detects duplicate boat in same race", () => {
      const db = createTestDb();
      insertRace(db, 1, { boatNumberOverride: [1, 1, 3, 4, 5, 6] });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Duplicate entries");
      expect(check.status).toBe("error");
    });
  });

  describe("entry count", () => {
    test("detects race with fewer than 6 entries", () => {
      const db = createTestDb();
      insertRace(db, 1, { entryCount: 5 });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Entry count");
      expect(check.status).toBe("error");
    });

    test("ok for exactly 6 entries", () => {
      const db = createTestDb();
      insertRace(db, 1);
      const results = checkIntegrity(db);
      const check = findCheck(results, "Entry count");
      expect(check.status).toBe("ok");
    });
  });

  describe("races without entries", () => {
    test("detects race with no entries", () => {
      const db = createTestDb();
      // Insert race without entries
      db.run(
        "INSERT INTO races VALUES (1,'2025-01-10',1,4,'一般','テスト','晴',3,5,1,20,18)",
      );
      const results = checkIntegrity(db);
      const check = findCheck(results, "Races without entries");
      expect(check.status).toBe("error");
    });
  });

  describe("boat number range", () => {
    test("detects boat number outside 1-6", () => {
      const db = createTestDb();
      insertRace(db, 1, { boatNumberOverride: [0, 2, 3, 4, 5, 6] });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Boat number range");
      expect(check.status).toBe("error");
    });

    test("detects boat number 7", () => {
      const db = createTestDb();
      insertRace(db, 1, { boatNumberOverride: [1, 2, 3, 4, 5, 7] });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Boat number range");
      expect(check.status).toBe("error");
    });
  });

  describe("finish position range", () => {
    test("detects position outside 1-6", () => {
      const db = createTestDb();
      insertRace(db, 1, { finishPositions: [1, 2, 3, 4, 5, 7] });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Finish position range");
      expect(check.status).toBe("error");
    });

    test("allows NULL finish position", () => {
      const db = createTestDb();
      insertRace(db, 1, { finishPositions: [1, 2, 3, 4, 5, null] });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Finish position range");
      expect(check.status).toBe("ok");
    });
  });

  describe("race number range", () => {
    test("detects race number outside 1-12", () => {
      const db = createTestDb();
      insertRace(db, 1, { raceNumber: 13 });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Race number range");
      expect(check.status).toBe("error");
    });

    test("allows race number 12", () => {
      const db = createTestDb();
      insertRace(db, 1, { raceNumber: 12 });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Race number range");
      expect(check.status).toBe("ok");
    });
  });

  describe("result coverage", () => {
    test("warns when many races have no results", () => {
      const db = createTestDb();
      // 10 races without results (weather=NULL), 1 with
      for (let i = 1; i <= 10; i++) {
        insertRace(db, i, { weather: null, date: `2025-01-${10 + i}` });
      }
      insertRace(db, 11, { weather: "晴", date: "2025-01-21" });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Result coverage");
      expect(check.status).toBe("warn");
    });

    test("ok when most races have results", () => {
      const db = createTestDb();
      for (let i = 1; i <= 20; i++) {
        insertRace(db, i, { date: `2025-01-${String(i).padStart(2, "0")}` });
      }
      const results = checkIntegrity(db);
      const check = findCheck(results, "Result coverage");
      expect(check.status).toBe("ok");
    });
  });

  describe("payout coverage", () => {
    test("warns when completed race has no payout", () => {
      const db = createTestDb();
      insertRace(db, 1, { weather: "晴", withPayout: false });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Payout coverage");
      expect(check.status).toBe("warn");
    });

    test("ok for race without results (no payout expected)", () => {
      const db = createTestDb();
      insertRace(db, 1, { weather: null, withPayout: false });
      const results = checkIntegrity(db);
      const check = findCheck(results, "Payout coverage");
      expect(check.status).toBe("ok");
    });
  });
});
