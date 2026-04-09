import { Database } from "bun:sqlite";
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { CREATE_TABLES_SQL } from "./schema";
import {
  type BeforeInfoData,
  type RaceData,
  type RaceResultData,
  isRaceScraped,
  saveBeforeInfo,
  saveRaceResults,
  saveRaces,
} from "./storage";

let db: Database;

function createTestDb(): Database {
  const testDb = new Database(":memory:");
  testDb.exec("PRAGMA foreign_keys=ON");
  testDb.exec(CREATE_TABLES_SQL);
  testDb.query("INSERT INTO schema_version (version) VALUES (1)").run();
  return testDb;
}

function makeRaceData(overrides?: Partial<RaceData>): RaceData {
  return {
    stadiumId: 4,
    stadiumName: "平和島",
    raceDate: "2025-01-15",
    raceNumber: 1,
    entries: [
      {
        racerId: 3456,
        boatNumber: 1,
        racerName: "田中太郎",
        racerClass: "A1",
        nationalWinRate: 7.5,
      },
      {
        racerId: 4567,
        boatNumber: 2,
        racerName: "鈴木次郎",
        racerClass: "B1",
        nationalWinRate: 5.2,
      },
    ],
    ...overrides,
  };
}

beforeEach(() => {
  db = createTestDb();
});

afterEach(() => {
  db.close();
});

describe("saveRaces", () => {
  test("inserts stadiums, racers, races, and entries", () => {
    const data = makeRaceData();
    const result = saveRaces([data], db);

    expect(result.racesUpserted).toBe(1);
    expect(result.entriesUpserted).toBe(2);

    const stadium = db
      .query("SELECT * FROM stadiums WHERE id = 4")
      .get() as Record<string, unknown>;
    expect(stadium.name).toBe("平和島");

    const racers = db.query("SELECT * FROM racers ORDER BY id").all() as Record<
      string,
      unknown
    >[];
    expect(racers).toHaveLength(2);
    expect(racers[0].name).toBe("田中太郎");

    const entries = db
      .query("SELECT * FROM race_entries ORDER BY boat_number")
      .all() as Record<string, unknown>[];
    expect(entries).toHaveLength(2);
    expect(entries[0].national_win_rate).toBe(7.5);
  });

  test("upsert is idempotent", () => {
    const data = makeRaceData();
    saveRaces([data], db);
    saveRaces([data], db);

    const races = db.query("SELECT * FROM races").all();
    expect(races).toHaveLength(1);

    const entries = db.query("SELECT * FROM race_entries").all();
    expect(entries).toHaveLength(2);
  });

  test("updates racer data on conflict", () => {
    const data1 = makeRaceData();
    saveRaces([data1], db);

    const data2 = makeRaceData({
      entries: [
        {
          racerId: 3456,
          boatNumber: 1,
          racerName: "田中太郎",
          racerClass: "A1",
          nationalWinRate: 8.0,
          branch: "東京",
        },
        {
          racerId: 4567,
          boatNumber: 2,
          racerName: "鈴木次郎",
          racerClass: "A2",
          nationalWinRate: 5.5,
        },
      ],
    });
    saveRaces([data2], db);

    const racer = db
      .query("SELECT * FROM racers WHERE id = 3456")
      .get() as Record<string, unknown>;
    expect(racer.branch).toBe("東京");

    const entry = db
      .query("SELECT * FROM race_entries WHERE racer_id = 3456")
      .get() as Record<string, unknown>;
    expect(entry.national_win_rate).toBe(8.0);
  });
});

describe("saveRaceResults", () => {
  test("updates race conditions and entry results", () => {
    saveRaces([makeRaceData()], db);

    const results: RaceResultData[] = [
      {
        stadiumId: 4,
        raceDate: "2025-01-15",
        raceNumber: 1,
        weather: "晴",
        windSpeed: 3,
        windDirection: 8,
        waveHeight: 5,
        temperature: 12.5,
        waterTemperature: 10.0,
        technique: "逃げ",
        entries: [
          {
            boatNumber: 1,
            courseNumber: 1,
            startTiming: 0.15,
            finishPosition: 1,
            raceTime: "1'48\"5",
          },
          {
            boatNumber: 2,
            courseNumber: 2,
            startTiming: 0.2,
            finishPosition: 2,
            raceTime: "1'49\"0",
          },
        ],
        payouts: [
          { betType: "trifecta", combination: "1-2-3", payout: 1234 },
          { betType: "win", combination: "1", payout: 300 },
        ],
      },
    ];

    saveRaceResults(results, db);

    const race = db
      .query("SELECT * FROM races WHERE race_number = 1")
      .get() as Record<string, unknown>;
    expect(race.weather).toBe("晴");
    expect(race.wind_speed).toBe(3);
    expect(race.technique).toBe("逃げ");

    const entry1 = db
      .query("SELECT * FROM race_entries WHERE boat_number = 1")
      .get() as Record<string, unknown>;
    expect(entry1.finish_position).toBe(1);
    expect(entry1.start_timing).toBe(0.15);
    expect(entry1.course_number).toBe(1);

    const payouts = db
      .query("SELECT * FROM race_payouts ORDER BY bet_type")
      .all() as Record<string, unknown>[];
    expect(payouts).toHaveLength(2);
    expect(payouts[0].bet_type).toBe("trifecta");
    expect(payouts[0].payout).toBe(1234);
  });

  test("warns when race not found", () => {
    const results: RaceResultData[] = [
      {
        stadiumId: 99,
        raceDate: "2099-01-01",
        raceNumber: 1,
        entries: [],
      },
    ];
    // Should not throw
    saveRaceResults(results, db);
  });
});

describe("saveBeforeInfo", () => {
  test("updates exhibition data and parts", () => {
    saveRaces([makeRaceData()], db);

    const beforeInfo: BeforeInfoData[] = [
      {
        stadiumId: 4,
        raceDate: "2025-01-15",
        raceNumber: 1,
        entries: [
          {
            boatNumber: 1,
            exhibitionTime: 6.78,
            tilt: -0.5,
            exhibitionSt: 0.12,
            stabilizer: false,
            partsReplaced: ["ピストン", "リング"],
          },
          {
            boatNumber: 2,
            exhibitionTime: 6.85,
            tilt: 0.0,
            stabilizer: true,
          },
        ],
      },
    ];

    saveBeforeInfo(beforeInfo, db);

    const entry1 = db
      .query("SELECT * FROM race_entries WHERE boat_number = 1")
      .get() as Record<string, unknown>;
    expect(entry1.exhibition_time).toBe(6.78);
    expect(entry1.tilt).toBe(-0.5);
    expect(entry1.parts_replaced).toBe('["ピストン","リング"]');
    expect(entry1.stabilizer).toBe(0);

    const entry2 = db
      .query("SELECT * FROM race_entries WHERE boat_number = 2")
      .get() as Record<string, unknown>;
    expect(entry2.stabilizer).toBe(1);
  });
});

describe("isRaceScraped", () => {
  test("returns false when no results exist", () => {
    saveRaces([makeRaceData()], db);
    expect(isRaceScraped(4, "2025-01-15", 1, db)).toBe(false);
  });

  test("returns true when finish_position exists", () => {
    saveRaces([makeRaceData()], db);
    saveRaceResults(
      [
        {
          stadiumId: 4,
          raceDate: "2025-01-15",
          raceNumber: 1,
          entries: [{ boatNumber: 1, finishPosition: 1 }],
        },
      ],
      db,
    );
    expect(isRaceScraped(4, "2025-01-15", 1, db)).toBe(true);
  });

  test("returns false for non-existent race", () => {
    expect(isRaceScraped(99, "2099-01-01", 1, db)).toBe(false);
  });
});

describe("foreign key constraints", () => {
  test("race_entries requires valid race_id", () => {
    db.query("INSERT INTO racers (id, name) VALUES (1, 'テスト選手')").run();

    expect(() => {
      db.query(
        `INSERT INTO race_entries (race_id, racer_id, boat_number)
         VALUES (999, 1, 1)`,
      ).run();
    }).toThrow();
  });

  test("race_payouts requires valid race_id", () => {
    expect(() => {
      db.query(
        `INSERT INTO race_payouts (race_id, bet_type, combination, payout)
         VALUES (999, 'win', '1', 300)`,
      ).run();
    }).toThrow();
  });
});

describe("schema_version", () => {
  test("version is recorded", () => {
    const row = db
      .query("SELECT MAX(version) as v FROM schema_version")
      .get() as { v: number };
    expect(row.v).toBe(1);
  });
});
