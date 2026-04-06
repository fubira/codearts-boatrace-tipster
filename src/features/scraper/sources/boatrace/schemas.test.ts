import { describe, expect, test } from "bun:test";
import {
  beforeInfoDataSchema,
  beforeInfoEntrySchema,
  payoutSchema,
  raceDataSchema,
  raceEntrySchema,
  raceResultDataSchema,
  raceResultEntrySchema,
} from "./schemas";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function validRaceEntry() {
  return {
    racerId: 4444,
    boatNumber: 1,
    racerName: "田中太郎",
  };
}

function validRaceData() {
  return {
    stadiumId: 4,
    stadiumName: "平和島",
    raceDate: "2026-03-03",
    raceNumber: 1,
    entries: [validRaceEntry()],
  };
}

function validRaceResultEntry() {
  return {
    boatNumber: 3,
  };
}

function validRaceResultData() {
  return {
    stadiumId: 12,
    raceDate: "2026-01-15",
    raceNumber: 6,
    entries: [validRaceResultEntry()],
  };
}

function validBeforeInfoEntry() {
  return {
    boatNumber: 2,
  };
}

function validBeforeInfoData() {
  return {
    stadiumId: 1,
    raceDate: "2026-04-01",
    raceNumber: 12,
    entries: [validBeforeInfoEntry()],
  };
}

// ---------------------------------------------------------------------------
// raceEntrySchema
// ---------------------------------------------------------------------------

describe("raceEntrySchema", () => {
  test("accepts minimal valid entry", () => {
    const result = raceEntrySchema.safeParse(validRaceEntry());
    expect(result.success).toBe(true);
  });

  test("accepts full entry with all optional fields", () => {
    const full = {
      ...validRaceEntry(),
      racerClass: "A1",
      racerWeight: 52.0,
      flyingCount: 0,
      lateCount: 1,
      averageSt: 0.15,
      nationalWinRate: 7.35,
      nationalTop2Rate: 20.0,
      nationalTop3Rate: 40.5,
      localWinRate: 8.0,
      localTop2Rate: 25.0,
      localTop3Rate: 50.0,
      motorNumber: 71,
      motorTop2Rate: 33.3,
      motorTop3Rate: 55.0,
      boatNumberAssigned: 33,
      boatTop2Rate: 30.0,
      boatTop3Rate: 45.0,
      branch: "東京",
      birthplace: "東京都",
      birthDate: "1990-01-15",
    };
    const result = raceEntrySchema.safeParse(full);
    expect(result.success).toBe(true);
  });

  test("rejects missing racerId", () => {
    const { racerId, ...rest } = validRaceEntry();
    expect(raceEntrySchema.safeParse(rest).success).toBe(false);
  });

  test("rejects missing boatNumber", () => {
    const { boatNumber, ...rest } = validRaceEntry();
    expect(raceEntrySchema.safeParse(rest).success).toBe(false);
  });

  test("rejects empty racerName", () => {
    const entry = { ...validRaceEntry(), racerName: "" };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects non-integer racerId", () => {
    const entry = { ...validRaceEntry(), racerId: 3.5 };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects zero racerId", () => {
    const entry = { ...validRaceEntry(), racerId: 0 };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects negative racerId", () => {
    const entry = { ...validRaceEntry(), racerId: -1 };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test.each([0, 7, -1])("rejects boatNumber %d", (n) => {
    const entry = { ...validRaceEntry(), boatNumber: n };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test.each([1, 6])("accepts boatNumber boundary %d", (n) => {
    const entry = { ...validRaceEntry(), boatNumber: n };
    expect(raceEntrySchema.safeParse(entry).success).toBe(true);
  });

  test.each([0, 100])("accepts rate boundary %d", (v) => {
    const entry = { ...validRaceEntry(), nationalWinRate: v };
    expect(raceEntrySchema.safeParse(entry).success).toBe(true);
  });

  test.each([-0.01, 100.01])("rejects rate out of range %d", (v) => {
    const entry = { ...validRaceEntry(), nationalWinRate: v };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects zero racerWeight", () => {
    const entry = { ...validRaceEntry(), racerWeight: 0 };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects negative flyingCount", () => {
    const entry = { ...validRaceEntry(), flyingCount: -1 };
    expect(raceEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects garbage data", () => {
    expect(raceEntrySchema.safeParse("not an object").success).toBe(false);
    expect(raceEntrySchema.safeParse(null).success).toBe(false);
    expect(raceEntrySchema.safeParse(42).success).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// raceDataSchema
// ---------------------------------------------------------------------------

describe("raceDataSchema", () => {
  test("accepts minimal valid data", () => {
    expect(raceDataSchema.safeParse(validRaceData()).success).toBe(true);
  });

  test("accepts full data with optional fields", () => {
    const full = {
      ...validRaceData(),
      stadiumPrefecture: "東京都",
      raceTitle: "優勝戦",
      raceGrade: "G1",
      distance: 1800,
      deadline: "15:30",
    };
    expect(raceDataSchema.safeParse(full).success).toBe(true);
  });

  test.each([0, 25, -1])("rejects stadiumId %d", (id) => {
    const data = { ...validRaceData(), stadiumId: id };
    expect(raceDataSchema.safeParse(data).success).toBe(false);
  });

  test.each([1, 24])("accepts stadiumId boundary %d", (id) => {
    const data = { ...validRaceData(), stadiumId: id };
    expect(raceDataSchema.safeParse(data).success).toBe(true);
  });

  test.each([0, 13, -1])("rejects raceNumber %d", (n) => {
    const data = { ...validRaceData(), raceNumber: n };
    expect(raceDataSchema.safeParse(data).success).toBe(false);
  });

  test.each([1, 12])("accepts raceNumber boundary %d", (n) => {
    const data = { ...validRaceData(), raceNumber: n };
    expect(raceDataSchema.safeParse(data).success).toBe(true);
  });

  test("rejects empty stadiumName", () => {
    const data = { ...validRaceData(), stadiumName: "" };
    expect(raceDataSchema.safeParse(data).success).toBe(false);
  });

  test("rejects invalid raceDate format", () => {
    for (const bad of ["20260303", "2026/03/03", "03-03-2026", ""]) {
      const data = { ...validRaceData(), raceDate: bad };
      expect(raceDataSchema.safeParse(data).success).toBe(false);
    }
  });

  test("rejects empty entries", () => {
    const data = { ...validRaceData(), entries: [] };
    expect(raceDataSchema.safeParse(data).success).toBe(false);
  });

  test("rejects more than 6 entries", () => {
    const entries = Array.from({ length: 7 }, (_, i) => ({
      ...validRaceEntry(),
      boatNumber: (i % 6) + 1,
    }));
    const data = { ...validRaceData(), entries };
    expect(raceDataSchema.safeParse(data).success).toBe(false);
  });

  test("accepts exactly 6 entries", () => {
    const entries = Array.from({ length: 6 }, (_, i) => ({
      ...validRaceEntry(),
      boatNumber: i + 1,
    }));
    const data = { ...validRaceData(), entries };
    expect(raceDataSchema.safeParse(data).success).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// raceResultEntrySchema
// ---------------------------------------------------------------------------

describe("raceResultEntrySchema", () => {
  test("accepts minimal entry (boatNumber only)", () => {
    expect(raceResultEntrySchema.safeParse({ boatNumber: 1 }).success).toBe(
      true,
    );
  });

  test("accepts full entry with all optional fields", () => {
    const full = {
      boatNumber: 4,
      courseNumber: 3,
      startTiming: 0.12,
      finishPosition: 2,
      raceTime: "1'49\"3",
    };
    expect(raceResultEntrySchema.safeParse(full).success).toBe(true);
  });

  test.each([0, 7])("rejects courseNumber %d", (n) => {
    const entry = { boatNumber: 1, courseNumber: n };
    expect(raceResultEntrySchema.safeParse(entry).success).toBe(false);
  });

  test.each([0, 7])("rejects finishPosition %d", (n) => {
    const entry = { boatNumber: 1, finishPosition: n };
    expect(raceResultEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("accepts negative startTiming (flying)", () => {
    const entry = { boatNumber: 1, startTiming: -0.02 };
    expect(raceResultEntrySchema.safeParse(entry).success).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// payoutSchema
// ---------------------------------------------------------------------------

describe("payoutSchema", () => {
  test("accepts valid payout", () => {
    const payout = { betType: "3連単", combination: "3-4-1", payout: 12300 };
    expect(payoutSchema.safeParse(payout).success).toBe(true);
  });

  test("rejects empty betType", () => {
    const payout = { betType: "", combination: "1-2", payout: 100 };
    expect(payoutSchema.safeParse(payout).success).toBe(false);
  });

  test("rejects empty combination", () => {
    const payout = { betType: "単勝", combination: "", payout: 100 };
    expect(payoutSchema.safeParse(payout).success).toBe(false);
  });

  test("rejects zero payout", () => {
    const payout = { betType: "単勝", combination: "1", payout: 0 };
    expect(payoutSchema.safeParse(payout).success).toBe(false);
  });

  test("rejects negative payout", () => {
    const payout = { betType: "単勝", combination: "1", payout: -100 };
    expect(payoutSchema.safeParse(payout).success).toBe(false);
  });

  test("rejects non-integer payout", () => {
    const payout = { betType: "単勝", combination: "1", payout: 100.5 };
    expect(payoutSchema.safeParse(payout).success).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// raceResultDataSchema
// ---------------------------------------------------------------------------

describe("raceResultDataSchema", () => {
  test("accepts minimal valid data", () => {
    expect(raceResultDataSchema.safeParse(validRaceResultData()).success).toBe(
      true,
    );
  });

  test("accepts full data with all optional fields", () => {
    const full = {
      ...validRaceResultData(),
      weather: "晴",
      windSpeed: 3,
      windDirection: 180,
      waveHeight: 5,
      temperature: 22.5,
      waterTemperature: 18.0,
      technique: "逃げ",
      payouts: [
        { betType: "3連単", combination: "1-2-3", payout: 1200 },
        { betType: "3連複", combination: "1-2-3", payout: 400 },
      ],
    };
    expect(raceResultDataSchema.safeParse(full).success).toBe(true);
  });

  test("rejects invalid raceDate format", () => {
    const data = { ...validRaceResultData(), raceDate: "2026/01/15" };
    expect(raceResultDataSchema.safeParse(data).success).toBe(false);
  });

  test("rejects negative windSpeed", () => {
    const data = { ...validRaceResultData(), windSpeed: -1 };
    expect(raceResultDataSchema.safeParse(data).success).toBe(false);
  });

  test("rejects negative waveHeight", () => {
    const data = { ...validRaceResultData(), waveHeight: -1 };
    expect(raceResultDataSchema.safeParse(data).success).toBe(false);
  });

  test("accepts zero windSpeed and waveHeight", () => {
    const data = { ...validRaceResultData(), windSpeed: 0, waveHeight: 0 };
    expect(raceResultDataSchema.safeParse(data).success).toBe(true);
  });

  test("rejects empty entries", () => {
    const data = { ...validRaceResultData(), entries: [] };
    expect(raceResultDataSchema.safeParse(data).success).toBe(false);
  });

  test("validates nested payout entries", () => {
    const data = {
      ...validRaceResultData(),
      payouts: [{ betType: "", combination: "1", payout: 100 }],
    };
    expect(raceResultDataSchema.safeParse(data).success).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// beforeInfoEntrySchema
// ---------------------------------------------------------------------------

describe("beforeInfoEntrySchema", () => {
  test("accepts minimal entry (boatNumber only)", () => {
    expect(beforeInfoEntrySchema.safeParse({ boatNumber: 5 }).success).toBe(
      true,
    );
  });

  test("accepts full entry with all optional fields", () => {
    const full = {
      boatNumber: 1,
      exhibitionTime: 6.72,
      tilt: -0.5,
      exhibitionSt: 0.14,
      stabilizer: true,
      partsReplaced: ["プロペラ", "ピストン"],
    };
    expect(beforeInfoEntrySchema.safeParse(full).success).toBe(true);
  });

  test("rejects zero exhibitionTime", () => {
    const entry = { boatNumber: 1, exhibitionTime: 0 };
    expect(beforeInfoEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("rejects negative exhibitionTime", () => {
    const entry = { boatNumber: 1, exhibitionTime: -1 };
    expect(beforeInfoEntrySchema.safeParse(entry).success).toBe(false);
  });

  test("accepts negative tilt", () => {
    const entry = { boatNumber: 1, tilt: -0.5 };
    expect(beforeInfoEntrySchema.safeParse(entry).success).toBe(true);
  });

  test("accepts empty partsReplaced array", () => {
    const entry = { boatNumber: 1, partsReplaced: [] };
    expect(beforeInfoEntrySchema.safeParse(entry).success).toBe(true);
  });

  test("rejects non-boolean stabilizer", () => {
    const entry = { boatNumber: 1, stabilizer: "yes" };
    expect(beforeInfoEntrySchema.safeParse(entry).success).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// beforeInfoDataSchema
// ---------------------------------------------------------------------------

describe("beforeInfoDataSchema", () => {
  test("accepts minimal valid data", () => {
    expect(beforeInfoDataSchema.safeParse(validBeforeInfoData()).success).toBe(
      true,
    );
  });

  test("rejects invalid raceDate format", () => {
    const data = { ...validBeforeInfoData(), raceDate: "20260401" };
    expect(beforeInfoDataSchema.safeParse(data).success).toBe(false);
  });

  test.each([0, 25])("rejects stadiumId %d", (id) => {
    const data = { ...validBeforeInfoData(), stadiumId: id };
    expect(beforeInfoDataSchema.safeParse(data).success).toBe(false);
  });

  test.each([0, 13])("rejects raceNumber %d", (n) => {
    const data = { ...validBeforeInfoData(), raceNumber: n };
    expect(beforeInfoDataSchema.safeParse(data).success).toBe(false);
  });

  test("rejects empty entries", () => {
    const data = { ...validBeforeInfoData(), entries: [] };
    expect(beforeInfoDataSchema.safeParse(data).success).toBe(false);
  });

  test("rejects more than 6 entries", () => {
    const entries = Array.from({ length: 7 }, (_, i) => ({
      boatNumber: (i % 6) + 1,
    }));
    const data = { ...validBeforeInfoData(), entries };
    expect(beforeInfoDataSchema.safeParse(data).success).toBe(false);
  });
});
