import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as cheerio from "cheerio";
import type { RaceContext } from "./parsers";
import {
  extractGrade,
  parseBeforeInfo,
  parseRaceList,
  parseRaceResult,
  parseWeather,
} from "./parsers";

const FIXTURES_DIR = resolve(import.meta.dirname, "__fixtures__");

function loadFixture(name: string): string {
  return readFileSync(resolve(FIXTURES_DIR, name), "utf-8");
}

/** Assert non-null and return typed value */
function assertDefined<T>(value: T | null | undefined): T {
  expect(value).toBeDefined();
  expect(value).not.toBeNull();
  return value as T;
}

// Context for 平和島 R1 2026-03-03
const raceListContext: RaceContext = {
  params: { raceNumber: 1, stadiumCode: "04", date: "20260303" },
  raceDate: "2026-03-03",
};

// Context for 平和島 R1 2026-03-02 (result)
const resultContext: RaceContext = {
  params: { raceNumber: 1, stadiumCode: "04", date: "20260302" },
  raceDate: "2026-03-02",
};

describe("parseRaceList", () => {
  const html = loadFixture("racelist.html");
  const result = assertDefined(parseRaceList(html, raceListContext));

  test("parses race metadata", () => {
    expect(result.stadiumId).toBe(4);
    expect(result.stadiumName).toBe("平和島");
    expect(result.raceDate).toBe("2026-03-03");
    expect(result.raceNumber).toBe(1);
    expect(result.raceTitle).toContain("府中市長杯");
    expect(result.raceGrade).toBe("一般");
    expect(result.distance).toBe(1800);
  });

  test("parses deadline time for R1", () => {
    expect(result.deadline).toBe("10:57");
  });

  test("parses deadline time for R12", () => {
    const r12Context: RaceContext = {
      params: { raceNumber: 12, stadiumCode: "04", date: "20260303" },
      raceDate: "2026-03-03",
    };
    const r12 = assertDefined(parseRaceList(html, r12Context));
    expect(r12.deadline).toBe("16:40");
  });

  test("parses 6 entries", () => {
    expect(result.entries).toHaveLength(6);
  });

  test("parses first entry (boat 1)", () => {
    const entry = result.entries[0];
    expect(entry.boatNumber).toBe(1);
    expect(entry.racerId).toBe(3470);
    expect(entry.racerName).toContain("新田");
    expect(entry.racerClass).toBe("B1");
  });

  test("parses racer statistics", () => {
    const entry = result.entries[0];
    // F0/L0/0.17
    expect(entry.flyingCount).toBe(0);
    expect(entry.lateCount).toBe(0);
    expect(entry.averageSt).toBe(0.17);
    // National: 4.86/28.79/46.97
    expect(entry.nationalWinRate).toBe(4.86);
    expect(entry.nationalTop2Rate).toBe(28.79);
    expect(entry.nationalTop3Rate).toBe(46.97);
  });

  test("parses motor and boat stats", () => {
    const entry = result.entries[0];
    expect(entry.motorNumber).toBeNumber();
    expect(entry.boatNumberAssigned).toBeNumber();
  });

  test("parses branch and birthplace", () => {
    const entry = result.entries[0];
    expect(entry.branch).toBe("徳島");
    expect(entry.birthplace).toBe("徳島");
  });

  test("parses all 6 boat numbers correctly", () => {
    const boatNumbers = result.entries.map((e) => e.boatNumber);
    expect(boatNumbers).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe("parseBeforeInfo", () => {
  const html = loadFixture("beforeinfo.html");
  const result = assertDefined(parseBeforeInfo(html, raceListContext));

  test("parses before info metadata", () => {
    expect(result.stadiumId).toBe(4);
    expect(result.raceDate).toBe("2026-03-03");
    expect(result.raceNumber).toBe(1);
  });

  test("parses 6 entries", () => {
    expect(result.entries).toHaveLength(6);
  });

  test("parses exhibition time for boat 1", () => {
    const entry = result.entries[0];
    expect(entry.boatNumber).toBe(1);
    expect(entry.exhibitionTime).toBe(6.69);
  });

  test("parses tilt for boat 1 and 2", () => {
    expect(result.entries[0].tilt).toBe(0.0);
    expect(result.entries[1].tilt).toBe(-0.5);
  });

  test("parses exhibition ST", () => {
    const entry1 = result.entries[0];
    // ST for boat 1: .01
    expect(entry1.exhibitionSt).toBeCloseTo(0.01, 2);
  });

  test("all boats have exhibition times", () => {
    for (const entry of result.entries) {
      expect(entry.exhibitionTime).toBeNumber();
      expect(entry.exhibitionTime).toBeGreaterThan(0);
    }
  });
});

describe("parseBeforeInfo weather", () => {
  const html = loadFixture("beforeinfo.html");

  test("parses weather info", () => {
    const $ = cheerio.load(html);
    const weather = parseWeather($);
    expect(weather.windSpeed).toBe(4);
    expect(weather.windDirection).toBe(13);
    expect(weather.temperature).toBe(8.0);
    expect(weather.waterTemperature).toBe(12.0);
    expect(weather.waveHeight).toBe(3);
  });
});

describe("parseRaceResult", () => {
  const html = loadFixture("raceresult.html");
  const result = assertDefined(parseRaceResult(html, resultContext));

  test("parses result metadata", () => {
    expect(result.stadiumId).toBe(4);
    expect(result.raceDate).toBe("2026-03-02");
    expect(result.raceNumber).toBe(1);
  });

  test("parses 6 result entries", () => {
    expect(result.entries).toHaveLength(6);
  });

  test("parses finish positions", () => {
    // 1st place: boat 2
    const first = assertDefined(
      result.entries.find((e) => e.finishPosition === 1),
    );
    expect(first.boatNumber).toBe(2);
  });

  test("parses race time", () => {
    const first = assertDefined(
      result.entries.find((e) => e.finishPosition === 1),
    );
    expect(first.raceTime).toContain("1'48");
  });

  test("parses start timing", () => {
    // Boat 2 start timing: .18
    const boat2 = assertDefined(result.entries.find((e) => e.boatNumber === 2));
    expect(boat2.startTiming).toBeCloseTo(0.18, 2);
  });

  test("parses course number", () => {
    // Boat 1 is in course 1 (first position in start info)
    const boat1 = assertDefined(result.entries.find((e) => e.boatNumber === 1));
    expect(boat1.courseNumber).toBe(1);
  });

  test("parses technique", () => {
    expect(result.technique).toBe("まくり");
  });

  test("parses weather conditions", () => {
    expect(result.windSpeed).toBe(3);
    expect(result.temperature).toBe(11.0);
    expect(result.waterTemperature).toBe(13.0);
    expect(result.waveHeight).toBe(3);
    expect(result.weather).toBe("曇り");
  });

  test("parses payouts", () => {
    const payouts = assertDefined(result.payouts);
    expect(payouts.length).toBeGreaterThan(0);

    // 3連単 2-4-5 ¥4,360
    const trifecta = assertDefined(payouts.find((p) => p.betType === "3連単"));
    expect(trifecta.combination).toBe("2-4-5");
    expect(trifecta.payout).toBe(4360);

    // 単勝 2 ¥230
    const win = assertDefined(payouts.find((p) => p.betType === "単勝"));
    expect(win.combination).toBe("2");
    expect(win.payout).toBe(230);
  });
});

describe("extractGrade", () => {
  test("parses actual CSS class variations from boatrace.jp", () => {
    expect(extractGrade("heading2_title is-ippan ")).toBe("一般");
    expect(extractGrade("heading2_title is-SGa ")).toBe("SG");
    expect(extractGrade("heading2_title is-G1a ")).toBe("G1");
    expect(extractGrade("heading2_title is-G1b ")).toBe("G1");
    expect(extractGrade("heading2_title is-G2b ")).toBe("G2");
    expect(extractGrade("heading2_title is-G3b ")).toBe("G3");
  });

  test("returns undefined for unknown class", () => {
    expect(extractGrade("heading2_title")).toBeUndefined();
    expect(extractGrade("")).toBeUndefined();
  });
});
