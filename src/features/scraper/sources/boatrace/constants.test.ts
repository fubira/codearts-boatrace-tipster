import { describe, expect, test } from "bun:test";
import {
  STADIUMS,
  STADIUM_PREFECTURES,
  beforeInfoUrl,
  dailyScheduleUrl,
  monthlyScheduleUrl,
  odds2TfUrl,
  odds3FUrl,
  odds3TUrl,
  oddsTfUrl,
  raceListUrl,
  raceResultUrl,
} from "./constants";

const params = { raceNumber: 1, stadiumCode: "04", date: "20250115" };

describe("URL builders", () => {
  test("raceListUrl", () => {
    expect(raceListUrl(params)).toBe(
      "/owpc/pc/race/racelist?rno=1&jcd=04&hd=20250115",
    );
  });

  test("beforeInfoUrl", () => {
    expect(beforeInfoUrl(params)).toBe(
      "/owpc/pc/race/beforeinfo?rno=1&jcd=04&hd=20250115",
    );
  });

  test("raceResultUrl", () => {
    expect(raceResultUrl(params)).toBe(
      "/owpc/pc/race/raceresult?rno=1&jcd=04&hd=20250115",
    );
  });

  test("oddsTfUrl", () => {
    expect(oddsTfUrl(params)).toBe(
      "/owpc/pc/race/oddstf?rno=1&jcd=04&hd=20250115",
    );
  });

  test("odds2TfUrl", () => {
    expect(odds2TfUrl(params)).toBe(
      "/owpc/pc/race/odds2tf?rno=1&jcd=04&hd=20250115",
    );
  });

  test("odds3TUrl", () => {
    expect(odds3TUrl(params)).toBe(
      "/owpc/pc/race/odds3t?rno=1&jcd=04&hd=20250115",
    );
  });

  test("odds3FUrl", () => {
    expect(odds3FUrl(params)).toBe(
      "/owpc/pc/race/odds3f?rno=1&jcd=04&hd=20250115",
    );
  });

  test("dailyScheduleUrl", () => {
    expect(dailyScheduleUrl("20250115")).toBe(
      "/owpc/pc/race/index?hd=20250115",
    );
  });

  test("monthlyScheduleUrl appends 01", () => {
    expect(monthlyScheduleUrl("202501")).toBe(
      "/owpc/pc/race/index?hd=20250101",
    );
  });

  test("URL builders with race 12 and stadium 24", () => {
    const edge = { raceNumber: 12, stadiumCode: "24", date: "20260407" };
    expect(raceListUrl(edge)).toBe(
      "/owpc/pc/race/racelist?rno=12&jcd=24&hd=20260407",
    );
  });
});

describe("STADIUMS", () => {
  test("has all 24 venues", () => {
    expect(Object.keys(STADIUMS)).toHaveLength(24);
  });

  test("codes are 01-24 zero-padded", () => {
    for (let i = 1; i <= 24; i++) {
      const code = String(i).padStart(2, "0");
      expect(STADIUMS[code]).toBeDefined();
    }
  });

  test("known mappings", () => {
    expect(STADIUMS["01"]).toBe("桐生");
    expect(STADIUMS["12"]).toBe("住之江");
    expect(STADIUMS["24"]).toBe("大村");
  });
});

describe("STADIUM_PREFECTURES", () => {
  test("has same keys as STADIUMS", () => {
    const stadiumKeys = Object.keys(STADIUMS).sort();
    const prefKeys = Object.keys(STADIUM_PREFECTURES).sort();
    expect(prefKeys).toEqual(stadiumKeys);
  });
});
