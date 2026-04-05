import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseOdds2Tf, parseOdds3T } from "./odds-parsers";

const FIXTURES_DIR = resolve(import.meta.dirname, "__fixtures__");

function loadFixture(name: string): string {
  return readFileSync(resolve(FIXTURES_DIR, name), "utf-8");
}

describe("parseOdds3T — 3連単 (full race)", () => {
  const entries = parseOdds3T(loadFixture("odds3t.html"));

  test("returns exactly 120 combinations (6×5×4)", () => {
    expect(entries).toHaveLength(120);
  });

  test("all combinations are unique", () => {
    const combos = entries.map((e) => e.combination);
    expect(new Set(combos).size).toBe(120);
  });

  test("all betType is 3連単", () => {
    for (const e of entries) {
      expect(e.betType).toBe("3連単");
    }
  });

  test("combination format is X-Y-Z with single digits", () => {
    for (const e of entries) {
      expect(e.combination).toMatch(/^[1-6]-[1-6]-[1-6]$/);
    }
  });

  test("no combination has duplicate boats", () => {
    for (const e of entries) {
      const [a, b, c] = e.combination.split("-");
      expect(a).not.toBe(b);
      expect(a).not.toBe(c);
      expect(b).not.toBe(c);
    }
  });

  test("all odds are positive numbers", () => {
    for (const e of entries) {
      expect(e.odds).toBeGreaterThan(0);
    }
  });

  test("every boat appears exactly 20 times as 1st", () => {
    for (let boat = 1; boat <= 6; boat++) {
      const count = entries.filter((e) =>
        e.combination.startsWith(`${boat}-`),
      ).length;
      expect(count).toBe(20);
    }
  });

  test("every boat appears exactly 20 times as 2nd", () => {
    for (let boat = 1; boat <= 6; boat++) {
      const count = entries.filter(
        (e) => e.combination.split("-")[1] === String(boat),
      ).length;
      expect(count).toBe(20);
    }
  });

  test("every boat appears exactly 20 times as 3rd", () => {
    for (let boat = 1; boat <= 6; boat++) {
      const count = entries.filter(
        (e) => e.combination.split("-")[2] === String(boat),
      ).length;
      expect(count).toBe(20);
    }
  });

  test("X-1-Y combinations exist for all non-1 first boats (4 each)", () => {
    for (let x = 2; x <= 6; x++) {
      const combos = entries.filter((e) => e.combination.startsWith(`${x}-1-`));
      expect(combos.length).toBe(4);
    }
  });

  test("popular 1-2-3 has lower odds than unpopular 6-5-4", () => {
    const popular = entries.find((e) => e.combination === "1-2-3");
    const unpopular = entries.find((e) => e.combination === "6-5-4");
    expect(popular).toBeDefined();
    expect(unpopular).toBeDefined();
    // biome-ignore lint/style/noNonNullAssertion: guarded by toBeDefined above
    expect(popular!.odds).toBeLessThan(unpopular!.odds); // biome-ignore lint/style/noNonNullAssertion: guarded
  });
});

describe("parseOdds2Tf — 2連単・2連複 (full race)", () => {
  const entries = parseOdds2Tf(loadFixture("odds2tf.html"));
  const exacta = entries.filter((e) => e.betType === "2連単");
  const quinella = entries.filter((e) => e.betType === "2連複");

  test("2連単 returns 30 combinations (6×5)", () => {
    expect(exacta).toHaveLength(30);
  });

  test("2連複 returns 15 combinations (6C2)", () => {
    expect(quinella).toHaveLength(15);
  });

  test("2連単 combinations are unique", () => {
    const combos = exacta.map((e) => e.combination);
    expect(new Set(combos).size).toBe(30);
  });

  test("2連単 format is X-Y, no duplicate boats", () => {
    for (const e of exacta) {
      expect(e.combination).toMatch(/^[1-6]-[1-6]$/);
      const [a, b] = e.combination.split("-");
      expect(a).not.toBe(b);
    }
  });

  test("2連複 format is X=Y with X<Y", () => {
    for (const e of quinella) {
      expect(e.combination).toMatch(/^[1-6]=[1-6]$/);
      const [a, b] = e.combination.split("=");
      expect(Number(a)).toBeLessThan(Number(b));
    }
  });

  test("all odds are positive", () => {
    for (const e of entries) {
      expect(e.odds).toBeGreaterThan(0);
    }
  });

  test("every boat appears as 1st in 2連単 exactly 5 times", () => {
    for (let boat = 1; boat <= 6; boat++) {
      const count = exacta.filter((e) =>
        e.combination.startsWith(`${boat}-`),
      ).length;
      expect(count).toBe(5);
    }
  });
});

describe("parseOdds2Tf — missing odds", () => {
  const entries = parseOdds2Tf(loadFixture("odds2tf-missing.html"));
  const exacta = entries.filter((e) => e.betType === "2連単");

  test("returns fewer than 30 when some odds are missing", () => {
    expect(exacta.length).toBeLessThan(30);
    expect(exacta.length).toBeGreaterThan(0);
  });

  test("no duplicate boats even with missing data", () => {
    for (const e of exacta) {
      const [a, b] = e.combination.split("-");
      expect(a).not.toBe(b);
    }
  });

  test("all returned odds are still positive", () => {
    for (const e of exacta) {
      expect(e.odds).toBeGreaterThan(0);
    }
  });
});
