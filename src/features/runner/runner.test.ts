import { describe, expect, test } from "bun:test";
import { calcTrifectaUnit } from "./runner";

describe("calcTrifectaUnit", () => {
  const CAP = 2000;

  test("basic calculation: 70000 / 800 = 87.5 → floor to ¥100", () => {
    expect(calcTrifectaUnit(70000, CAP)).toBe(100);
  });

  test("rounds down to 100 yen units", () => {
    // 160000 / 800 = 200
    expect(calcTrifectaUnit(160000, CAP)).toBe(200);
    // 200000 / 800 = 250 → floor(250/100)*100 = 200
    expect(calcTrifectaUnit(200000, CAP)).toBe(200);
  });

  test("caps at betCap", () => {
    // 2000000 / 800 = 2500 → capped at 2000
    expect(calcTrifectaUnit(2000000, CAP)).toBe(CAP);
  });

  test("minimum is ¥100", () => {
    // 10000 / 800 = 12.5 → floor = 0 → max(0, 100) = 100
    expect(calcTrifectaUnit(10000, CAP)).toBe(100);
  });

  test("returns 0 when bankroll is less than unit", () => {
    // bankroll=50 → unit would be 100 but 100 > 50
    expect(calcTrifectaUnit(50, CAP)).toBe(0);
  });

  test("returns ¥100 when bankroll exactly ¥100", () => {
    expect(calcTrifectaUnit(100, CAP)).toBe(100);
  });

  test("MAX unit with large bankroll", () => {
    // 1600000 / 800 = 2000 = CAP exactly
    expect(calcTrifectaUnit(1600000, CAP)).toBe(2000);
  });
});
