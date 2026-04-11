import { describe, expect, test } from "bun:test";
import { calcTrifectaUnit } from "./runner";

describe("calcTrifectaUnit", () => {
  const CAP = 30000;
  const DIV = 150;

  test("basic calculation: 70000 / 150 = 466 → floor to ¥400", () => {
    expect(calcTrifectaUnit(70000, CAP, DIV)).toBe(400);
  });

  test("rounds down to 100 yen units", () => {
    // 150000 / 150 = 1000
    expect(calcTrifectaUnit(150000, CAP, DIV)).toBe(1000);
    // 200000 / 150 = 1333 → floor(1333/100)*100 = 1300
    expect(calcTrifectaUnit(200000, CAP, DIV)).toBe(1300);
  });

  test("caps at betCap", () => {
    // 10000000 / 150 = 66666 → capped at 30000
    expect(calcTrifectaUnit(10000000, CAP, DIV)).toBe(CAP);
  });

  test("minimum is ¥100", () => {
    // 10000 / 150 = 66 → floor = 0 → max(0, 100) = 100
    expect(calcTrifectaUnit(10000, CAP, DIV)).toBe(100);
  });

  test("returns 0 when bankroll is less than unit", () => {
    expect(calcTrifectaUnit(50, CAP, DIV)).toBe(0);
  });

  test("returns ¥100 when bankroll exactly ¥100", () => {
    expect(calcTrifectaUnit(100, CAP, DIV)).toBe(100);
  });

  test("MAX unit with large bankroll", () => {
    // 4500000 / 150 = 30000 = CAP exactly
    expect(calcTrifectaUnit(4500000, CAP, DIV)).toBe(30000);
  });
});
