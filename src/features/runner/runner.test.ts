import { describe, expect, test } from "bun:test";
import { calcKellyBet } from "./runner";

describe("calcKellyBet", () => {
  const KELLY = 0.25;
  const CAP = 4000;

  test("returns positive bet for positive EV", () => {
    // prob=0.6, odds=2.0 → EV=0.2, kelly_full=0.2/1.0=0.2, frac=0.05
    const bet = calcKellyBet(0.6, 2.0, 50000, KELLY, CAP);
    expect(bet).toBeGreaterThan(0);
  });

  test("returns 0 for negative EV", () => {
    // prob=0.3, odds=2.0 → EV=-0.4
    const bet = calcKellyBet(0.3, 2.0, 50000, KELLY, CAP);
    expect(bet).toBe(0);
  });

  test("returns 0 for odds <= 1", () => {
    const bet = calcKellyBet(0.9, 1.0, 50000, KELLY, CAP);
    expect(bet).toBe(0);
  });

  test("rounds down to 100 yen units", () => {
    const bet = calcKellyBet(0.55, 2.5, 50000, KELLY, CAP);
    expect(bet % 100).toBe(0);
  });

  test("caps at betCap", () => {
    // Large bankroll should still cap
    const bet = calcKellyBet(0.7, 2.0, 1000000, KELLY, CAP);
    expect(bet).toBeLessThanOrEqual(CAP);
  });

  test("minimum bet is 100", () => {
    // Tiny bankroll
    const bet = calcKellyBet(0.55, 2.0, 5000, KELLY, CAP);
    expect(bet).toBeGreaterThanOrEqual(100);
  });

  test("returns 0 when bankroll below 100", () => {
    const bet = calcKellyBet(0.6, 2.0, 50, KELLY, CAP);
    expect(bet).toBe(0);
  });

  test("caps kelly fraction at 5% of bankroll", () => {
    // Very high edge → kelly_full large, but capped at 5%
    // prob=0.9, odds=3.0 → EV=1.7, kelly_full=0.85, frac=min(0.2125, 0.05)=0.05
    const bet = calcKellyBet(0.9, 3.0, 100000, KELLY, CAP);
    // 5% of 100000 = 5000, capped at 4000
    expect(bet).toBe(CAP);
  });

  test("concrete example: standard EV bet", () => {
    // prob=0.55, odds=2.5 → EV=0.375
    // kelly_full = 0.375 / 1.5 = 0.25, frac = 0.25 * 0.25 = 0.0625 → capped at 0.05
    // bet = floor(50000 * 0.05 / 100) * 100 = 2500
    const bet = calcKellyBet(0.55, 2.5, 50000, KELLY, CAP);
    expect(bet).toBe(2500);
  });
});
