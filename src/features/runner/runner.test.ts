import { describe, expect, test } from "bun:test";
import { calcTrifectaUnit, formatSkipReason } from "./runner";

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

describe("formatSkipReason", () => {
  const opts = {
    evThreshold: 0.0,
    gap23Threshold: 0.13,
    top3ConcThreshold: 0.6,
  };

  test("not_b1_top shows predicted top1 boat", () => {
    expect(
      formatSkipReason({ skipReason: "not_b1_top", top1Boat: 3 }, opts),
    ).toBe("not_b1_top (top1=3号艇)");
  });

  test("not_b1_top without top1Boat falls back to plain reason", () => {
    expect(formatSkipReason({ skipReason: "not_b1_top" }, opts)).toBe(
      "not_b1_top",
    );
  });

  test("top3_conc_low shows value and threshold", () => {
    expect(
      formatSkipReason({ skipReason: "top3_conc_low", top3Conc: 0.58 }, opts),
    ).toBe("top3_conc_low (58% < th=60%)");
  });

  test("gap23_low shows value and threshold with 1 decimal", () => {
    expect(
      formatSkipReason({ skipReason: "gap23_low", gap23: 0.0821 }, opts),
    ).toBe("gap23_low (8.2% < th=13.0%)");
  });

  test("no_ev_tickets shows both conc and gap23 with ev threshold", () => {
    expect(
      formatSkipReason(
        { skipReason: "no_ev_tickets", top3Conc: 0.83, gap23: 0.15 },
        opts,
      ),
    ).toBe("no_ev_tickets (conc=83%, gap23=15.0%, all EV<0%)");
  });

  test("no_ev_tickets with only conc still shows partial info", () => {
    expect(
      formatSkipReason({ skipReason: "no_ev_tickets", top3Conc: 0.8 }, opts),
    ).toBe("no_ev_tickets (conc=80%, all EV<0%)");
  });

  test("unknown reason falls back to plain string", () => {
    expect(formatSkipReason({ skipReason: "unknown_reason" }, opts)).toBe(
      "unknown_reason",
    );
  });

  test("stadium_excluded returns fixed label", () => {
    expect(
      formatSkipReason({ skipReason: "stadium_excluded", stadiumId: 18 }, opts),
    ).toBe("stadium_excluded");
  });

  test("different threshold values are reflected", () => {
    const customOpts = {
      evThreshold: 0.05,
      gap23Threshold: 0.1,
      top3ConcThreshold: 0.65,
    };
    expect(
      formatSkipReason(
        { skipReason: "top3_conc_low", top3Conc: 0.62 },
        customOpts,
      ),
    ).toBe("top3_conc_low (62% < th=65%)");
    expect(
      formatSkipReason({ skipReason: "gap23_low", gap23: 0.08 }, customOpts),
    ).toBe("gap23_low (8.0% < th=10.0%)");
  });
});
