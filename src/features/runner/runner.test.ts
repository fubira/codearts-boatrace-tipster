import { describe, expect, test } from "bun:test";
import {
  calcTrifectaUnit,
  formatSkipReason,
  resultTag,
  ticketContainsRefundedBoat,
} from "./runner";

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

describe("ticketContainsRefundedBoat", () => {
  test("returns true when combo includes a refunded boat", () => {
    expect(ticketContainsRefundedBoat("1-3-2", new Set([3]))).toBe(true);
  });

  test("returns true when refunded boat is at any position", () => {
    expect(ticketContainsRefundedBoat("3-1-2", new Set([3]))).toBe(true);
    expect(ticketContainsRefundedBoat("1-2-3", new Set([3]))).toBe(true);
    expect(ticketContainsRefundedBoat("2-3-1", new Set([3]))).toBe(true);
  });

  test("returns false when no refunded boat is in the combo", () => {
    expect(ticketContainsRefundedBoat("1-4-5", new Set([3]))).toBe(false);
  });

  test("returns false with empty refunded set", () => {
    expect(ticketContainsRefundedBoat("1-2-3", new Set())).toBe(false);
  });

  test("handles multiple refunded boats", () => {
    expect(ticketContainsRefundedBoat("1-4-5", new Set([2, 3]))).toBe(false);
    expect(ticketContainsRefundedBoat("1-2-5", new Set([2, 3]))).toBe(true);
    expect(ticketContainsRefundedBoat("1-2-3", new Set([2, 3]))).toBe(true);
  });
});

describe("resultTag", () => {
  test("won → WIN regardless of refunds", () => {
    expect(resultTag(true, 0, 2)).toBe("WIN");
    expect(resultTag(true, 1, 2)).toBe("WIN");
  });

  test("not won, all tickets refunded → REFUND", () => {
    expect(resultTag(false, 2, 2)).toBe("REFUND");
    expect(resultTag(false, 1, 1)).toBe("REFUND");
  });

  test("not won, partial refund → LOSE", () => {
    expect(resultTag(false, 1, 2)).toBe("LOSE");
  });

  test("not won, no refund → LOSE", () => {
    expect(resultTag(false, 0, 2)).toBe("LOSE");
  });

  test("zero tickets is LOSE, never REFUND (cache-lost guard)", () => {
    // ticketCount=0 means the result phase has no ticket info (e.g. cache
    // was wiped by a restart and BetDecision was missing tickets). Must NOT
    // report REFUND because we cannot prove the bet was actually refunded.
    expect(resultTag(false, 0, 0)).toBe("LOSE");
  });
});

describe("formatSkipReason", () => {
  const opts = {
    evThreshold: 0.0,
    gap23Threshold: 0.13,
    top3ConcThreshold: 0.6,
    gap12MinThreshold: 0.04,
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

  test("gap12_low shows value and threshold with 1 decimal", () => {
    expect(
      formatSkipReason({ skipReason: "gap12_low", gap12: 0.028 }, opts),
    ).toBe("gap12_low (2.8% < th=4.0%)");
  });

  test("gap12_low appends would-be tickets when provided", () => {
    expect(
      formatSkipReason(
        {
          skipReason: "gap12_low",
          gap12: 0.028,
          wouldBeTickets: [
            { combo: "1-2-3", modelProb: 0.12, marketOdds: 15.2, ev: 0.37 },
            { combo: "1-3-2", modelProb: 0.08, marketOdds: 22.0, ev: 0.32 },
          ],
        },
        opts,
      ),
    ).toBe(
      "gap12_low (2.8% < th=4.0%) | cut: 1-2-3(EV 37% @15.2), 1-3-2(EV 32% @22.0)",
    );
  });

  test("cut display marks missing odds as n/a", () => {
    expect(
      formatSkipReason(
        {
          skipReason: "top3_conc_low",
          top3Conc: 0.55,
          wouldBeTickets: [
            { combo: "1-2-3", modelProb: 0.12, marketOdds: null, ev: null },
          ],
        },
        opts,
      ),
    ).toBe("top3_conc_low (55% < th=60%) | cut: 1-2-3(odds=n/a)");
  });

  test("no_ev_tickets shows cut tickets with their actual EVs", () => {
    expect(
      formatSkipReason(
        {
          skipReason: "no_ev_tickets",
          top3Conc: 0.83,
          gap23: 0.15,
          wouldBeTickets: [
            { combo: "1-2-3", modelProb: 0.1, marketOdds: 5.0, ev: -0.25 },
            { combo: "1-3-2", modelProb: 0.05, marketOdds: 8.0, ev: -0.7 },
          ],
        },
        opts,
      ),
    ).toBe(
      "no_ev_tickets (conc=83%, gap23=15.0%, all EV<0%) | cut: 1-2-3(EV -25% @5.0), 1-3-2(EV -70% @8.0)",
    );
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

  test("withdrawal shows withdrawn boat numbers", () => {
    expect(
      formatSkipReason({ skipReason: "withdrawal", withdrawnBoats: [2] }, opts),
    ).toBe("withdrawal (2号艇欠場)");
  });

  test("withdrawal with multiple boats joins them", () => {
    expect(
      formatSkipReason(
        { skipReason: "withdrawal", withdrawnBoats: [2, 5] },
        opts,
      ),
    ).toBe("withdrawal (2号艇,5号艇欠場)");
  });

  test("different threshold values are reflected", () => {
    const customOpts = {
      evThreshold: 0.05,
      gap23Threshold: 0.1,
      top3ConcThreshold: 0.65,
      gap12MinThreshold: 0.04,
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
