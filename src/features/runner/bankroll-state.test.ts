import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  type BankrollState,
  loadBankrollState,
  saveBankrollState,
  updateBankroll,
} from "./bankroll-state";

let tmpDir: string;
let statePath: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "bankroll-state-"));
  statePath = join(tmpDir, "runner-state.json");
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

describe("loadBankrollState", () => {
  test("creates fresh state when no file exists", () => {
    const state = loadBankrollState(70000, statePath);
    expect(state.bankroll).toBe(70000);
    expect(state.allTimeInitial).toBe(70000);
    expect(state.startedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(existsSync(statePath)).toBe(true);
  });

  test("fresh state is persisted to disk immediately", () => {
    loadBankrollState(50000, statePath);
    const disk = JSON.parse(readFileSync(statePath, "utf-8"));
    expect(disk.bankroll).toBe(50000);
    expect(disk.allTimeInitial).toBe(50000);
  });

  test("_path is not persisted to disk", () => {
    loadBankrollState(70000, statePath);
    const disk = JSON.parse(readFileSync(statePath, "utf-8"));
    expect("_path" in disk).toBe(false);
  });

  test("loads existing state, ignoring CLI bankroll", () => {
    const existing: BankrollState = {
      bankroll: 42000,
      allTimeInitial: 70000,
      startedAt: "2026-01-01T00:00:00.000Z",
      lastUpdate: "2026-04-11T12:00:00.000Z",
    };
    writeFileSync(statePath, JSON.stringify(existing));

    const loaded = loadBankrollState(99999, statePath);
    expect(loaded.bankroll).toBe(42000);
    expect(loaded.allTimeInitial).toBe(70000);
    expect(loaded.startedAt).toBe("2026-01-01T00:00:00.000Z");
  });

  test("falls back to CLI value on corrupt JSON", () => {
    writeFileSync(statePath, "{not json");
    const state = loadBankrollState(70000, statePath);
    expect(state.bankroll).toBe(70000);
    expect(state.allTimeInitial).toBe(70000);
  });
});

describe("updateBankroll", () => {
  test("persists new bankroll to disk", () => {
    const state = loadBankrollState(70000, statePath);
    updateBankroll(state, 65000);

    const disk = JSON.parse(readFileSync(statePath, "utf-8"));
    expect(disk.bankroll).toBe(65000);
    expect(disk.allTimeInitial).toBe(70000);
  });

  test("lastUpdate advances on update", async () => {
    const state = loadBankrollState(70000, statePath);
    const initialUpdate = state.lastUpdate;
    await new Promise((r) => setTimeout(r, 5));
    updateBankroll(state, 80000);
    expect(state.lastUpdate > initialUpdate).toBe(true);
  });

  test("allTimeInitial stays constant across updates", () => {
    const state = loadBankrollState(70000, statePath);
    updateBankroll(state, 60000);
    updateBankroll(state, 90000);
    updateBankroll(state, 50000);

    const reloaded = loadBankrollState(999999, statePath);
    expect(reloaded.allTimeInitial).toBe(70000);
    expect(reloaded.bankroll).toBe(50000);
  });

  test("roundtrip: write then read restores state", () => {
    const s1 = loadBankrollState(70000, statePath);
    updateBankroll(s1, 85000);

    const s2 = loadBankrollState(99999, statePath);
    expect(s2.bankroll).toBe(85000);
    expect(s2.allTimeInitial).toBe(70000);
    expect(s2.startedAt).toBe(s1.startedAt);
  });
});

describe("saveBankrollState", () => {
  test("creates parent directory if missing", () => {
    const deepPath = join(tmpDir, "nested", "dir", "state.json");
    const state: BankrollState = {
      bankroll: 1000,
      allTimeInitial: 1000,
      startedAt: "2026-01-01T00:00:00.000Z",
      lastUpdate: "2026-01-01T00:00:00.000Z",
      _path: deepPath,
    };
    saveBankrollState(state);
    expect(existsSync(deepPath)).toBe(true);
  });
});

describe("today snapshot persistence", () => {
  test("today snapshot survives save/load round trip", () => {
    const state = loadBankrollState(70000, statePath);
    state.today = {
      date: "2026-04-13",
      bets: [
        {
          raceId: 100,
          decision: {
            raceId: 100,
            stadiumName: "桐生",
            raceNumber: 5,
            boatNumber: 1,
            prob: 0,
            odds: 0,
            ev: 25,
            betAmount: 800,
            recommend: true,
            tickets: [
              { combo: "1-3-2", modelProb: 0.18, marketOdds: 12.4, ev: 0.67 },
              { combo: "1-2-3", modelProb: 0.12, marketOdds: 18.7, ev: 0.68 },
            ],
          },
        },
      ],
      results: [{ raceId: 100, won: false, payout: 0 }],
      skipCounts: {
        not_b1_top: 1,
        gap12_low: 2,
        top3_conc_low: 3,
        gap23_low: 4,
        no_ev_tickets: 5,
        drift_drop: 6,
        stadium_excluded: 7,
        withdrawal: 8,
      },
      t1DroppedTickets: 9,
    };
    saveBankrollState(state);

    const reloaded = loadBankrollState(99999, statePath);
    expect(reloaded.today).toBeDefined();
    expect(reloaded.today?.date).toBe("2026-04-13");
    expect(reloaded.today?.bets).toHaveLength(1);
    expect(reloaded.today?.bets[0].decision.stadiumName).toBe("桐生");
    expect(reloaded.today?.bets[0].decision.betAmount).toBe(800);
    expect(reloaded.today?.bets[0].decision.tickets).toHaveLength(2);
    expect(reloaded.today?.bets[0].decision.tickets[0].combo).toBe("1-3-2");
    expect(reloaded.today?.bets[0].decision.tickets[0].marketOdds).toBe(12.4);
    expect(reloaded.today?.results).toHaveLength(1);
    expect(reloaded.today?.results[0].won).toBe(false);
    expect(reloaded.today?.skipCounts.gap12_low).toBe(2);
    expect(reloaded.today?.skipCounts.withdrawal).toBe(8);
    expect(reloaded.today?.t1DroppedTickets).toBe(9);
  });

  test("absent today field loads as undefined (backward compat)", () => {
    const legacy: BankrollState = {
      bankroll: 50000,
      allTimeInitial: 70000,
      startedAt: "2026-01-01T00:00:00.000Z",
      lastUpdate: "2026-04-12T12:00:00.000Z",
    };
    writeFileSync(statePath, JSON.stringify(legacy));

    const loaded = loadBankrollState(99999, statePath);
    expect(loaded.bankroll).toBe(50000);
    expect(loaded.today).toBeUndefined();
  });
});
