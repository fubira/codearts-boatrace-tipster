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
