/** Persist runner state across restarts.
 *
 * Tracks bankroll (long-lived, across days) and today's daily snapshot
 * (bets/results/skipCounts) so the daily Slack summary survives a restart
 * mid-day. The `today` snapshot is keyed by JST date and discarded on day
 * change; only bankroll persists across days.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";

const DEFAULT_STATE_PATH = resolve(config.dataDir, "runner-state.json");

/** Per-day runner counters. Reset on day boundary. */
export interface DailyRunnerSnapshot {
  /** YYYY-MM-DD in JST. Reset when a new day starts. */
  date: string;
  /** All bets placed today, JSON-serialized form of state.bets Map.
   * Decision shape mirrors BetDecision in race-scheduler.ts; kept inline to
   * avoid a circular import. Update both together. */
  bets: Array<{
    raceId: number;
    decision: {
      raceId: number;
      stadiumName: string;
      raceNumber: number;
      boatNumber: number;
      prob: number;
      odds: number;
      ev: number;
      betAmount: number;
      recommend: boolean;
      tickets: Array<{
        combo: string;
        modelProb: number;
        marketOdds: number;
        ev: number;
      }>;
    };
  }>;
  /** All race results today, JSON-serialized form of state.results Map. */
  results: Array<{
    raceId: number;
    won: boolean;
    payout: number;
  }>;
  /** Daily skip count breakdown. */
  skipCounts: {
    not_b1_top: number;
    gap12_low: number;
    top3_conc_low: number;
    gap23_low: number;
    no_ev_tickets: number;
    drift_drop: number;
    stadium_excluded: number;
    withdrawal: number;
  };
  /** Cumulative T-1 drift drops for the day. */
  t1DroppedTickets: number;
}

export interface BankrollState {
  bankroll: number;
  allTimeInitial: number;
  startedAt: string; // first run ISO date
  lastUpdate: string; // ISO timestamp
  /** Today's snapshot. Null/missing on first run or after day rollover. */
  today?: DailyRunnerSnapshot;
  /** Path this state is persisted to. Not serialized. */
  _path?: string;
}

function pathOf(state: BankrollState): string {
  return state._path ?? DEFAULT_STATE_PATH;
}

/** Load persisted bankroll state, or create fresh from CLI value. */
export function loadBankrollState(
  cliBankroll: number,
  statePath: string = DEFAULT_STATE_PATH,
): BankrollState {
  if (existsSync(statePath)) {
    try {
      const raw = readFileSync(statePath, "utf-8");
      const parsed = JSON.parse(raw) as BankrollState;
      const s: BankrollState = { ...parsed, _path: statePath };
      logger.info(
        `Loaded bankroll state: ¥${s.bankroll.toLocaleString()} (all-time initial ¥${s.allTimeInitial.toLocaleString()}, started ${s.startedAt})`,
      );
      return s;
    } catch (err) {
      logger.warn(
        `Failed to read ${statePath}: ${err instanceof Error ? err.message : String(err)}. Using CLI value.`,
      );
    }
  }
  const now = new Date().toISOString();
  const fresh: BankrollState = {
    bankroll: cliBankroll,
    allTimeInitial: cliBankroll,
    startedAt: now,
    lastUpdate: now,
    _path: statePath,
  };
  saveBankrollState(fresh);
  logger.info(`Initialized bankroll state: ¥${cliBankroll.toLocaleString()}`);
  return fresh;
}

export function saveBankrollState(state: BankrollState): void {
  const target = pathOf(state);
  try {
    mkdirSync(dirname(target), { recursive: true });
    const { _path, ...serializable } = state;
    writeFileSync(target, JSON.stringify(serializable, null, 2));
  } catch (err) {
    logger.error(
      `Failed to save bankroll state: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/** Update bankroll and persist. */
export function updateBankroll(
  state: BankrollState,
  newBankroll: number,
): void {
  state.bankroll = newBankroll;
  state.lastUpdate = new Date().toISOString();
  saveBankrollState(state);
}
