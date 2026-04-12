/** Persist bankroll across runner restarts.
 *
 * Short-term solution until the purchase history DB (see plan) is built.
 * Tracks current bankroll + all-time starting value for cumulative P/L.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";

const STATE_PATH = resolve(config.dataDir, "runner-state.json");

export interface BankrollState {
  bankroll: number;
  allTimeInitial: number;
  startedAt: string; // first run ISO date
  lastUpdate: string; // ISO timestamp
}

/** Load persisted bankroll state, or create fresh from CLI value. */
export function loadBankrollState(cliBankroll: number): BankrollState {
  if (existsSync(STATE_PATH)) {
    try {
      const raw = readFileSync(STATE_PATH, "utf-8");
      const s = JSON.parse(raw) as BankrollState;
      logger.info(
        `Loaded bankroll state: ¥${s.bankroll.toLocaleString()} (all-time initial ¥${s.allTimeInitial.toLocaleString()}, started ${s.startedAt})`,
      );
      return s;
    } catch (err) {
      logger.warn(
        `Failed to read ${STATE_PATH}: ${err instanceof Error ? err.message : String(err)}. Using CLI value.`,
      );
    }
  }
  const now = new Date().toISOString();
  const fresh: BankrollState = {
    bankroll: cliBankroll,
    allTimeInitial: cliBankroll,
    startedAt: now,
    lastUpdate: now,
  };
  saveBankrollState(fresh);
  logger.info(`Initialized bankroll state: ¥${cliBankroll.toLocaleString()}`);
  return fresh;
}

export function saveBankrollState(state: BankrollState): void {
  try {
    mkdirSync(dirname(STATE_PATH), { recursive: true });
    writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));
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
