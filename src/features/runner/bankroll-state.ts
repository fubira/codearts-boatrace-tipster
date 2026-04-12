/** Persist bankroll across runner restarts.
 *
 * Short-term solution until the purchase history DB (see plan) is built.
 * Tracks current bankroll + all-time starting value for cumulative P/L.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";

const DEFAULT_STATE_PATH = resolve(config.dataDir, "runner-state.json");

export interface BankrollState {
  bankroll: number;
  allTimeInitial: number;
  startedAt: string; // first run ISO date
  lastUpdate: string; // ISO timestamp
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
