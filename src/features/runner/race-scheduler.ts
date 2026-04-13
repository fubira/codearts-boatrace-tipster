/** Race scheduling and deadline-based state management. */

import { logger } from "@/shared/logger";

export interface RaceSlot {
  raceId: number;
  stadiumId: number;
  stadiumName: string;
  raceNumber: number;
  deadline: string; // "HH:MM" JST
  deadlineMs: number; // epoch ms (JST deadline converted to UTC epoch)
  status:
    | "waiting"
    | "active" // runner: DB data-driven processing
    | "before_info" // scrape-daemon: exhibition scraped
    | "predicted" // scrape-daemon: T-5 odds scraped
    | "decided" // both: bet decision made / T-1 odds scraped
    | "result_pending"
    | "done";
}

/** Bet decision stored during prediction. */
/** A single trifecta ticket as known at BET time.
 * Persisted with BetDecision so the result phase can detect refunds and
 * compute payouts even after a runner restart wipes the prediction cache. */
export interface BetTicketRecord {
  combo: string;
  modelProb: number;
  marketOdds: number;
  ev: number;
}

export interface BetDecision {
  raceId: number;
  stadiumName: string;
  raceNumber: number;
  boatNumber: number;
  prob: number;
  odds: number;
  ev: number;
  betAmount: number;
  recommend: boolean;
  /** Tickets the runner committed to at BET time (post-T-1 drift). */
  tickets: BetTicketRecord[];
}

// Timing constants (minutes relative to deadline)
export const ACTIVATE_LEAD = 7; // waiting → active
export const ODDS_LEAD = 1; // T-1 drift evaluation window
const SKIP_THRESHOLD = 5; // auto-skip if deadline passed by this much
const RESULT_DELAY = 12; // wait before checking results

/**
 * Parse "HH:MM" deadline string to epoch ms for a given date (JST).
 */
function deadlineToEpoch(raceDate: string, deadline: string): number {
  const iso = `${raceDate}T${deadline}:00+09:00`;
  return new Date(iso).getTime();
}

/**
 * Build race schedule from DB rows.
 */
export function buildSchedule(
  races: {
    id: number;
    stadium_id: number;
    race_number: number;
    deadline: string | null;
  }[],
  stadiumNames: Map<number, string>,
  raceDate: string,
): RaceSlot[] {
  const slots: RaceSlot[] = [];

  for (const r of races) {
    if (!r.deadline) {
      logger.warn(
        `No deadline for race ${r.id} (stadium ${r.stadium_id} R${r.race_number}), skipping`,
      );
      continue;
    }

    slots.push({
      raceId: r.id,
      stadiumId: r.stadium_id,
      stadiumName: stadiumNames.get(r.stadium_id) ?? `場${r.stadium_id}`,
      raceNumber: r.race_number,
      deadline: r.deadline,
      deadlineMs: deadlineToEpoch(raceDate, r.deadline),
      status: "waiting",
    });
  }

  return slots.sort((a, b) => a.deadlineMs - b.deadlineMs);
}

/** Categorize races for runner polling. */
export function getActiveRaces(
  schedule: RaceSlot[],
  now: number = Date.now(),
): {
  activate: RaceSlot[];
  active: RaceSlot[];
  results: RaceSlot[];
} {
  const activate: RaceSlot[] = [];
  const active: RaceSlot[] = [];
  const results: RaceSlot[] = [];

  for (const slot of schedule) {
    const minutesToDeadline = (slot.deadlineMs - now) / 60_000;

    switch (slot.status) {
      case "waiting":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          logger.warn(
            `Auto-skip: ${slot.stadiumName} R${slot.raceNumber} | deadline passed (${slot.status})`,
          );
          slot.status = "done";
        } else if (minutesToDeadline <= ACTIVATE_LEAD) {
          activate.push(slot);
        }
        break;
      case "active":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          logger.warn(
            `Auto-skip: ${slot.stadiumName} R${slot.raceNumber} | deadline passed (${slot.status})`,
          );
          slot.status = "done";
        } else {
          active.push(slot);
        }
        break;
      case "decided":
      case "result_pending":
        if (minutesToDeadline <= -RESULT_DELAY) {
          results.push(slot);
        }
        break;
    }
  }

  return { activate, active, results };
}

// Legacy function used by scrape-daemon and tests
export interface ActionableRaces {
  beforeInfo: RaceSlot[];
  predict: RaceSlot[];
  oddsT3: RaceSlot[];
  odds: RaceSlot[];
  results: RaceSlot[];
}

export const BEFORE_INFO_LEAD = 7;
export const PREDICT_LEAD = 5;
export const ODDS_T3_LEAD = 3;

export function getActionableRaces(
  schedule: RaceSlot[],
  now: number = Date.now(),
): ActionableRaces {
  const beforeInfo: RaceSlot[] = [];
  const predict: RaceSlot[] = [];
  const oddsT3: RaceSlot[] = [];
  const odds: RaceSlot[] = [];
  const results: RaceSlot[] = [];

  for (const slot of schedule) {
    const minutesToDeadline = (slot.deadlineMs - now) / 60_000;
    switch (slot.status) {
      case "waiting":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          logger.warn(
            `Auto-skip: ${slot.stadiumName} R${slot.raceNumber} | deadline passed (${slot.status})`,
          );
          slot.status = "done";
        } else if (minutesToDeadline <= BEFORE_INFO_LEAD) {
          beforeInfo.push(slot);
        }
        break;
      case "before_info":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          logger.warn(
            `Auto-skip: ${slot.stadiumName} R${slot.raceNumber} | deadline passed (${slot.status})`,
          );
          slot.status = "done";
        } else if (minutesToDeadline <= PREDICT_LEAD) {
          predict.push(slot);
        }
        break;
      case "predicted":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          logger.warn(
            `Auto-skip: ${slot.stadiumName} R${slot.raceNumber} | deadline passed (${slot.status})`,
          );
          slot.status = "done";
        } else if (minutesToDeadline <= ODDS_LEAD) {
          odds.push(slot);
        } else if (minutesToDeadline <= ODDS_T3_LEAD) {
          oddsT3.push(slot);
        }
        break;
      case "decided":
      case "result_pending":
        if (minutesToDeadline <= -RESULT_DELAY) {
          results.push(slot);
        }
        break;
    }
  }
  return { beforeInfo, predict, oddsT3, odds, results };
}

/**
 * Check if all races are done for the day.
 */
export function allDone(schedule: RaceSlot[]): boolean {
  return schedule.length > 0 && schedule.every((s) => s.status === "done");
}
