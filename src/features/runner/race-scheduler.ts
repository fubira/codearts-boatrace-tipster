/** Race scheduling and deadline-based state management. */

import { logger } from "@/shared/logger";

export interface RaceSlot {
  raceId: number;
  stadiumId: number;
  stadiumName: string;
  raceNumber: number;
  deadline: string; // "HH:MM" JST
  deadlineMs: number; // epoch ms (JST deadline converted to UTC epoch)
  status: "waiting" | "before_info" | "predicted" | "result_pending" | "done";
}

/** Bet decision stored during prediction. */
export interface BetDecision {
  raceId: number;
  stadiumName: string;
  raceNumber: number;
  prob: number;
  odds: number;
  ev: number;
  betAmount: number;
  recommend: boolean;
}

// Minutes before deadline to trigger each action
const BEFORE_INFO_LEAD = 10;
const PREDICT_LEAD = 3;
// Minutes after deadline to check for results
const RESULT_DELAY = 10;
// If deadline passed by this many minutes and still "waiting", skip entirely
const SKIP_THRESHOLD = 5;

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

export interface ActionableRaces {
  beforeInfo: RaceSlot[];
  predict: RaceSlot[];
  results: RaceSlot[];
}

/**
 * Determine which races need action based on current time.
 */
export function getActionableRaces(
  schedule: RaceSlot[],
  now: number = Date.now(),
): ActionableRaces {
  const beforeInfo: RaceSlot[] = [];
  const predict: RaceSlot[] = [];
  const results: RaceSlot[] = [];

  for (const slot of schedule) {
    const minutesToDeadline = (slot.deadlineMs - now) / 60_000;

    switch (slot.status) {
      case "waiting":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          // Late start: deadline already passed, skip this race
          slot.status = "done";
        } else if (minutesToDeadline <= BEFORE_INFO_LEAD) {
          beforeInfo.push(slot);
        }
        break;
      case "before_info":
        if (minutesToDeadline <= -SKIP_THRESHOLD) {
          slot.status = "done";
        } else if (minutesToDeadline <= PREDICT_LEAD) {
          predict.push(slot);
        }
        break;
      case "predicted":
      case "result_pending":
        // Check for results after deadline + delay
        if (minutesToDeadline <= -RESULT_DELAY) {
          results.push(slot);
        }
        break;
    }
  }

  return { beforeInfo, predict, results };
}

/**
 * Check if all races are done for the day.
 */
export function allDone(schedule: RaceSlot[]): boolean {
  return schedule.length > 0 && schedule.every((s) => s.status === "done");
}
