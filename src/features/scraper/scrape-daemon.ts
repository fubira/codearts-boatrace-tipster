/** Scraping daemon — collects race data independently from runner. */

import { resolve } from "node:path";
import {
  type OddsTiming,
  closeDatabase,
  getDatabase,
  initializeDatabase,
  saveBeforeInfo,
  saveRaceResults,
  saveRaces,
} from "@/features/database";
import { type RaceSlot, buildSchedule } from "@/features/runner/race-scheduler";
import {
  fetchTrifectaOdds,
  scrapeBeforeInfoForRace,
  scrapeResultForRace,
} from "@/features/runner/scrape-helpers";
import {
  disableCacheRead,
  enableCache,
  enableCacheRead,
} from "@/features/scraper/cache-manager";
import { getScraper } from "@/features/scraper/registry";
import { fetchAndSaveBoatcast } from "@/features/scraper/sources/boatcast/fetcher";
import { STADIUMS } from "@/features/scraper/sources/boatrace/constants";
import { discoverDateSchedule } from "@/features/scraper/sources/boatrace/discovery";
import { config } from "@/shared/config";
import { enableFileLog, logger } from "@/shared/logger";

const POLL_INTERVAL_MS = 30_000;
const DISCOVER_RETRY_INTERVAL_MS = 5 * 60_000;
const DISCOVER_MAX_RETRIES = 6;

// Minutes before/after deadline for each action
const BEFORE_INFO_LEAD = 7;
const ODDS_T5_LEAD = 5;
const ODDS_T3_LEAD = 3;
const ODDS_T1_LEAD = 1;
const SKIP_THRESHOLD = 5;
const RESULT_DELAY = 12;

const stadiumNames = new Map(
  Object.entries(STADIUMS).map(([code, name]) => [
    Number.parseInt(code, 10),
    name,
  ]),
);

interface ScrapeState {
  schedule: RaceSlot[];
  oddsTimings: Map<number, Set<OddsTiming>>;
  date: string;
  lastStatusLine: string;
}

function todayJST(): string {
  return new Date()
    .toLocaleDateString("sv-SE", { timeZone: "Asia/Tokyo" })
    .replace(/\//g, "-");
}

/** Determine which races need scraping based on current time. */
function getScrapableRaces(
  schedule: RaceSlot[],
  now: number,
): {
  beforeInfo: RaceSlot[];
  oddsT5: RaceSlot[];
  oddsT3: RaceSlot[];
  oddsT1: RaceSlot[];
  results: RaceSlot[];
} {
  const beforeInfo: RaceSlot[] = [];
  const oddsT5: RaceSlot[] = [];
  const oddsT3: RaceSlot[] = [];
  const oddsT1: RaceSlot[] = [];
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
        } else if (minutesToDeadline <= ODDS_T1_LEAD) {
          oddsT1.push(slot);
        } else if (minutesToDeadline <= ODDS_T3_LEAD) {
          oddsT3.push(slot);
        } else if (minutesToDeadline <= ODDS_T5_LEAD) {
          oddsT5.push(slot);
        }
        break;
      case "predicted": // odds_done — waiting for results
      case "decided":
      case "result_pending":
        if (minutesToDeadline <= -RESULT_DELAY) {
          results.push(slot);
        }
        break;
    }
  }

  return { beforeInfo, oddsT5, oddsT3, oddsT1, results };
}

async function setupDay(): Promise<ScrapeState | null> {
  const date = todayJST();
  const yyyymmdd = date.replace(/-/g, "");

  logger.info(`[SCRAPER] Starting for ${date}`);

  // Discover venues
  let venueCodes: { stadiumCode: string; date: string }[] = [];
  for (let attempt = 0; attempt <= DISCOVER_MAX_RETRIES; attempt++) {
    if (attempt > 0) disableCacheRead();
    venueCodes = discoverDateSchedule(yyyymmdd);
    if (attempt > 0) enableCacheRead();
    if (venueCodes.length > 0) break;
    if (attempt < DISCOVER_MAX_RETRIES) {
      logger.warn(
        `No venues found yet (attempt ${attempt + 1}/${DISCOVER_MAX_RETRIES + 1}), retrying in 5 min...`,
      );
      await Bun.sleep(DISCOVER_RETRY_INTERVAL_MS);
    }
  }

  if (venueCodes.length === 0) {
    logger.warn("[SCRAPER] No venues found after retries");
    return null;
  }

  // Scrape race lists
  logger.info(
    `[SCRAPER] Found ${venueCodes.length} venues, scraping race lists...`,
  );
  const scraper = getScraper("boatrace");
  if (!scraper) {
    logger.error("[SCRAPER] Scraper 'boatrace' not found");
    return null;
  }

  let totalScraped = 0;
  scraper.scrape({
    date: yyyymmdd,
    skipResults: true,
    onBatchComplete: (batch) => {
      if (batch.races.length > 0) saveRaces(batch.races);
      if (batch.results.length > 0) saveRaceResults(batch.results);
      if (batch.beforeInfo.length > 0) saveBeforeInfo(batch.beforeInfo);
      totalScraped += batch.races.length;
    },
  });
  logger.info(`[SCRAPER] Scraped ${totalScraped} races`);

  // Build schedule
  const db = getDatabase();
  const races = db
    .query(
      `SELECT id, stadium_id, race_number, deadline FROM races
       WHERE race_date = ? ORDER BY deadline, stadium_id, race_number`,
    )
    .all(date) as {
    id: number;
    stadium_id: number;
    race_number: number;
    deadline: string | null;
  }[];

  const schedule = buildSchedule(races, stadiumNames, date);
  if (schedule.length > 0) {
    const venues = new Set(schedule.map((s) => s.stadiumName));
    const first = schedule[0];
    const last = schedule[schedule.length - 1];
    logger.info(
      `[SCRAPER] Schedule: ${schedule.length}R / ${venues.size} venues | ${first.deadline} ~ ${last.deadline}`,
    );
  } else {
    logger.info("[SCRAPER] Schedule: 0 races");
  }

  return {
    schedule,
    oddsTimings: new Map(),
    date,
    lastStatusLine: "",
  };
}

async function poll(state: ScrapeState): Promise<void> {
  const now = Date.now();
  const actionable = getScrapableRaces(state.schedule, now);

  // Status log
  const counts = { waiting: 0, active: 0, done: 0 };
  for (const s of state.schedule) {
    if (s.status === "waiting") counts.waiting++;
    else if (s.status === "done") counts.done++;
    else counts.active++;
  }
  const statusLine = `${counts.waiting}/${counts.active}/${counts.done}`;
  if (statusLine !== state.lastStatusLine) {
    logger.info(
      `[SCRAPER] Status: ${state.schedule.length}R | wait:${counts.waiting} active:${counts.active} done:${counts.done}`,
    );
    state.lastStatusLine = statusLine;
  }

  // 1. Before-info + BOATCAST
  if (actionable.beforeInfo.length > 0) {
    let scraped = 0;
    let bcFetched = 0;
    for (const slot of actionable.beforeInfo) {
      try {
        const ok = scrapeBeforeInfoForRace(slot, state.date);
        if (ok) scraped++;
      } catch {
        // Non-critical
      }
      try {
        const bc = await fetchAndSaveBoatcast(
          slot.stadiumId,
          state.date,
          slot.raceNumber,
        );
        if (bc) bcFetched++;
      } catch {
        // Non-critical
      }
      slot.status = "before_info";
    }
    logger.info(
      `[SCRAPER] Before-info: ${scraped}/${actionable.beforeInfo.length} scraped, BOATCAST: ${bcFetched}`,
    );
  }

  // 2. Odds snapshots (T-5, T-3, T-1)
  if (actionable.oddsT5.length > 0) {
    const fetched = fetchTrifectaOdds(
      actionable.oddsT5,
      state.date,
      "T-5",
      state.oddsTimings,
    );
    if (fetched > 0) {
      logger.info(
        `[SCRAPER] Odds (T-5): ${fetched}/${actionable.oddsT5.length} fetched`,
      );
    }
  }
  if (actionable.oddsT3.length > 0) {
    const fetched = fetchTrifectaOdds(
      actionable.oddsT3,
      state.date,
      "T-3",
      state.oddsTimings,
    );
    if (fetched > 0) {
      logger.info(
        `[SCRAPER] Odds (T-3): ${fetched}/${actionable.oddsT3.length} fetched`,
      );
    }
  }
  if (actionable.oddsT1.length > 0) {
    const fetched = fetchTrifectaOdds(
      actionable.oddsT1,
      state.date,
      "T-1",
      state.oddsTimings,
    );
    if (fetched > 0) {
      logger.info(
        `[SCRAPER] Odds (T-1): ${fetched}/${actionable.oddsT1.length} fetched`,
      );
    }
    // Transition to result-pending after T-1
    for (const slot of actionable.oddsT1) {
      slot.status = "decided";
    }
  }

  // 3. Results + final odds
  if (actionable.results.length > 0) {
    const finalFetched = fetchTrifectaOdds(
      actionable.results,
      state.date,
      "final",
      state.oddsTimings,
    );
    if (finalFetched > 0) {
      logger.info(
        `[SCRAPER] Odds (final): ${finalFetched}/${actionable.results.length} fetched`,
      );
    }

    for (const slot of actionable.results) {
      try {
        const result = scrapeResultForRace(slot, state.date);
        if (result === null) {
          slot.status = "result_pending";
          continue;
        }
        logger.info(
          `[SCRAPER] Result: ${slot.stadiumName} R${slot.raceNumber} | scraped`,
        );
        slot.status = "done";
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(
          `[SCRAPER] Result error: ${slot.stadiumName} R${slot.raceNumber} | ${msg}`,
        );
        slot.status = "result_pending";
      }
    }
  }
}

function allDone(schedule: RaceSlot[]): boolean {
  return schedule.length > 0 && schedule.every((s) => s.status === "done");
}

export async function runScrapeDaemon(): Promise<void> {
  enableFileLog(resolve(config.projectRoot, "logs"), "scraper");
  enableCache();
  initializeDatabase();

  const state = await setupDay();
  if (!state) {
    closeDatabase();
    return;
  }

  function shutdown(): void {
    logger.info("[SCRAPER] Shutting down...");
    closeDatabase();
    process.exit(0);
  }

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  while (true) {
    await poll(state);

    let dayDone = false;
    while (!dayDone) {
      await Bun.sleep(POLL_INTERVAL_MS);
      try {
        await poll(state);
        if (allDone(state.schedule)) {
          logger.info("[SCRAPER] All races done for today");
          dayDone = true;
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(`[SCRAPER] Poll error: ${msg}`);
      }
    }

    // Sleep until next day 7:00 JST
    const now = Date.now();
    const jstNow = now + 9 * 3600_000;
    const jstMidnight = jstNow - (jstNow % (24 * 3600_000));
    const jstHourMs = jstNow - jstMidnight;
    const jst7am = 7 * 3600_000;
    const target =
      jstHourMs >= jst7am
        ? jstMidnight + 24 * 3600_000 + jst7am
        : jstMidnight + jst7am;
    const sleepMs = target - 9 * 3600_000 - now;

    logger.info(
      `[SCRAPER] Sleeping until tomorrow 07:00 JST (${Math.round(sleepMs / 3600_000)}h)`,
    );
    await Bun.sleep(sleepMs);

    // Restart for next day
    logger.info("[SCRAPER] New day starting...");
    closeDatabase();
    initializeDatabase();

    const newState = await setupDay();
    if (!newState) continue;

    Object.assign(state, newState);
  }
}
