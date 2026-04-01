/** Main orchestrator for boatrace.jp scraping */

import { isCacheEnabled } from "@/features/scraper/cache-manager";
import { fetchPage } from "@/features/scraper/http-client";
import type {
  Scraper,
  ScraperOptions,
  ScraperResult,
} from "@/features/scraper/types";
import { logger } from "@/shared/logger";
import {
  COOLDOWN_BETWEEN_PAGES_MS,
  COOLDOWN_BETWEEN_VENUES_MS,
  MAX_CONCURRENCY,
  MAX_RACES_PER_VENUE,
  type RaceParams,
  STADIUMS,
  beforeInfoUrl,
  raceListUrl,
  raceResultUrl,
} from "./constants";
import { discoverDateSchedule, discoverMonthSchedule } from "./discovery";
import type { RaceContext } from "./parsers";
import { parseBeforeInfo, parseRaceList, parseRaceResult } from "./parsers";

import type {
  BeforeInfoData,
  RaceData,
  RaceResultData,
} from "@/features/database";

function buildRaceDate(yyyymmdd: string): string {
  return `${yyyymmdd.slice(0, 4)}-${yyyymmdd.slice(4, 6)}-${yyyymmdd.slice(6, 8)}`;
}

async function scrapeOneRace(
  params: RaceParams,
  shouldSkip?: ScraperOptions["shouldSkip"],
): Promise<{
  race: RaceData | null;
  result: RaceResultData | null;
  beforeInfo: BeforeInfoData | null;
  skipped: boolean;
}> {
  const stadiumId = Number.parseInt(params.stadiumCode, 10);
  const raceDate = buildRaceDate(params.date);
  const context: RaceContext = { params, raceDate };

  if (shouldSkip?.(stadiumId, raceDate, params.raceNumber)) {
    logger.debug(
      `Skip ${STADIUMS[params.stadiumCode] ?? params.stadiumCode} R${params.raceNumber} (already scraped)`,
    );
    return { race: null, result: null, beforeInfo: null, skipped: true };
  }

  // Fetch all 3 pages in parallel
  const [raceListPage, beforeInfoPage, resultPage] = await Promise.all([
    fetchPage(raceListUrl(params)),
    fetchPage(beforeInfoUrl(params)),
    fetchPage(raceResultUrl(params)),
  ]);

  const race = parseRaceList(raceListPage.html, context);
  const beforeInfo = parseBeforeInfo(beforeInfoPage.html, context);
  const result = parseRaceResult(resultPage.html, context);

  const anyFetched =
    !raceListPage.fromCache ||
    !beforeInfoPage.fromCache ||
    !resultPage.fromCache;
  if (anyFetched) await Bun.sleep(COOLDOWN_BETWEEN_PAGES_MS);

  return { race, result, beforeInfo, skipped: false };
}

/** Scrape all 12 races for a single venue-day */
async function scrapeVenueDay(
  stadiumCode: string,
  date: string,
  options: ScraperOptions,
): Promise<ScraperResult & { scraped: number; skipped: number }> {
  const venueName = STADIUMS[stadiumCode] ?? stadiumCode;
  const races: RaceData[] = [];
  const results: RaceResultData[] = [];
  const beforeInfo: BeforeInfoData[] = [];
  let skipped = 0;

  for (let rno = 1; rno <= MAX_RACES_PER_VENUE; rno++) {
    if (options.raceNumbers && !options.raceNumbers.includes(rno)) continue;
    const params: RaceParams = { raceNumber: rno, stadiumCode, date };
    const result = await scrapeOneRace(params, options.shouldSkip);

    if (result.skipped) {
      skipped++;
      continue;
    }

    if (result.race) races.push(result.race);
    if (result.result) results.push(result.result);
    if (result.beforeInfo) beforeInfo.push(result.beforeInfo);
  }

  const scraped = MAX_RACES_PER_VENUE - skipped;
  logger.info(
    `${venueName} ${buildRaceDate(date)}: ${scraped} scraped, ${skipped} skipped`,
  );

  return { races, results, beforeInfo, scraped, skipped };
}

async function resolveVenueDays(
  options: ScraperOptions,
): Promise<{ stadiumCode: string; date: string }[]> {
  if (options.date) {
    const yyyymmdd = options.date.replace(/-/g, "");
    const venues = await discoverDateSchedule(options.date);

    if (options.stadiumId) {
      return venues
        .filter((v) => v.stadiumCode === options.stadiumId)
        .map((v) => ({ ...v, date: yyyymmdd }));
    }
    return venues.map((v) => ({ ...v, date: yyyymmdd }));
  }

  if (options.month) {
    const venues = await discoverMonthSchedule(options.month);
    if (options.stadiumId) {
      return venues.filter((v) => v.stadiumCode === options.stadiumId);
    }
    return venues;
  }

  if (options.year) {
    const allVenues: { stadiumCode: string; date: string }[] = [];
    const months = Array.from(
      { length: 12 },
      (_, i) => `${options.year}${String(i + 1).padStart(2, "0")}`,
    );
    for (const month of months) {
      logger.info(`Fetching schedule for ${month}...`);
      const venues = await discoverMonthSchedule(month);
      allVenues.push(...venues);
      if (!isCacheEnabled()) await Bun.sleep(COOLDOWN_BETWEEN_PAGES_MS);
    }
    if (options.stadiumId) {
      return allVenues.filter((v) => v.stadiumCode === options.stadiumId);
    }
    return allVenues;
  }

  throw new Error(
    "--date, --month, or --year is required for boatrace scraper",
  );
}

/** Run async tasks with concurrency limit */
async function parallelMap<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = [];
  let index = 0;

  async function worker(): Promise<void> {
    while (index < items.length) {
      const i = index++;
      results[i] = await fn(items[i]);
    }
  }

  const workers = Array.from(
    { length: Math.min(concurrency, items.length) },
    () => worker(),
  );
  await Promise.all(workers);
  return results;
}

export const boatraceScraper: Scraper = {
  name: "boatrace",
  description: "boatrace.jp 公式サイト",

  async scrape(options: ScraperOptions): Promise<ScraperResult> {
    const venueDays = await resolveVenueDays(options);

    if (venueDays.length === 0) {
      logger.warn("No venues found for the specified period");
      return { races: [], results: [], beforeInfo: [] };
    }

    const totalRaces = venueDays.length * MAX_RACES_PER_VENUE;
    logger.info(
      `Found ${venueDays.length} venue-day(s), up to ${totalRaces} races (concurrency: ${MAX_CONCURRENCY})`,
    );

    // Only accumulate results when no onBatchComplete callback (e.g. dry-run)
    const accumulate = !options.onBatchComplete;
    const allRaces: RaceData[] = [];
    const allResults: RaceResultData[] = [];
    const allBeforeInfo: BeforeInfoData[] = [];
    let completedVenues = 0;
    let totalRacesScraped = 0;

    await parallelMap(venueDays, MAX_CONCURRENCY, async (vd) => {
      const batch = await scrapeVenueDay(vd.stadiumCode, vd.date, options);

      if (accumulate) {
        allRaces.push(...batch.races);
        allResults.push(...batch.results);
        allBeforeInfo.push(...batch.beforeInfo);
      } else if (batch.races.length > 0 || batch.results.length > 0) {
        options.onBatchComplete?.({
          races: batch.races,
          results: batch.results,
          beforeInfo: batch.beforeInfo,
        });
      }

      completedVenues++;
      totalRacesScraped += batch.races.length;

      // Cooldown between venues (per-worker)
      if (batch.scraped > 0 && !isCacheEnabled()) {
        await Bun.sleep(COOLDOWN_BETWEEN_VENUES_MS);
      }
    });

    logger.info(
      `Completed: ${completedVenues} venue-day(s), ${totalRacesScraped} races`,
    );

    return { races: allRaces, results: allResults, beforeInfo: allBeforeInfo };
  },
};
