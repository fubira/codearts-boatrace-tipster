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

interface ScrapeOneRaceResult {
  race: RaceData | null;
  result: RaceResultData | null;
  beforeInfo: BeforeInfoData | null;
  skipped: boolean;
  cacheHits: number;
  cacheMisses: number;
}

function scrapeOneRace(
  params: RaceParams,
  shouldSkip?: ScraperOptions["shouldSkip"],
  skipResults?: boolean,
): ScrapeOneRaceResult {
  const stadiumId = Number.parseInt(params.stadiumCode, 10);
  const raceDate = buildRaceDate(params.date);
  const context: RaceContext = { params, raceDate };

  if (shouldSkip?.(stadiumId, raceDate, params.raceNumber)) {
    logger.debug(
      `Skip ${STADIUMS[params.stadiumCode] ?? params.stadiumCode} R${params.raceNumber} (already scraped)`,
    );
    return {
      race: null,
      result: null,
      beforeInfo: null,
      skipped: true,
      cacheHits: 0,
      cacheMisses: 0,
    };
  }

  // Fetch pages sequentially (curl-based, synchronous)
  const raceListPage = fetchPage(raceListUrl(params));
  const beforeInfoPage = fetchPage(beforeInfoUrl(params));
  const resultPage = skipResults ? null : fetchPage(raceResultUrl(params));

  // Count cache stats from non-null results
  const pages = [raceListPage, beforeInfoPage, resultPage];
  const cacheHits = pages.filter((p) => p?.fromCache).length;
  const cacheMisses = pages.filter((p) => p !== null && !p.fromCache).length;

  const race = raceListPage ? parseRaceList(raceListPage.html, context) : null;
  const beforeInfo = beforeInfoPage
    ? parseBeforeInfo(beforeInfoPage.html, context)
    : null;
  const result = resultPage ? parseRaceResult(resultPage.html, context) : null;

  const anyFetched = pages.some((p) => p !== null && !p.fromCache);
  if (anyFetched) Bun.sleepSync(COOLDOWN_BETWEEN_PAGES_MS);

  return { race, result, beforeInfo, skipped: false, cacheHits, cacheMisses };
}

interface VenueDayResult extends ScraperResult {
  scraped: number;
  skipped: number;
  cacheHits: number;
  cacheMisses: number;
}

/** Scrape all 12 races for a single venue-day */
function scrapeVenueDay(
  stadiumCode: string,
  date: string,
  options: ScraperOptions,
): VenueDayResult {
  const venueName = STADIUMS[stadiumCode] ?? stadiumCode;
  const races: RaceData[] = [];
  const results: RaceResultData[] = [];
  const beforeInfo: BeforeInfoData[] = [];
  let skipped = 0;
  let cacheHits = 0;
  let cacheMisses = 0;

  for (let rno = 1; rno <= MAX_RACES_PER_VENUE; rno++) {
    if (options.raceNumbers && !options.raceNumbers.includes(rno)) continue;
    const params: RaceParams = { raceNumber: rno, stadiumCode, date };
    const result = scrapeOneRace(
      params,
      options.shouldSkip,
      options.skipResults,
    );

    cacheHits += result.cacheHits;
    cacheMisses += result.cacheMisses;

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

  return {
    races,
    results,
    beforeInfo,
    scraped,
    skipped,
    cacheHits,
    cacheMisses,
  };
}

function resolveVenueDays(
  options: ScraperOptions,
): { stadiumCode: string; date: string }[] {
  if (options.date) {
    const yyyymmdd = options.date.replace(/-/g, "");
    const venues = discoverDateSchedule(options.date);

    if (options.stadiumId) {
      return venues
        .filter((v) => v.stadiumCode === options.stadiumId)
        .map((v) => ({ ...v, date: yyyymmdd }));
    }
    return venues.map((v) => ({ ...v, date: yyyymmdd }));
  }

  if (options.month) {
    const venues = discoverMonthSchedule(options.month);
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
      const venues = discoverMonthSchedule(month);
      allVenues.push(...venues);
      if (!isCacheEnabled()) Bun.sleepSync(COOLDOWN_BETWEEN_PAGES_MS);
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

const PROGRESS_LOG_INTERVAL = 50;

export const boatraceScraper: Scraper = {
  name: "boatrace",
  description: "boatrace.jp 公式サイト",

  scrape(options: ScraperOptions): ScraperResult {
    const venueDays = resolveVenueDays(options);

    if (venueDays.length === 0) {
      logger.warn("No venues found for the specified period");
      return { races: [], results: [], beforeInfo: [] };
    }

    const totalRaces = venueDays.length * MAX_RACES_PER_VENUE;
    logger.info(
      `Found ${venueDays.length} venue-day(s), up to ${totalRaces} races`,
    );

    const accumulate = !options.onBatchComplete;
    const allRaces: RaceData[] = [];
    const allResults: RaceResultData[] = [];
    const allBeforeInfo: BeforeInfoData[] = [];

    const stats = {
      completedVenues: 0,
      totalRacesScraped: 0,
      cacheHits: 0,
      cacheMisses: 0,
      startTime: Date.now(),
    };

    for (const vd of venueDays) {
      const batch = scrapeVenueDay(vd.stadiumCode, vd.date, options);

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

      stats.completedVenues++;
      stats.totalRacesScraped += batch.races.length;
      stats.cacheHits += batch.cacheHits;
      stats.cacheMisses += batch.cacheMisses;

      if (stats.completedVenues % PROGRESS_LOG_INTERVAL === 0) {
        const elapsed = (Date.now() - stats.startTime) / 1000;
        const rate =
          elapsed > 0 ? (stats.totalRacesScraped / elapsed).toFixed(1) : "—";
        const total = stats.cacheHits + stats.cacheMisses;
        const hitRate =
          total > 0 ? ((stats.cacheHits / total) * 100).toFixed(0) : "—";
        const pct = ((stats.completedVenues / venueDays.length) * 100).toFixed(
          1,
        );
        logger.info(
          `Progress: ${stats.completedVenues}/${venueDays.length} venue-days (${pct}%) | ${stats.totalRacesScraped} races at ${rate}/sec | Cache: ${stats.cacheHits}/${total} (${hitRate}%)`,
        );
      }

      if (batch.scraped > 0 && !isCacheEnabled()) {
        Bun.sleepSync(COOLDOWN_BETWEEN_VENUES_MS);
      }
    }

    const elapsed = ((Date.now() - stats.startTime) / 1000).toFixed(1);
    const total = stats.cacheHits + stats.cacheMisses;
    const hitRate =
      total > 0 ? ((stats.cacheHits / total) * 100).toFixed(0) : "—";
    logger.info(
      `Completed: ${stats.completedVenues} venue-day(s), ${stats.totalRacesScraped} races in ${elapsed}s | Cache: ${stats.cacheHits}/${total} (${hitRate}%)`,
    );

    return { races: allRaces, results: allResults, beforeInfo: allBeforeInfo };
  },
};
