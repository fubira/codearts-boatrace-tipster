import {
  closeDatabase,
  initializeDatabase,
  saveOdds,
} from "@/features/database";
import type { OddsData } from "@/features/database";
import {
  disableCacheRead,
  enableCache,
  setCacheRequired,
} from "@/features/scraper/cache-manager";
import { fetchPage, fetchPages } from "@/features/scraper/http-client";
import {
  COOLDOWN_BETWEEN_PAGES_MS,
  MAX_RACES_PER_VENUE,
  type RaceParams,
  STADIUMS,
  odds2TfUrl,
  odds3FUrl,
  odds3TUrl,
  oddsTfUrl,
} from "@/features/scraper/sources/boatrace/constants";
import {
  discoverDateSchedule,
  discoverMonthSchedule,
} from "@/features/scraper/sources/boatrace/discovery";
import type { OddsEntry } from "@/features/scraper/sources/boatrace/odds-parsers";
import {
  parseOdds2Tf,
  parseOdds3F,
  parseOdds3T,
  parseOddsTf,
} from "@/features/scraper/sources/boatrace/odds-parsers";
import { logger } from "@/shared/logger";
import { Command } from "commander";

const ODDS_PAGES = [
  { urlFn: oddsTfUrl, parseFn: parseOddsTf },
  { urlFn: odds2TfUrl, parseFn: parseOdds2Tf },
  { urlFn: odds3TUrl, parseFn: parseOdds3T },
  { urlFn: odds3FUrl, parseFn: parseOdds3F },
] as const;

/** Download odds cache for a venue-day using batch curl (48 URLs in 1 call) */
function downloadVenueDayOdds(stadiumCode: string, date: string): void {
  const allPaths: string[] = [];
  for (let rno = 1; rno <= MAX_RACES_PER_VENUE; rno++) {
    const params: RaceParams = { raceNumber: rno, stadiumCode, date };
    allPaths.push(oddsTfUrl(params));
    allPaths.push(odds2TfUrl(params));
    allPaths.push(odds3TUrl(params));
    allPaths.push(odds3FUrl(params));
  }
  const pages = fetchPages(allPaths);
  const anyFetched = pages.some((p) => p !== null && !p.fromCache);
  if (anyFetched) Bun.sleepSync(COOLDOWN_BETWEEN_PAGES_MS);
}

/** Scrape and parse odds for a venue-day (individual fetchPage for full HTML read) */
function scrapeVenueDayOdds(
  stadiumCode: string,
  date: string,
): { raceNumber: number; entries: OddsEntry[] }[] {
  const results: { raceNumber: number; entries: OddsEntry[] }[] = [];

  for (let rno = 1; rno <= MAX_RACES_PER_VENUE; rno++) {
    const params: RaceParams = { raceNumber: rno, stadiumCode, date };
    const entries: OddsEntry[] = [];
    let anyFetched = false;

    for (const p of ODDS_PAGES) {
      const page = fetchPage(p.urlFn(params));
      if (!page) continue;
      if (!page.fromCache) anyFetched = true;
      entries.push(...p.parseFn(page.html));
    }

    if (anyFetched) Bun.sleepSync(COOLDOWN_BETWEEN_PAGES_MS);
    if (entries.length > 0) {
      results.push({ raceNumber: rno, entries });
    }
  }

  return results;
}

function buildRaceDate(yyyymmdd: string): string {
  return `${yyyymmdd.slice(0, 4)}-${yyyymmdd.slice(4, 6)}-${yyyymmdd.slice(6, 8)}`;
}

export const scrapeOddsCommand = new Command("scrape-odds")
  .description("Scrape odds data from boatrace.jp")
  .option("-d, --date <date>", "target date (YYYY-MM-DD)")
  .option("-m, --month <month>", "target month (YYYYMM)")
  .option("-y, --year <year>", "target year (YYYY)")
  .option("-s, --stadium <id>", "stadium code (01-24)")
  .option("--dry-run", "parse only, do not save to DB")
  .option("--no-cache", "ignore cached HTML, always fetch from network")
  .option("--cache-only", "download HTML to cache without parsing")
  .option("--from-cache", "parse from cache only, never fetch from network")
  .action((opts) => {
    if (!opts.date && !opts.month && !opts.year) {
      console.error("Error: --date, --month, or --year is required");
      process.exit(1);
    }

    if (opts.cacheOnly && opts.fromCache) {
      console.error(
        "Error: --cache-only and --from-cache cannot be used together",
      );
      process.exit(1);
    }

    if (opts.noCache && opts.fromCache) {
      console.error(
        "Error: --no-cache and --from-cache cannot be used together",
      );
      process.exit(1);
    }

    enableCache();
    if (opts.noCache) disableCacheRead();
    if (opts.fromCache) setCacheRequired();

    if (!opts.dryRun && !opts.cacheOnly) {
      initializeDatabase();
    }

    // Resolve venue-days
    let venueDays: { stadiumCode: string; date: string }[] = [];

    if (opts.date) {
      const yyyymmdd = opts.date.replace(/-/g, "");
      const venues = discoverDateSchedule(opts.date);
      venueDays = venues.map((v) => ({ ...v, date: yyyymmdd }));
    } else if (opts.month) {
      venueDays = discoverMonthSchedule(opts.month);
    } else if (opts.year) {
      const months = Array.from(
        { length: 12 },
        (_, i) => `${opts.year}${String(i + 1).padStart(2, "0")}`,
      );
      for (const month of months) {
        logger.info(`Fetching schedule for ${month}...`);
        const venues = discoverMonthSchedule(month);
        venueDays.push(...venues);
      }
    }

    if (opts.stadium) {
      venueDays = venueDays.filter((v) => v.stadiumCode === opts.stadium);
    }

    logger.info(`Found ${venueDays.length} venue-day(s)`);

    let totalOdds = 0;
    let completedVenues = 0;
    const startTime = Date.now();

    for (const vd of venueDays) {
      const venueName = STADIUMS[vd.stadiumCode] ?? vd.stadiumCode;
      const raceDate = buildRaceDate(vd.date);

      if (opts.cacheOnly) {
        downloadVenueDayOdds(vd.stadiumCode, vd.date);
      } else {
        const raceResults = scrapeVenueDayOdds(vd.stadiumCode, vd.date);
        const batchOdds: OddsData[] = raceResults.map((r) => ({
          stadiumId: Number.parseInt(vd.stadiumCode, 10),
          raceDate,
          raceNumber: r.raceNumber,
          entries: r.entries,
        }));

        if (!opts.dryRun && batchOdds.length > 0) {
          saveOdds(batchOdds);
        }

        totalOdds += batchOdds.reduce((sum, o) => sum + o.entries.length, 0);
      }
      completedVenues++;

      if (completedVenues % 50 === 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const pct = ((completedVenues / venueDays.length) * 100).toFixed(1);
        logger.info(
          `Progress: ${completedVenues}/${venueDays.length} venue-days (${pct}%) | ${totalOdds} odds in ${elapsed.toFixed(0)}s`,
        );
      }

      logger.info(
        `${venueName} ${raceDate}: ${opts.cacheOnly ? "cached" : `${totalOdds} odds`}`,
      );
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    logger.info(
      `Completed: ${completedVenues} venue-day(s), ${totalOdds} odds in ${elapsed}s`,
    );

    if (!opts.dryRun && !opts.cacheOnly) {
      closeDatabase();
    }
  });
