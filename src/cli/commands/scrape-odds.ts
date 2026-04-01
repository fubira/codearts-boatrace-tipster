import {
  closeDatabase,
  initializeDatabase,
  saveOdds,
} from "@/features/database";
import type { OddsData } from "@/features/database";
import {
  enableCache,
  setCacheRequired,
} from "@/features/scraper/cache-manager";
import { fetchPage } from "@/features/scraper/http-client";
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
  { name: "oddstf", urlFn: oddsTfUrl, parseFn: parseOddsTf },
  { name: "odds2tf", urlFn: odds2TfUrl, parseFn: parseOdds2Tf },
  { name: "odds3t", urlFn: odds3TUrl, parseFn: parseOdds3T },
  { name: "odds3f", urlFn: odds3FUrl, parseFn: parseOdds3F },
] as const;

async function scrapeOddsForRace(params: RaceParams): Promise<OddsEntry[]> {
  const allEntries: OddsEntry[] = [];

  const pages = await Promise.all(
    ODDS_PAGES.map((p) => fetchPage(p.urlFn(params))),
  );

  let anyFetched = false;
  for (let i = 0; i < pages.length; i++) {
    const page = pages[i];
    if (!page) continue;
    if (!page.fromCache) anyFetched = true;
    const entries = ODDS_PAGES[i].parseFn(page.html);
    allEntries.push(...entries);
  }

  if (anyFetched) await Bun.sleep(COOLDOWN_BETWEEN_PAGES_MS);
  return allEntries;
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
  .option("--cache-only", "download HTML to cache without parsing")
  .option("--from-cache", "parse from cache only, never fetch from network")
  .action(async (opts) => {
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

    enableCache();
    if (opts.fromCache) setCacheRequired();

    if (!opts.dryRun && !opts.cacheOnly) {
      initializeDatabase();
    }

    // Resolve venue-days
    let venueDays: { stadiumCode: string; date: string }[] = [];

    if (opts.date) {
      const yyyymmdd = opts.date.replace(/-/g, "");
      const venues = await discoverDateSchedule(opts.date);
      venueDays = venues.map((v) => ({ ...v, date: yyyymmdd }));
    } else if (opts.month) {
      venueDays = await discoverMonthSchedule(opts.month);
    } else if (opts.year) {
      const months = Array.from(
        { length: 12 },
        (_, i) => `${opts.year}${String(i + 1).padStart(2, "0")}`,
      );
      for (const month of months) {
        logger.info(`Fetching schedule for ${month}...`);
        const venues = await discoverMonthSchedule(month);
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
      const batchOdds: OddsData[] = [];

      for (let rno = 1; rno <= MAX_RACES_PER_VENUE; rno++) {
        const params: RaceParams = {
          raceNumber: rno,
          stadiumCode: vd.stadiumCode,
          date: vd.date,
        };
        const entries = await scrapeOddsForRace(params);
        if (entries.length > 0) {
          batchOdds.push({
            stadiumId: Number.parseInt(vd.stadiumCode, 10),
            raceDate,
            raceNumber: rno,
            entries,
          });
        }
      }

      if (!opts.cacheOnly && !opts.dryRun && batchOdds.length > 0) {
        saveOdds(batchOdds);
      }

      totalOdds += batchOdds.reduce((sum, o) => sum + o.entries.length, 0);
      completedVenues++;

      if (completedVenues % 50 === 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const pct = ((completedVenues / venueDays.length) * 100).toFixed(1);
        logger.info(
          `Progress: ${completedVenues}/${venueDays.length} venue-days (${pct}%) | ${totalOdds} odds in ${elapsed.toFixed(0)}s`,
        );
      }

      logger.info(
        `${venueName} ${raceDate}: ${batchOdds.reduce((s, o) => s + o.entries.length, 0)} odds`,
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
