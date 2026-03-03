import {
  closeDatabase,
  initializeDatabase,
  isRaceScraped,
  saveBeforeInfo,
  saveRaceResults,
  saveRaces,
} from "@/features/database";
import {
  disableCacheRead,
  enableCache,
} from "@/features/scraper/cache-manager";
import { getScraper } from "@/features/scraper/registry";
import type { ScraperResult } from "@/features/scraper/types";
import { logger } from "@/shared/logger";
import { Command } from "commander";

export const scrapeCommand = new Command("scrape")
  .description("Scrape race data from boatrace.jp")
  .option("-d, --date <date>", "target date (YYYY-MM-DD)")
  .option("-m, --month <month>", "target month (YYYYMM)")
  .option("-y, --year <year>", "target year (YYYY)")
  .option("-s, --stadium <id>", "stadium code (01-24)")
  .option("-l, --limit <n>", "max races to scrape", Number.parseInt)
  .option("--dry-run", "parse only, do not save to DB")
  .option("--force", "re-scrape even if already in DB")
  .option("--cache-only", "download HTML to cache without parsing")
  .action(async (opts) => {
    if (!opts.date && !opts.month && !opts.year) {
      console.error("Error: --date, --month, or --year is required");
      process.exit(1);
    }

    enableCache();

    if (opts.force) {
      disableCacheRead();
    }

    const scraper = getScraper("boatrace");
    if (!scraper) {
      console.error("Error: boatrace scraper not found");
      process.exit(1);
    }

    if (!opts.dryRun && !opts.cacheOnly) {
      initializeDatabase();
    }

    let totalSaved = 0;

    const shouldSkip = opts.force
      ? undefined
      : (stadiumId: number, raceDate: string, raceNumber: number) => {
          if (opts.dryRun || opts.cacheOnly) return false;
          return isRaceScraped(stadiumId, raceDate, raceNumber);
        };

    const onBatchComplete = opts.cacheOnly
      ? undefined
      : (batch: ScraperResult) => {
          if (opts.dryRun) {
            logger.info(
              `[dry-run] Would save: ${batch.races.length} races, ${batch.results.length} results, ${batch.beforeInfo.length} before-info`,
            );
            return;
          }

          if (batch.races.length > 0) {
            const result = saveRaces(batch.races);
            totalSaved += result.racesUpserted;
          }
          if (batch.results.length > 0) {
            saveRaceResults(batch.results);
          }
          if (batch.beforeInfo.length > 0) {
            saveBeforeInfo(batch.beforeInfo);
          }
        };

    try {
      const result = await scraper.scrape({
        date: opts.date,
        month: opts.month,
        year: opts.year,
        stadiumId: opts.stadium,
        limit: opts.limit,
        shouldSkip,
        onBatchComplete,
      });

      if (opts.cacheOnly) {
        logger.info("Cache-only mode: HTML files saved to data/cache/");
      } else if (opts.dryRun) {
        logger.info(
          `[dry-run] Total: ${result.races.length} races, ${result.results.length} results, ${result.beforeInfo.length} before-info`,
        );
      } else {
        logger.info(`Completed: ${totalSaved} races saved to database`);
      }
    } finally {
      if (!opts.dryRun && !opts.cacheOnly) {
        closeDatabase();
      }
    }
  });
