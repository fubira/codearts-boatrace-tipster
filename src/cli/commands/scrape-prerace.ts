import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { getDatabase, initializeDatabase } from "@/features/database";
import { MAX_RACES_PER_VENUE } from "@/features/scraper/sources/boatrace/constants";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { Command } from "commander";

const BOATCAST_BASE = "https://race.boatcast.jp";

/** Sleep between races (ms). Conservative to avoid getting blocked. */
const DEFAULT_SLEEP_MS = 1500;

/** BOATCAST data types to fetch per race */
const DATA_TYPES = {
  /** 展示タイム詳細: 一周・まわり足・直線 */
  oriten: (jcd: string, date: string, race: number) =>
    `${BOATCAST_BASE}/txt/${jcd}/bc_oriten_${date}_${jcd}_${String(race).padStart(2, "0")}.txt`,
  /** ST展示: 展示ST×2、進入コース、Fフラグ、スリット差 */
  stt: (jcd: string, date: string, race: number) =>
    `${BOATCAST_BASE}/hp_txt/${jcd}/bc_j_stt_${date}_${jcd}_${String(race).padStart(2, "0")}.txt`,
} as const;

type DataType = keyof typeof DATA_TYPES;

/** Cache path: data/cache/boatcast/{type}/{YYYYMM}/{YYYYMMDD}_{jcd}_{RR}.txt */
function cachePath(
  type: DataType,
  date: string,
  jcd: string,
  race: number,
): string {
  const yyyymm = date.slice(0, 6);
  const filename = `${date}_${jcd}_${String(race).padStart(2, "0")}.txt`;
  return resolve(config.dataDir, "cache/boatcast", type, yyyymm, filename);
}

function hasCached(
  type: DataType,
  date: string,
  jcd: string,
  race: number,
): boolean {
  return existsSync(cachePath(type, date, jcd, race));
}

function saveCache(
  type: DataType,
  date: string,
  jcd: string,
  race: number,
  content: string,
): void {
  const path = cachePath(type, date, jcd, race);
  const dir = resolve(path, "..");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(path, content, "utf-8");
}

async function fetchText(url: string): Promise<string | null> {
  try {
    const resp = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; boatrace-tipster/0.1)",
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return null;
    const text = await resp.text();
    if (!text || text.length < 10) return null;
    return text;
  } catch {
    return null;
  }
}

/** Get unique venue-days that have races in DB */
function getVenueDays(
  fromDate: string,
  toDate: string,
  stadium?: string,
): { date: string; stadiumCode: string }[] {
  const db = getDatabase();
  let sql =
    "SELECT DISTINCT race_date, stadium_id FROM races WHERE race_date >= $from AND race_date <= $to";
  const params: Record<string, string> = { $from: fromDate, $to: toDate };
  if (stadium) {
    sql += " AND stadium_id = $stadium";
    params.$stadium = stadium;
  }
  sql += " ORDER BY race_date, stadium_id";

  const rows = db.query(sql).all(params) as {
    race_date: string;
    stadium_id: number;
  }[];
  return rows.map((r) => ({
    date: r.race_date.replace(/-/g, ""),
    stadiumCode: String(r.stadium_id).padStart(2, "0"),
  }));
}

export const scrapePreraceCommand = new Command("scrape-prerace")
  .description(
    "Fetch pre-race exhibition data from BOATCAST (まわり足・直線・スリット差)",
  )
  .requiredOption("-d, --date <date>", "start date (YYYY-MM-DD)")
  .option("--to <date>", "end date (YYYY-MM-DD, default: same as --date)")
  .option("-s, --stadium <id>", "stadium code (01-24)")
  .option("--sleep <ms>", "sleep between races in ms", String(DEFAULT_SLEEP_MS))
  .option("--dry-run", "count only, do not download")
  .action(async (opts) => {
    const fromDate = opts.date;
    const toDate = opts.to ?? opts.date;
    const sleepMs = Number.parseInt(opts.sleep, 10);

    initializeDatabase();

    const venueDays = getVenueDays(fromDate, toDate, opts.stadium);
    logger.info(`Period: ${fromDate} to ${toDate}`);
    logger.info(`Found ${venueDays.length} venue-day(s)`);

    // Count total races and already cached
    let totalRaces = 0;
    let alreadyCached = 0;
    const toDownload: { date: string; jcd: string; race: number }[] = [];

    for (const vd of venueDays) {
      for (let race = 1; race <= MAX_RACES_PER_VENUE; race++) {
        totalRaces++;
        const allCached = (Object.keys(DATA_TYPES) as DataType[]).every((t) =>
          hasCached(t, vd.date, vd.stadiumCode, race),
        );
        if (allCached) {
          alreadyCached++;
        } else {
          toDownload.push({ date: vd.date, jcd: vd.stadiumCode, race });
        }
      }
    }

    const estMinutes = (toDownload.length * sleepMs) / 1000 / 60;
    logger.info(
      `Total: ${totalRaces} races, cached: ${alreadyCached}, to download: ${toDownload.length}`,
    );
    logger.info(
      `Estimated time: ${estMinutes.toFixed(0)} min (${(estMinutes / 60).toFixed(1)}h) at ${sleepMs}ms sleep`,
    );

    if (opts.dryRun) {
      logger.info("Dry run — exiting");
      return;
    }

    if (toDownload.length === 0) {
      logger.info("Nothing to download");
      return;
    }

    let downloaded = 0;
    let empty = 0;
    const startTime = Date.now();

    for (let i = 0; i < toDownload.length; i++) {
      const { date, jcd, race } = toDownload[i];

      for (const [type, urlFn] of Object.entries(DATA_TYPES) as [
        DataType,
        (typeof DATA_TYPES)[DataType],
      ][]) {
        if (hasCached(type, date, jcd, race)) continue;

        const url = urlFn(jcd, date, race);
        const text = await fetchText(url);

        if (text) {
          saveCache(type, date, jcd, race, text);
          downloaded++;
        } else {
          // Write empty marker to avoid re-fetching
          saveCache(type, date, jcd, race, "");
          empty++;
        }
      }

      // Progress every 100 races or 60s
      if ((i + 1) % 100 === 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = (i + 1) / elapsed;
        const remaining = (toDownload.length - i - 1) / rate;
        const pct = (((i + 1) / toDownload.length) * 100).toFixed(1);
        logger.info(
          `[${pct}%] ${i + 1}/${toDownload.length} | ` +
            `downloaded=${downloaded} empty=${empty} | ` +
            `ETA ${(remaining / 60).toFixed(0)}min`,
        );
      }

      // Sleep between races
      if (i < toDownload.length - 1) {
        Bun.sleepSync(sleepMs);
      }
    }

    const elapsed = (Date.now() - startTime) / 1000;
    logger.info(
      `Done in ${(elapsed / 60).toFixed(1)}min: downloaded=${downloaded}, empty=${empty}`,
    );
  });
