import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { getDatabase, initializeDatabase } from "@/features/database";
import { saveBoatcastData } from "@/features/database/storage";
import {
  parseOriten,
  parseStt,
} from "@/features/scraper/sources/boatcast/parsers";
import { MAX_RACES_PER_VENUE } from "@/features/scraper/sources/boatrace/constants";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { Glob } from "bun";
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
  .option("--dry-run", "count only, do not download/import")
  .option("--import", "import cached BOATCAST data into DB")
  .action(async (opts) => {
    const fromDate = opts.date;
    const toDate = opts.to ?? opts.date;
    const sleepMs = Number.parseInt(opts.sleep, 10);

    initializeDatabase();

    if (opts.import) {
      await importBoatcastCache(fromDate, toDate, opts.stadium, opts.dryRun);
      return;
    }

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

/** Parse cache filename: YYYYMMDD_JJ_RR.txt → {date, stadiumId, raceNumber} */
function parseCacheFilename(
  filename: string,
): { date: string; stadiumId: number; raceNumber: number } | null {
  // YYYYMMDD_JJ_RR.txt → regex captures date, stadium code, race number
  const match = filename.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})\.txt$/);
  if (!match) return null;
  const [, y, m, d, jcd, rr] = match;
  return {
    date: `${y}-${m}-${d}`,
    stadiumId: Number.parseInt(jcd, 10),
    raceNumber: Number.parseInt(rr, 10),
  };
}

/** Import cached BOATCAST data (oriten + stt) into DB */
async function importBoatcastCache(
  fromDate: string,
  toDate: string,
  stadium: string | undefined,
  dryRun: boolean | undefined,
): Promise<void> {
  const cacheBase = resolve(config.dataDir, "cache/boatcast");
  const oritenDir = resolve(cacheBase, "oriten");
  const sttDir = resolve(cacheBase, "stt");

  // Collect all oriten files within date range
  const fromYM = fromDate.replace(/-/g, "").slice(0, 6);
  const toYM = toDate.replace(/-/g, "").slice(0, 6);
  const fromD = fromDate.replace(/-/g, "");
  const toD = toDate.replace(/-/g, "");

  const glob = new Glob("*/*.txt");
  const files: { date: string; stadiumId: number; raceNumber: number }[] = [];

  for await (const path of glob.scan(oritenDir)) {
    const yyyymm = path.split("/")[0];
    if (yyyymm < fromYM || yyyymm > toYM) continue;

    const filename = basename(path);
    const parsed = parseCacheFilename(filename);
    if (!parsed) continue;

    const dateStr = filename.slice(0, 8);
    if (dateStr < fromD || dateStr > toD) continue;
    if (stadium && parsed.stadiumId !== Number.parseInt(stadium, 10)) continue;

    files.push(parsed);
  }

  files.sort(
    (a, b) =>
      a.date.localeCompare(b.date) ||
      a.stadiumId - b.stadiumId ||
      a.raceNumber - b.raceNumber,
  );

  logger.info(`Found ${files.length} cache file(s) to import`);
  if (dryRun) {
    logger.info("Dry run — exiting");
    return;
  }

  if (files.length === 0) return;

  const BATCH_SIZE = 500;
  let totalUpdated = 0;
  let totalSkipped = 0;
  let totalEmpty = 0;

  for (
    let batchStart = 0;
    batchStart < files.length;
    batchStart += BATCH_SIZE
  ) {
    const batch = files.slice(batchStart, batchStart + BATCH_SIZE);
    const dataList = [];

    for (const f of batch) {
      const dateStr = f.date.replace(/-/g, "");
      const jcd = String(f.stadiumId).padStart(2, "0");
      const rr = String(f.raceNumber).padStart(2, "0");
      const yyyymm = dateStr.slice(0, 6);
      const fname = `${dateStr}_${jcd}_${rr}.txt`;

      const oritenPath = resolve(oritenDir, yyyymm, fname);
      const sttPath = resolve(sttDir, yyyymm, fname);

      const oritenContent = existsSync(oritenPath)
        ? readFileSync(oritenPath, "utf-8")
        : "";
      const sttContent = existsSync(sttPath)
        ? readFileSync(sttPath, "utf-8")
        : "";

      const oriten = parseOriten(oritenContent);
      const stt = parseStt(sttContent);

      if (oriten.length === 0 && stt.length === 0) {
        totalEmpty++;
        continue;
      }

      dataList.push({
        stadiumId: f.stadiumId,
        raceDate: f.date,
        raceNumber: f.raceNumber,
        oriten,
        stt,
      });
    }

    if (dataList.length > 0) {
      const result = saveBoatcastData(dataList);
      totalUpdated += result.updated;
      totalSkipped += result.skipped;
    }

    if ((batchStart + BATCH_SIZE) % 1000 < BATCH_SIZE) {
      const processed = Math.min(batchStart + BATCH_SIZE, files.length);
      const pct = ((processed / files.length) * 100).toFixed(1);
      logger.info(
        `[${pct}%] ${processed}/${files.length} | updated=${totalUpdated} skipped=${totalSkipped} empty=${totalEmpty}`,
      );
    }
  }

  logger.info(
    `Import complete: updated=${totalUpdated} entries, skipped=${totalSkipped} races, empty=${totalEmpty} files`,
  );
}
