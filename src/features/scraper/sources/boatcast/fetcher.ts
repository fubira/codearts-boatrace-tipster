/** Fetch and import BOATCAST exhibition data for a single race. */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { saveBoatcastData } from "@/features/database/storage";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { parseOriten, parseStt } from "./parsers";

const BOATCAST_BASE = "https://race.boatcast.jp";

const DATA_URLS = {
  oriten: (jcd: string, date: string, race: number) =>
    `${BOATCAST_BASE}/txt/${jcd}/bc_oriten_${date}_${jcd}_${String(race).padStart(2, "0")}.txt`,
  stt: (jcd: string, date: string, race: number) =>
    `${BOATCAST_BASE}/hp_txt/${jcd}/bc_j_stt_${date}_${jcd}_${String(race).padStart(2, "0")}.txt`,
} as const;

type DataType = keyof typeof DATA_URLS;

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

function hasCacheContent(
  type: DataType,
  dateStr: string,
  jcd: string,
  race: number,
): boolean {
  const path = cachePath(type, dateStr, jcd, race);
  if (!existsSync(path)) return false;
  return readFileSync(path, "utf-8").length >= 10;
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

async function fetchOrReadCache(
  type: DataType,
  dateStr: string,
  jcd: string,
  raceNumber: number,
  skipCacheRead: boolean,
): Promise<string> {
  const path = cachePath(type, dateStr, jcd, raceNumber);

  if (!skipCacheRead && existsSync(path)) {
    return readFileSync(path, "utf-8");
  }

  const url = DATA_URLS[type](jcd, dateStr, raceNumber);
  const text = await fetchText(url);
  if (text) {
    saveCache(type, dateStr, jcd, raceNumber, text);
    return text;
  }
  return "";
}

/**
 * Fetch BOATCAST exhibition data for a single race, parse, and save to DB.
 *
 * @param skipCacheRead - Bypass cache and re-fetch from network (for retry)
 */
export async function fetchAndSaveBoatcast(
  stadiumId: number,
  raceDate: string,
  raceNumber: number,
  options?: { skipCacheRead?: boolean },
): Promise<boolean> {
  const jcd = String(stadiumId).padStart(2, "0");
  const dateStr = raceDate.replace(/-/g, "");
  const skip = options?.skipCacheRead ?? false;

  const oritenContent = await fetchOrReadCache(
    "oriten",
    dateStr,
    jcd,
    raceNumber,
    skip,
  );
  const sttContent = await fetchOrReadCache(
    "stt",
    dateStr,
    jcd,
    raceNumber,
    skip,
  );

  const oriten = parseOriten(oritenContent);
  const stt = parseStt(sttContent);

  if (oriten.length === 0 && stt.length === 0) return false;

  const result = saveBoatcastData([
    { stadiumId, raceDate, raceNumber, oriten, stt },
  ]);

  return result.updated > 0;
}

/**
 * Retry BOATCAST fetch for a race if prior attempt got empty/partial data.
 * Skips re-fetch for types that already have cached content.
 */
export async function retryBoatcastIfMissing(
  stadiumId: number,
  raceDate: string,
  raceNumber: number,
): Promise<boolean> {
  const jcd = String(stadiumId).padStart(2, "0");
  const dateStr = raceDate.replace(/-/g, "");

  const oritenOk = hasCacheContent("oriten", dateStr, jcd, raceNumber);
  const sttOk = hasCacheContent("stt", dateStr, jcd, raceNumber);

  if (oritenOk && sttOk) return false;

  return fetchAndSaveBoatcast(stadiumId, raceDate, raceNumber, {
    skipCacheRead: true,
  });
}
