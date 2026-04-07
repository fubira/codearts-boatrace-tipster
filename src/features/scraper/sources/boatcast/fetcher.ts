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

/**
 * Fetch BOATCAST exhibition data for a single race, parse, and save to DB.
 *
 * @param stadiumId Stadium ID (1-24)
 * @param raceDate Date in YYYY-MM-DD format
 * @param raceNumber Race number (1-12)
 * @returns true if data was fetched and saved
 */
export async function fetchAndSaveBoatcast(
  stadiumId: number,
  raceDate: string,
  raceNumber: number,
): Promise<boolean> {
  const jcd = String(stadiumId).padStart(2, "0");
  const dateStr = raceDate.replace(/-/g, "");

  let oritenContent = "";
  let sttContent = "";

  // Fetch oriten
  const oritenCachePath = cachePath("oriten", dateStr, jcd, raceNumber);
  if (existsSync(oritenCachePath)) {
    oritenContent = readFileSync(oritenCachePath, "utf-8");
  } else {
    const url = DATA_URLS.oriten(jcd, dateStr, raceNumber);
    const text = await fetchText(url);
    if (text) {
      saveCache("oriten", dateStr, jcd, raceNumber, text);
      oritenContent = text;
    } else {
      saveCache("oriten", dateStr, jcd, raceNumber, "");
    }
  }

  // Fetch stt
  const sttCachePath = cachePath("stt", dateStr, jcd, raceNumber);
  if (existsSync(sttCachePath)) {
    sttContent = readFileSync(sttCachePath, "utf-8");
  } else {
    const url = DATA_URLS.stt(jcd, dateStr, raceNumber);
    const text = await fetchText(url);
    if (text) {
      saveCache("stt", dateStr, jcd, raceNumber, text);
      sttContent = text;
    } else {
      saveCache("stt", dateStr, jcd, raceNumber, "");
    }
  }

  const oriten = parseOriten(oritenContent);
  const stt = parseStt(sttContent);

  if (oriten.length === 0 && stt.length === 0) return false;

  const result = saveBoatcastData([
    { stadiumId, raceDate, raceNumber, oriten, stt },
  ]);

  return result.updated > 0;
}
