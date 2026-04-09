/** Shared scraping helpers used by both runner and scrape-daemon. */

import {
  type OddsTiming,
  saveBeforeInfo,
  saveOdds,
  saveOddsSnapshot,
  saveRaceResults,
} from "@/features/database";
import { fetchPage } from "@/features/scraper/http-client";
import {
  type RaceParams,
  beforeInfoUrl,
  odds3TUrl,
  raceResultUrl,
} from "@/features/scraper/sources/boatrace/constants";
import { parseOdds3T } from "@/features/scraper/sources/boatrace/odds-parsers";
import {
  parseBeforeInfo,
  parseRaceResult,
} from "@/features/scraper/sources/boatrace/parsers";
import type { RaceSlot } from "./race-scheduler";

export function padStadiumCode(id: number): string {
  return String(id).padStart(2, "0");
}

export function scrapeBeforeInfoForRace(slot: RaceSlot, date: string): boolean {
  const params: RaceParams = {
    raceNumber: slot.raceNumber,
    stadiumCode: padStadiumCode(slot.stadiumId),
    date: date.replace(/-/g, ""),
  };

  // Always skip cache — exhibition data changes up to race time
  const page = fetchPage(beforeInfoUrl(params), { skipCache: true });
  if (!page) return false;

  const context = { params, raceDate: date };
  const data = parseBeforeInfo(page.html, context);
  if (!data) return false;

  saveBeforeInfo([data]);
  return true;
}

export function scrapeResultForRace(
  slot: RaceSlot,
  date: string,
): {
  entries: { boatNumber: number; finishPosition?: number }[];
  payouts?: { betType: string; combination: string; payout: number }[];
} | null {
  const params: RaceParams = {
    raceNumber: slot.raceNumber,
    stadiumCode: padStadiumCode(slot.stadiumId),
    date: date.replace(/-/g, ""),
  };

  // Always skip cache — result page only available after race
  const page = fetchPage(raceResultUrl(params), { skipCache: true });
  if (!page) return null;

  const context = { params, raceDate: date };
  const data = parseRaceResult(page.html, context);
  if (!data) return null;

  saveRaceResults([data]);

  return data;
}

/**
 * Fetch trifecta odds and save as snapshot.
 * Pre-race timings (T-5/T-3/T-1) save to race_odds_snapshots ONLY.
 * "final" saves to both race_odds (confirmed) and race_odds_snapshots.
 */
export function fetchTrifectaOdds(
  slots: RaceSlot[],
  date: string,
  timing: OddsTiming,
  oddsTimings: Map<number, Set<OddsTiming>>,
): number {
  let fetched = 0;
  const isConfirmed = timing === "final";

  for (const slot of slots) {
    // Skip if this timing already collected for this race
    if (oddsTimings.get(slot.raceId)?.has(timing)) continue;

    const stadiumCode = padStadiumCode(slot.stadiumId);
    const params: RaceParams = {
      raceNumber: slot.raceNumber,
      stadiumCode,
      date: date.replace(/-/g, ""),
    };
    try {
      const page = fetchPage(odds3TUrl(params), { skipCache: true });
      if (page) {
        const entries = parseOdds3T(page.html);
        if (entries.length > 0) {
          const oddsEntries = entries.map((e) => ({
            betType: e.betType,
            combination: e.combination,
            odds: e.odds,
          }));

          // Confirmed odds → race_odds table (the source of truth)
          if (isConfirmed) {
            saveOdds([
              {
                stadiumId: slot.stadiumId,
                raceDate: date,
                raceNumber: slot.raceNumber,
                entries: oddsEntries,
              },
            ]);
          }

          // All timings → snapshots (for drift analysis)
          saveOddsSnapshot(slot.raceId, timing, oddsEntries);
          if (!oddsTimings.has(slot.raceId)) {
            oddsTimings.set(slot.raceId, new Set());
          }
          oddsTimings.get(slot.raceId)?.add(timing);

          fetched++;
        }
      }
    } catch {
      // Odds may not be available yet for early races
    }
  }
  return fetched;
}
