/** Discover active venues and dates from boatrace.jp schedule pages */

import { fetchPage } from "@/features/scraper/http-client";
import { logger } from "@/shared/logger";
import type { CheerioAPI } from "cheerio";
import { dailyScheduleUrl, monthlyScheduleUrl } from "./constants";

export interface VenueDay {
  stadiumCode: string;
  date: string;
}

/**
 * Parse the index page to extract active stadium codes.
 * The schedule page lists venues with links like /race/racelist?jcd=XX&hd=YYYYMMDD
 */
export function parseSchedulePage($: CheerioAPI, date: string): VenueDay[] {
  const venues: VenueDay[] = [];
  const seen = new Set<string>();

  // Stadium links on the index page contain jcd parameter
  $("a[href*='jcd=']").each((_i, el) => {
    const href = $(el).attr("href") ?? "";
    const jcdMatch = href.match(/jcd=(\d{2})/);
    if (jcdMatch) {
      const stadiumCode = jcdMatch[1];
      if (!seen.has(stadiumCode)) {
        seen.add(stadiumCode);
        venues.push({ stadiumCode, date });
      }
    }
  });

  return venues.sort((a, b) => a.stadiumCode.localeCompare(b.stadiumCode));
}

/** Discover venues for a specific date (YYYY-MM-DD or YYYYMMDD) */
export async function discoverDateSchedule(date: string): Promise<VenueDay[]> {
  const yyyymmdd = date.replace(/-/g, "");
  const { $ } = await fetchPage(dailyScheduleUrl(yyyymmdd));
  const venues = parseSchedulePage($, yyyymmdd);
  logger.info(`Found ${venues.length} venue(s) for ${date}`);
  return venues;
}

/** Discover all venue-days for a given month (YYYYMM) */
export async function discoverMonthSchedule(
  yearMonth: string,
): Promise<VenueDay[]> {
  const { $ } = await fetchPage(monthlyScheduleUrl(yearMonth));

  // Extract all date links from the monthly calendar
  const dates = new Set<string>();
  $("a[href*='hd=']").each((_i, el) => {
    const href = $(el).attr("href") ?? "";
    const hdMatch = href.match(/hd=(\d{8})/);
    if (hdMatch?.[1].startsWith(yearMonth)) {
      dates.add(hdMatch[1]);
    }
  });

  const allVenues: VenueDay[] = [];
  const sortedDates = [...dates].sort();

  for (const date of sortedDates) {
    const venues = await discoverDateSchedule(date);
    allVenues.push(...venues);
  }

  logger.info(
    `Found ${allVenues.length} venue-day(s) for ${yearMonth} across ${sortedDates.length} date(s)`,
  );
  return allVenues;
}
