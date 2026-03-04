/** Scraper plugin interface and shared types */

import type {
  BeforeInfoData,
  RaceData,
  RaceResultData,
} from "@/features/database";

export interface ScraperResult {
  races: RaceData[];
  results: RaceResultData[];
  beforeInfo: BeforeInfoData[];
}

export interface Scraper {
  readonly name: string;
  readonly description: string;
  scrape(options: ScraperOptions): Promise<ScraperResult>;
}

export interface ScraperOptions {
  date?: string;
  month?: string;
  year?: string;
  stadiumId?: string;
  raceNumbers?: number[];
  limit?: number;
  /** Callback to check if a race should be skipped */
  shouldSkip?: (
    stadiumId: number,
    raceDate: string,
    raceNumber: number,
  ) => boolean;
  /** Called after each venue-day group completes for incremental saving */
  onBatchComplete?: (batch: ScraperResult) => void;
}
