import { boatraceScraper } from "./sources/boatrace";
import type { Scraper } from "./types";

const scrapers = new Map<string, Scraper>();

scrapers.set(boatraceScraper.name, boatraceScraper);

export function getScraper(name: string): Scraper | undefined {
  return scrapers.get(name);
}
