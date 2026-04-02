/**
 * HTTP client for boatrace.jp scraping.
 * No authentication needed — simplified from tateyamakun.
 */

import { logger } from "@/shared/logger";
import { isCacheRequired, readCache, writeCache } from "./cache-manager";

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

const BASE_URL = "https://www.boatrace.jp";

const BASE_HEADERS: Record<string, string> = {
  "User-Agent": USER_AGENT,
  Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
  "Accept-Encoding": "gzip, deflate, br",
  "Sec-Fetch-Dest": "document",
  "Sec-Fetch-Mode": "navigate",
  "Sec-Fetch-Site": "none",
  "Sec-Fetch-User": "?1",
};

export interface FetchPageResult {
  html: string;
  status: number;
  fromCache: boolean;
}

export async function fetchPage(
  path: string,
  options?: { skipCache?: boolean },
): Promise<FetchPageResult | null> {
  if (!options?.skipCache) {
    const cached = readCache(path);
    if (cached) {
      return { html: cached, status: 200, fromCache: true };
    }
  }

  if (isCacheRequired()) {
    logger.debug(`Cache miss (from-cache mode): ${path}`);
    return null;
  }

  const url = path.startsWith("http") ? path : `${BASE_URL}${path}`;
  logger.warn(`[HTTP] GET ${url}`);

  const response = await fetch(url, { headers: BASE_HEADERS });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }

  const html = await response.text();
  writeCache(path, html);

  return { html, status: response.status, fromCache: false };
}
