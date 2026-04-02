/**
 * HTTP client for boatrace.jp scraping.
 * Uses curl for HTTP/2 support (Akamai throttles HTTP/1.1 requests).
 */

import { execSync } from "node:child_process";
import { logger } from "@/shared/logger";
import { isCacheRequired, readCache, writeCache } from "./cache-manager";

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

const BASE_URL = "https://www.boatrace.jp";

export interface FetchPageResult {
  html: string;
  status: number;
  fromCache: boolean;
}

export function fetchPage(
  path: string,
  options?: { skipCache?: boolean },
): FetchPageResult | null {
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

  const html = execSync(
    `curl -s --compressed -H "User-Agent: ${USER_AGENT}" -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" -H "Accept-Language: ja,en-US;q=0.9,en;q=0.8" -H "Accept-Encoding: gzip, deflate, br" -H "Sec-Fetch-Dest: document" -H "Sec-Fetch-Mode: navigate" -H "Sec-Fetch-Site: none" -H "Sec-Fetch-User: ?1" "${url}"`,
    { encoding: "utf-8", timeout: 30_000 },
  );

  if (!html || html.length === 0) {
    throw new Error(`Empty response for ${url}`);
  }

  writeCache(path, html);

  return { html, status: 200, fromCache: false };
}
