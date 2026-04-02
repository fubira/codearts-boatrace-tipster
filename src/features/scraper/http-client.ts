/**
 * HTTP client for boatrace.jp scraping.
 * Uses curl for HTTP/2 support (Akamai throttles HTTP/1.1 requests).
 * Supports batch fetching for connection reuse.
 */

import { execSync } from "node:child_process";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { logger } from "@/shared/logger";
import {
  hasCacheEntry,
  isCacheRequired,
  readCache,
  writeCache,
} from "./cache-manager";

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";

const BASE_URL = "https://www.boatrace.jp";

const CURL_HEADERS = [
  `-H "User-Agent: ${USER_AGENT}"`,
  '-H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"',
  '-H "Accept-Language: ja,en-US;q=0.9,en;q=0.8"',
  '-H "Accept-Encoding: gzip, deflate, br"',
  '-H "Sec-Fetch-Dest: document"',
  '-H "Sec-Fetch-Mode: navigate"',
  '-H "Sec-Fetch-Site: none"',
  '-H "Sec-Fetch-User: ?1"',
].join(" ");

export interface FetchPageResult {
  html: string;
  status: number;
  fromCache: boolean;
}

/**
 * Fetch a single page. Uses cache if available, otherwise curl.
 */
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

  const html = execSync(`curl -s --compressed ${CURL_HEADERS} "${url}"`, {
    encoding: "utf-8",
    timeout: 30_000,
  });

  if (!html || html.length === 0) {
    throw new Error(`Empty response for ${url}`);
  }

  writeCache(path, html);
  return { html, status: 200, fromCache: false };
}

/**
 * Fetch multiple pages in a single curl invocation (HTTP/2 connection reuse).
 * Returns results in the same order as input paths.
 * Cached pages return html="" (existence check only, no decompress).
 * Use for cache-only downloads or when html content of cached pages is not needed.
 */
export function fetchPages(paths: string[]): (FetchPageResult | null)[] {
  const results: (FetchPageResult | null)[] = new Array(paths.length).fill(
    null,
  );
  const toFetch: {
    index: number;
    path: string;
    url: string;
    tmpFile: string;
  }[] = [];

  // Check cache: existence-only check (skip decompress) for download mode,
  // full read for parse mode
  for (let i = 0; i < paths.length; i++) {
    if (hasCacheEntry(paths[i])) {
      results[i] = { html: "", status: 200, fromCache: true };
      continue;
    }
    if (isCacheRequired()) continue;

    const url = paths[i].startsWith("http")
      ? paths[i]
      : `${BASE_URL}${paths[i]}`;
    const tmpFile = join(tmpdir(), `boatrace-fetch-${i}-${Date.now()}.html`);
    toFetch.push({ index: i, path: paths[i], url, tmpFile });
  }

  if (toFetch.length === 0) return results;

  if (toFetch.length > 0) {
    logger.warn(`[HTTP] Batch GET ${toFetch.length} pages`);
  }

  // Build single curl command with all URLs
  const curlArgs = toFetch.map((f) => `-o "${f.tmpFile}" "${f.url}"`).join(" ");

  try {
    execSync(`curl -s --compressed ${CURL_HEADERS} ${curlArgs}`, {
      timeout: 60_000 + toFetch.length * 5_000,
      stdio: ["pipe", "pipe", "pipe"],
    });

    // Read results from temp files
    for (const f of toFetch) {
      try {
        const html = readFileSync(f.tmpFile, "utf-8");
        if (html && html.length > 0) {
          writeCache(f.path, html);
          results[f.index] = { html, status: 200, fromCache: false };
        }
      } finally {
        try {
          rmSync(f.tmpFile);
        } catch {}
      }
    }
  } catch (e) {
    // Clean up temp files on error
    for (const f of toFetch) {
      try {
        rmSync(f.tmpFile);
      } catch {}
    }
    throw e;
  }

  return results;
}
