/**
 * HTML cache with gzip compression.
 * Stores fetched HTML pages locally to avoid redundant network requests.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";

let cacheEnabled = false;
let cacheReadEnabled = true;

export function enableCache(): void {
  cacheEnabled = true;
  logger.info(`HTML cache enabled at ${config.cacheDir}`);
}

export function disableCacheRead(): void {
  cacheReadEnabled = false;
}

export function isCacheEnabled(): boolean {
  return cacheEnabled;
}

/** Convert URL path to cache file path: /race/racelist?rno=1&jcd=04&hd=20250115 -> data/cache/race/racelist/rno=1&jcd=04&hd=20250115.html.gz */
function cachePathFor(path: string): string {
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  // Replace '?' with '/' to create directory structure from query params
  const safePath = normalized.replace("?", "/");
  return resolve(config.cacheDir, `${safePath}.html.gz`);
}

export function readCache(path: string): string | undefined {
  if (!cacheEnabled || !cacheReadEnabled) return undefined;
  const cachePath = cachePathFor(path);
  if (!existsSync(cachePath)) return undefined;
  const compressed = readFileSync(cachePath);
  const html = new TextDecoder().decode(Bun.gunzipSync(compressed));
  logger.debug(`Cache hit: ${path}`);
  return html;
}

export function writeCache(path: string, html: string): void {
  if (!cacheEnabled) return;
  const cachePath = cachePathFor(path);
  mkdirSync(dirname(cachePath), { recursive: true });
  const compressed = Bun.gzipSync(Buffer.from(html));
  writeFileSync(cachePath, compressed);
}
