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
let cacheRequired = false;

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

export function setCacheRequired(): void {
  cacheRequired = true;
}

export function isCacheRequired(): boolean {
  return cacheRequired;
}

/** Convert URL path to cache file path: /race/racelist?rno=1&jcd=04&hd=20250115 -> data/cache/race/racelist/202501/rno=1&jcd=04&hd=20250115.html.gz */
function cachePathFor(path: string): string {
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  // Replace '?' with '/' to create directory structure from query params
  const safePath = normalized.replace("?", "/");
  // Extract YYYYMM from hd= parameter for subdirectory partitioning
  const hdMatch = path.match(/hd=(\d{6})/);
  const yyyymm = hdMatch ? hdMatch[1] : "";
  if (yyyymm) {
    const lastSlash = safePath.lastIndexOf("/");
    const dir = safePath.slice(0, lastSlash);
    const file = safePath.slice(lastSlash + 1);
    return resolve(config.cacheDir, `${dir}/${yyyymm}/${file}.html.gz`);
  }
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

/** Check if a cache file exists without reading it. */
export function hasCacheEntry(path: string): boolean {
  if (!cacheEnabled || !cacheReadEnabled) return false;
  return existsSync(cachePathFor(path));
}

export function writeCache(path: string, html: string): void {
  if (!cacheEnabled) return;
  const cachePath = cachePathFor(path);
  mkdirSync(dirname(cachePath), { recursive: true });
  const compressed = Bun.gzipSync(Buffer.from(html));
  writeFileSync(cachePath, compressed);
}
