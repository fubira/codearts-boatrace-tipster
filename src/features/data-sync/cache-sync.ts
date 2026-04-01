/** Bidirectional HTML cache sync using rsync --ignore-existing. */

import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { rsync } from "./ssh";
import type { SyncConfig } from "./sync-config";

export interface CacheSyncResult {
  pulled: number;
  pushed: number;
}

export function syncCache(
  conf: SyncConfig,
  options?: { dryRun?: boolean },
): CacheSyncResult {
  const localCache = `${config.cacheDir}/`;
  const remoteCache = `${conf.server}:${conf.prodDir}/data/cache/`;

  // Pull: server → local (new scraping results)
  logger.info("Pulling cache from server...");
  const pull = rsync(remoteCache, localCache, {
    ignoreExisting: true,
    dryRun: options?.dryRun,
  });

  // Push: local → server (historical cache)
  logger.info("Pushing cache to server...");
  const push = rsync(localCache, remoteCache, {
    ignoreExisting: true,
    dryRun: options?.dryRun,
  });

  return { pulled: pull.transferred, pushed: push.transferred };
}
