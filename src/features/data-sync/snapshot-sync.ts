/** Bidirectional stats-snapshot sync using rsync --ignore-existing. */

import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { rsync } from "./ssh";
import type { SyncConfig } from "./sync-config";

export interface SnapshotSyncResult {
  pulled: number;
  pushed: number;
}

export function syncSnapshots(
  conf: SyncConfig,
  options?: { dryRun?: boolean },
): SnapshotSyncResult {
  const localDir = `${config.dataDir}/stats-snapshots/`;
  const remoteDir = `${conf.server}:${conf.prodDir}/data/stats-snapshots/`;

  // Pull: server → local (snapshots built by runner on server)
  logger.info("Pulling snapshots from server...");
  const pull = rsync(remoteDir, localDir, {
    ignoreExisting: true,
    dryRun: options?.dryRun,
  });

  // Push: local → server (snapshots built locally)
  logger.info("Pushing snapshots to server...");
  const push = rsync(localDir, remoteDir, {
    ignoreExisting: true,
    dryRun: options?.dryRun,
  });

  return { pulled: pull.transferred, pushed: push.transferred };
}
