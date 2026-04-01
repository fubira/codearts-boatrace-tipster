/** Pull database from server with atomic swap and backup. */

import {
  existsSync,
  mkdirSync,
  readdirSync,
  renameSync,
  rmSync,
} from "node:fs";
import { resolve } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { rsync, sshExec } from "./ssh";
import type { SyncConfig } from "./sync-config";

const BACKUP_KEEP_COUNT = 7;

export interface DbSyncResult {
  backedUp: boolean;
  backupPath: string | null;
}

export function syncDb(conf: SyncConfig): DbSyncResult {
  const localDb = config.dbPath;
  const remoteDbPath = `${conf.prodDir}/data/boatrace-tipster.db`;

  // Step 1: WAL checkpoint on server
  logger.info("Running WAL checkpoint on server...");
  try {
    sshExec(
      conf,
      `sqlite3 ${remoteDbPath} "PRAGMA wal_checkpoint(TRUNCATE);"`,
      { timeout: 15_000 },
    );
  } catch {
    logger.warn(
      "WAL checkpoint failed (server may not have sqlite3 CLI). Proceeding with rsync.",
    );
  }

  // Step 2: Download to temp file (atomic — original untouched on failure)
  const tempDb = `${localDb}.downloading`;
  logger.info("Pulling database from server...");
  try {
    rsync(`${conf.server}:${remoteDbPath}`, tempDb);
  } catch (e) {
    if (existsSync(tempDb)) rmSync(tempDb);
    throw e;
  }

  // Step 3: Backup current local DB and swap
  let backupPath: string | null = null;
  let backedUp = false;

  if (existsSync(localDb)) {
    const backupDir = resolve(config.dataDir, "backups");
    mkdirSync(backupDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
    backupPath = resolve(backupDir, `boatrace-tipster-${ts}.db`);
    renameSync(localDb, backupPath);
    backedUp = true;
    logger.info(`Backed up local DB to ${backupPath}`);

    // Rotate old backups
    const backups = readdirSync(backupDir)
      .filter((f) => f.startsWith("boatrace-tipster-") && f.endsWith(".db"))
      .sort()
      .reverse();
    for (const old of backups.slice(BACKUP_KEEP_COUNT)) {
      rmSync(resolve(backupDir, old));
      logger.info(`Rotated old backup: ${old}`);
    }

    // Clean up WAL/SHM files
    for (const ext of ["-wal", "-shm"]) {
      const walPath = `${localDb}${ext}`;
      if (existsSync(walPath)) rmSync(walPath);
    }
  }

  renameSync(tempDb, localDb);
  logger.info("Database sync complete.");

  return { backedUp, backupPath };
}
