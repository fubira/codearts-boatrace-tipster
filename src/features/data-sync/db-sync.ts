/** Database sync: pull from server (default) or push to server (--push). */

import { execSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
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
  logger.info("Database pull complete.");

  return { backedUp, backupPath };
}

/** Push local database to server safely.
 *
 * 1. Local WAL checkpoint (TRUNCATE) — merge WAL into main DB
 * 2. Remote WAL/SHM cleanup — remove stale WAL files
 * 3. rsync main DB file only (no WAL/SHM)
 *
 * IMPORTANT: The server runner keeps a SQLite connection open even in sleep.
 * This is safe because: remote WAL/SHM are deleted before rsync, and the
 * runner will create a new WAL on its next write. However, if the runner
 * is actively writing (mid-transaction), the push could race. Prefer
 * pushing when the runner is stopped or between daily cycles.
 */
export function pushDb(conf: SyncConfig): void {
  const localDb = config.dbPath;
  const remoteDbPath = `${conf.prodDir}/data/boatrace-tipster.db`;

  // Step 1: Local WAL checkpoint
  logger.info("Running local WAL checkpoint...");
  try {
    execSync(`sqlite3 "${localDb}" "PRAGMA wal_checkpoint(TRUNCATE);"`, {
      encoding: "utf-8",
      timeout: 30_000,
    });
  } catch (e) {
    throw new Error(`Local WAL checkpoint failed: ${e}`);
  }

  // Verify no WAL remaining
  const walPath = `${localDb}-wal`;
  if (existsSync(walPath)) {
    const walSize = statSync(walPath).size;
    if (walSize > 0) {
      throw new Error(
        `WAL file still has data (${walSize} bytes) after checkpoint. Another process may be writing to the DB.`,
      );
    }
  }

  // Step 2: Remove remote WAL/SHM files
  logger.info("Cleaning remote WAL/SHM files...");
  try {
    sshExec(conf, `rm -f "${remoteDbPath}-wal" "${remoteDbPath}-shm"`, {
      timeout: 10_000,
    });
  } catch (e) {
    throw new Error(`Failed to clean remote WAL/SHM: ${e}`);
  }

  // Step 3: Push main DB file only
  logger.info("Pushing database to server...");
  rsync(localDb, `${conf.server}:${remoteDbPath}`);

  logger.info("Database push complete.");
}
