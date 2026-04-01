import { copyFileSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { Command } from "commander";

const BACKUP_KEEP_COUNT = 7;

export const backupCommand = new Command("backup")
  .description("Create a timestamped database backup")
  .option(
    "-n, --keep <count>",
    "number of backups to keep",
    Number.parseInt,
    BACKUP_KEEP_COUNT,
  )
  .action((opts) => {
    const backupDir = resolve(config.dataDir, "backups");
    mkdirSync(backupDir, { recursive: true });

    const ts = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
    const backupPath = resolve(backupDir, `boatrace-tipster-${ts}.db`);

    copyFileSync(config.dbPath, backupPath);
    logger.info(`Backup created: ${backupPath}`);

    // Rotate old backups
    const backups = readdirSync(backupDir)
      .filter((f) => f.startsWith("boatrace-tipster-") && f.endsWith(".db"))
      .sort()
      .reverse();

    const keepCount = opts.keep ?? BACKUP_KEEP_COUNT;
    for (const old of backups.slice(keepCount)) {
      rmSync(resolve(backupDir, old));
      logger.info(`Removed old backup: ${old}`);
    }

    logger.info(
      `${backups.length} backup(s) total, keeping latest ${keepCount}`,
    );
  });
