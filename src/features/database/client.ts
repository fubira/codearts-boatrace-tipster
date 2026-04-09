import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { CREATE_TABLES_SQL, MIGRATIONS, SCHEMA_VERSION } from "./schema";

let db: Database | null = null;

export function getDatabase(): Database {
  if (db) return db;

  mkdirSync(dirname(config.dbPath), { recursive: true });
  db = new Database(config.dbPath);
  db.exec("PRAGMA journal_mode=WAL");
  db.exec("PRAGMA busy_timeout=5000");
  db.exec("PRAGMA foreign_keys=ON");

  logger.debug(`Database opened: ${config.dbPath}`);
  return db;
}

export function initializeDatabase(): void {
  const database = getDatabase();
  database.exec(CREATE_TABLES_SQL);

  const row = database
    .query("SELECT MAX(version) as v FROM schema_version")
    .get() as { v: number | null } | null;
  const currentVersion = row?.v ?? 0;

  if (currentVersion < SCHEMA_VERSION) {
    for (let v = currentVersion + 1; v <= SCHEMA_VERSION; v++) {
      const migration = MIGRATIONS[v];
      if (migration) {
        database.exec(migration);
        logger.info(`Applied migration v${v}`);
      }
    }
    database
      .query("INSERT INTO schema_version (version) VALUES (?)")
      .run(SCHEMA_VERSION);
    logger.info(`Database schema updated to v${SCHEMA_VERSION}`);
  } else {
    logger.debug(`Database schema up to date (v${currentVersion})`);
  }
}

export function closeDatabase(): void {
  if (db) {
    db.close();
    db = null;
    logger.debug("Database closed");
  }
}
