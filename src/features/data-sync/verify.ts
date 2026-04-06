/** Compare local and remote database state. */

import { execSync } from "node:child_process";
import { statSync } from "node:fs";
import { config } from "@/shared/config";
import { sshExec } from "./ssh";
import type { SyncConfig } from "./sync-config";

export interface VerifyCheck {
  name: string;
  local: string;
  remote: string;
  match: boolean;
}

export interface VerifyResult {
  checks: VerifyCheck[];
  allPassed: boolean;
}

function localQuery(sql: string): string {
  try {
    return execSync(`sqlite3 "${config.dbPath}" "${sql}"`, {
      encoding: "utf-8",
      timeout: 10_000,
    }).trim();
  } catch {
    return "ERROR";
  }
}

function remoteQuery(conf: SyncConfig, sql: string): string {
  try {
    return sshExec(
      conf,
      `sqlite3 ${conf.prodDir}/data/boatrace-tipster.db "${sql}"`,
      { timeout: 10_000 },
    );
  } catch {
    return "ERROR";
  }
}

function localCacheCount(): string {
  try {
    return execSync(
      `find "${config.cacheDir}" -name "*.html.gz" -type f | wc -l`,
      { encoding: "utf-8", timeout: 30_000 },
    ).trim();
  } catch {
    return "ERROR";
  }
}

function remoteCacheCount(conf: SyncConfig): string {
  try {
    return sshExec(
      conf,
      `find ${conf.prodDir}/data/cache -name "*.html.gz" -type f | wc -l`,
      { timeout: 30_000 },
    );
  } catch {
    return "ERROR";
  }
}

export function verify(conf: SyncConfig): VerifyResult {
  const checks: VerifyCheck[] = [];

  // Schema version
  const schemaQ = "SELECT MAX(version) FROM schema_version;";
  const localSchema = localQuery(schemaQ);
  const remoteSchema = remoteQuery(conf, schemaQ);
  checks.push({
    name: "Schema version",
    local: localSchema,
    remote: remoteSchema,
    match: localSchema === remoteSchema,
  });

  // Row counts
  const tables = [
    { name: "Races", table: "races" },
    { name: "Racers", table: "racers" },
    { name: "Race entries", table: "race_entries" },
    { name: "Race payouts", table: "race_payouts" },
  ];

  for (const t of tables) {
    const q = `SELECT COUNT(*) FROM ${t.table};`;
    const local = localQuery(q);
    const remote = remoteQuery(conf, q);
    checks.push({
      name: `${t.name} count`,
      local,
      remote,
      match: local === remote,
    });
  }

  // ID checksums (detect different data with same count)
  for (const table of ["race_entries", "race_payouts"]) {
    const q = `SELECT SUM(id) FROM ${table};`;
    const local = localQuery(q);
    const remote = remoteQuery(conf, q);
    checks.push({
      name: `${table} id_sum`,
      local,
      remote,
      match: local === remote,
    });
  }

  // Date range
  const dateQ = "SELECT MIN(race_date) || ' ~ ' || MAX(race_date) FROM races;";
  const localDate = localQuery(dateQ);
  const remoteDate = remoteQuery(conf, dateQ);
  checks.push({
    name: "Date range",
    local: localDate,
    remote: remoteDate,
    match: localDate === remoteDate,
  });

  // Cache file count
  const localCache = localCacheCount();
  const remoteCache = remoteCacheCount(conf);
  checks.push({
    name: "Cache files",
    local: localCache,
    remote: remoteCache,
    match: localCache === remoteCache,
  });

  // Result coverage: finish_position NULL rate for recent races
  const resultQ =
    "SELECT COALESCE(ROUND(100.0 * SUM(CASE WHEN re.finish_position IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1), 0) " +
    "FROM races r JOIN race_entries re ON re.race_id = r.id " +
    "WHERE r.race_date >= date('now', '-30 days') AND r.weather IS NOT NULL;";
  const localResultNull = localQuery(resultQ);
  const remoteResultNull = remoteQuery(conf, resultQ);
  checks.push({
    name: "Result NULL% (30d)",
    local: `${localResultNull}%`,
    remote: `${remoteResultNull}%`,
    match: localResultNull === remoteResultNull,
  });

  // Races without results in recent period
  const noResultQ =
    "SELECT COUNT(*) FROM races WHERE race_date >= date('now', '-30 days') AND weather IS NULL;";
  const localNoResult = localQuery(noResultQ);
  const remoteNoResult = remoteQuery(conf, noResultQ);
  checks.push({
    name: "No-result races (30d)",
    local: localNoResult,
    remote: remoteNoResult,
    match: localNoResult === remoteNoResult,
  });

  // DB size
  try {
    const localSize = (statSync(config.dbPath).size / 1024 / 1024).toFixed(1);
    const remoteSize = sshExec(
      conf,
      `stat -c %s ${conf.prodDir}/data/boatrace-tipster.db`,
      { timeout: 5_000 },
    );
    const remoteMB = (Number.parseInt(remoteSize, 10) / 1024 / 1024).toFixed(1);
    checks.push({
      name: "DB size (MB)",
      local: localSize,
      remote: remoteMB,
      match: localSize === remoteMB,
    });
  } catch {
    checks.push({
      name: "DB size (MB)",
      local: "ERROR",
      remote: "ERROR",
      match: false,
    });
  }

  return { checks, allPassed: checks.every((c) => c.match) };
}
