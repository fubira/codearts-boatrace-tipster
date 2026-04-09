import {
  loadSyncConfig,
  pushDb,
  syncCache,
  syncDb,
  syncSnapshots,
  verify,
} from "@/features/data-sync";
import { closeDatabase, initializeDatabase } from "@/features/database";
import { checkIntegrity } from "@/features/database/integrity";
import { logger } from "@/shared/logger";
import { Command } from "commander";

export const dataCommand = new Command("data").description(
  "Data management: sync, verify, and inspect data across environments",
);

// --- data sync ---

dataCommand
  .command("sync")
  .description("Sync database and cache with server (default: pull)")
  .option("--cache-only", "sync HTML cache only")
  .option("--db-only", "sync database only")
  .option("--dry-run", "show what would transfer without syncing")
  .option("--push", "push local DB to server (instead of pull)")
  .action((opts) => {
    if (opts.cacheOnly && opts.dbOnly) {
      console.error(
        "Error: --cache-only and --db-only cannot be used together",
      );
      process.exit(1);
    }

    const conf = loadSyncConfig();

    if (opts.push) {
      // Push mode: local → server (DB only, no cache)
      if (opts.cacheOnly) {
        console.error("Error: --push only supports database sync");
        process.exit(1);
      }
      if (opts.dryRun) {
        logger.info("DB push skipped in dry-run mode");
      } else {
        pushDb(conf);
        logger.info(
          "Push complete. Run 'bun run start data verify' to confirm.",
        );
      }
      return;
    }

    // Pull mode (default): server → local
    const doCache = !opts.dbOnly;
    const doDb = !opts.cacheOnly;

    if (doCache) {
      const result = syncCache(conf, { dryRun: opts.dryRun });
      logger.info(
        `Cache sync: pulled ${result.pulled}, pushed ${result.pushed} file(s)`,
      );
    }

    if (doDb) {
      if (opts.dryRun) {
        logger.info("DB sync skipped in dry-run mode");
      } else {
        const result = syncDb(conf);
        if (result.backedUp) {
          logger.info(`Backup: ${result.backupPath}`);
        }
      }
    }

    // Snapshot sync: only in full sync mode (skip when --db-only or --cache-only)
    if (doCache && doDb) {
      const snapResult = syncSnapshots(conf, { dryRun: opts.dryRun });
      logger.info(
        `Snapshot sync: pulled ${snapResult.pulled}, pushed ${snapResult.pushed} file(s)`,
      );
    }

    logger.info(
      "Sync complete. Run 'bun run start data verify' to confirm integrity.",
    );
  });

// --- data verify ---

dataCommand
  .command("verify")
  .description("Compare local and server data")
  .action(() => {
    const conf = loadSyncConfig();
    const result = verify(conf);

    const nameWidth = Math.max(...result.checks.map((c) => c.name.length));

    console.log(
      `${"Check".padEnd(nameWidth)}  ${"Local".padEnd(20)}  ${"Remote".padEnd(20)}  Status`,
    );
    console.log("-".repeat(nameWidth + 50));

    for (const c of result.checks) {
      const status = c.match ? "OK" : "\x1b[31mMISMATCH ←\x1b[0m";
      console.log(
        `${c.name.padEnd(nameWidth)}  ${c.local.padEnd(20)}  ${c.remote.padEnd(20)}  ${status}`,
      );
    }

    console.log();
    if (result.allPassed) {
      console.log(
        "\x1b[32mAll checks passed. Local and server data are in sync.\x1b[0m",
      );
    } else {
      const mismatches = result.checks.filter((c) => !c.match).length;
      console.log(
        `\x1b[31m${mismatches} mismatch(es) found. Run 'bun run start data sync' to resolve.\x1b[0m`,
      );
      process.exit(1);
    }
  });

// --- data test ---

dataCommand
  .command("test")
  .description("Run local database integrity checks")
  .action(() => {
    initializeDatabase();

    try {
      const results = checkIntegrity();
      let hasError = false;
      let hasWarn = false;

      for (const r of results) {
        const icon = r.status === "ok" ? "✓" : r.status === "warn" ? "!" : "✗";
        const color =
          r.status === "ok"
            ? "\x1b[32m"
            : r.status === "warn"
              ? "\x1b[33m"
              : "\x1b[31m";
        console.log(`${color}  ${icon} ${r.name}\x1b[0m: ${r.detail}`);

        if (r.status === "error") hasError = true;
        if (r.status === "warn") hasWarn = true;
      }

      console.log();
      if (hasError) {
        console.log(
          "\x1b[31mErrors found. Database may have integrity issues.\x1b[0m",
        );
        process.exit(1);
      } else if (hasWarn) {
        console.log("\x1b[33mWarnings found. Review above for details.\x1b[0m");
      } else {
        console.log("\x1b[32mAll checks passed.\x1b[0m");
      }
    } finally {
      closeDatabase();
    }
  });

// --- data fingerprint ---

dataCommand
  .command("fingerprint")
  .description("Display local database statistics")
  .action(() => {
    initializeDatabase();

    try {
      const { getDatabase } = require("@/features/database");
      const db = getDatabase();

      const queries: [string, string][] = [
        ["Schema version", "SELECT MAX(version) FROM schema_version"],
        ["Races", "SELECT COUNT(*) FROM races"],
        ["Racers", "SELECT COUNT(*) FROM racers"],
        ["Race entries", "SELECT COUNT(*) FROM race_entries"],
        ["Race payouts", "SELECT COUNT(*) FROM race_payouts"],
        [
          "Date range",
          "SELECT MIN(race_date) || ' ~ ' || MAX(race_date) FROM races",
        ],
        ["Entry ID sum", "SELECT SUM(id) FROM race_entries"],
        ["Payout ID sum", "SELECT SUM(id) FROM race_payouts"],
      ];

      console.log(`Database: ${require("@/shared/config").config.dbPath}`);

      for (const [label, sql] of queries) {
        try {
          const row = db.prepare(sql).get() as Record<string, unknown>;
          const value = Object.values(row)[0];
          console.log(`  ${label}: ${value}`);
        } catch {
          console.log(`  ${label}: ERROR`);
        }
      }
    } finally {
      closeDatabase();
    }
  });
