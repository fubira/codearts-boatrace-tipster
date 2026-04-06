/** Database integrity checks */

import type { Database } from "bun:sqlite";
import { getDatabase } from "./client";

export interface CheckResult {
  name: string;
  status: "ok" | "warn" | "error";
  detail: string;
}

function query(db: Database, sql: string): number {
  const row = db.prepare(sql).get() as { cnt: number };
  return row.cnt;
}

/** Zero-count check: query should return 0 for ok, positive for error/warn */
function zeroCheck(
  db: Database,
  name: string,
  sql: string,
  okMsg: string,
  failMsg: (n: number) => string,
  status: "error" | "warn" = "error",
): CheckResult {
  const count = query(db, sql);
  return {
    name,
    status: count === 0 ? "ok" : status,
    detail: count === 0 ? okMsg : failMsg(count),
  };
}

function checkStructuralIntegrity(db: Database): CheckResult[] {
  return [
    zeroCheck(
      db,
      "Orphan entries",
      "SELECT COUNT(*) as cnt FROM race_entries re WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = re.race_id)",
      "No orphan race_entries",
      (n) => `${n} race_entries reference non-existent races`,
    ),
    zeroCheck(
      db,
      "Orphan payouts",
      "SELECT COUNT(*) as cnt FROM race_payouts rp WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = rp.race_id)",
      "No orphan race_payouts",
      (n) => `${n} race_payouts reference non-existent races`,
    ),
    zeroCheck(
      db,
      "Duplicate races",
      "SELECT COUNT(*) as cnt FROM (SELECT stadium_id, race_date, race_number FROM races GROUP BY stadium_id, race_date, race_number HAVING COUNT(*) > 1)",
      "No duplicate races",
      (n) => `${n} duplicate race keys found`,
    ),
    zeroCheck(
      db,
      "Duplicate entries",
      "SELECT COUNT(*) as cnt FROM (SELECT race_id, boat_number FROM race_entries GROUP BY race_id, boat_number HAVING COUNT(*) > 1)",
      "No duplicate entries",
      (n) => `${n} duplicate entry keys found`,
    ),
    zeroCheck(
      db,
      "Entry count",
      "SELECT COUNT(*) as cnt FROM (SELECT race_id, COUNT(*) as c FROM race_entries GROUP BY race_id HAVING c != 6)",
      "All races have 6 entries",
      (n) => `${n} races have entry count ≠ 6`,
    ),
    zeroCheck(
      db,
      "Races without entries",
      "SELECT COUNT(*) as cnt FROM races r WHERE NOT EXISTS (SELECT 1 FROM race_entries re WHERE re.race_id = r.id)",
      "All races have entries",
      (n) => `${n} races have no entries`,
    ),
  ];
}

function checkValueSanity(db: Database): CheckResult[] {
  return [
    zeroCheck(
      db,
      "Boat number range",
      "SELECT COUNT(*) as cnt FROM race_entries WHERE boat_number < 1 OR boat_number > 6",
      "All boat numbers in 1-6",
      (n) => `${n} entries with boat_number outside 1-6`,
    ),
    zeroCheck(
      db,
      "Finish position range",
      "SELECT COUNT(*) as cnt FROM race_entries WHERE finish_position IS NOT NULL AND (finish_position < 1 OR finish_position > 6)",
      "All finish positions in 1-6 or NULL",
      (n) => `${n} entries with finish_position outside 1-6`,
    ),
    zeroCheck(
      db,
      "Race number range",
      "SELECT COUNT(*) as cnt FROM races WHERE race_number < 1 OR race_number > 12",
      "All race numbers in 1-12",
      (n) => `${n} races with race_number outside 1-12`,
    ),
  ];
}

function checkResultCoverage(db: Database): CheckResult[] {
  const totalRaces = query(db, "SELECT COUNT(*) as cnt FROM races");
  const noResultRaces = query(
    db,
    "SELECT COUNT(*) as cnt FROM races WHERE weather IS NULL",
  );
  const noResultPct =
    totalRaces > 0 ? ((noResultRaces / totalRaces) * 100).toFixed(1) : "0";

  const totalEntries = query(db, "SELECT COUNT(*) as cnt FROM race_entries");
  const noFinish = query(
    db,
    "SELECT COUNT(*) as cnt FROM race_entries WHERE finish_position IS NULL",
  );
  const noFinishPct =
    totalEntries > 0 ? ((noFinish / totalEntries) * 100).toFixed(1) : "0";

  // Recent result coverage: races in last 7 days with all entries having NULL finish
  const recentVoidRaces = query(
    db,
    `SELECT COUNT(*) as cnt FROM races r
     WHERE r.race_date >= date('now', '-7 days')
       AND r.weather IS NOT NULL
       AND NOT EXISTS (
         SELECT 1 FROM race_entries re
         WHERE re.race_id = r.id AND re.finish_position IS NOT NULL
       )`,
  );
  const recentTotal = query(
    db,
    "SELECT COUNT(*) as cnt FROM races WHERE race_date >= date('now', '-7 days') AND weather IS NOT NULL",
  );

  return [
    {
      name: "Result coverage",
      status: noResultRaces / totalRaces < 0.05 ? "ok" : "warn",
      detail: `${totalRaces - noResultRaces}/${totalRaces} races have results (${noResultRaces} without, ${noResultPct}%)`,
    },
    {
      name: "Finish position coverage",
      status: noFinish / totalEntries < 0.05 ? "ok" : "warn",
      detail: `${totalEntries - noFinish}/${totalEntries} entries have finish position (${noFinishPct}% NULL)`,
    },
    {
      name: "Recent results (7d)",
      status: recentVoidRaces === 0 ? "ok" : "error",
      detail:
        recentVoidRaces === 0
          ? `All ${recentTotal} completed races have finish data`
          : `${recentVoidRaces}/${recentTotal} completed races missing ALL finish positions`,
    },
  ];
}

function checkMonthlyCompleteness(db: Database): CheckResult[] {
  const monthlyRows = db
    .prepare(
      `SELECT substr(race_date,1,7) as month, COUNT(*) as cnt
       FROM races GROUP BY month ORDER BY month`,
    )
    .all() as { month: string; cnt: number }[];

  const expectedMonths = buildExpectedMonths(monthlyRows);
  const missingMonths: string[] = [];
  const lowMonths: string[] = [];

  for (const { month, expected } of expectedMonths) {
    const actual = monthlyRows.find((r) => r.month === month);
    if (!actual || actual.cnt === 0) {
      missingMonths.push(month);
    } else if (actual.cnt < expected * 0.5) {
      lowMonths.push(`${month}(${actual.cnt}/${expected})`);
    }
  }

  const results: CheckResult[] = [];

  if (missingMonths.length > 0) {
    results.push({
      name: "Missing months",
      status: "warn",
      detail: `Missing: ${missingMonths.join(", ")}`,
    });
  }

  // Exclude the last month (likely partial/in-progress)
  const lastMonth = monthlyRows[monthlyRows.length - 1]?.month;
  const lowMonthsFiltered = lowMonths.filter((m) => !m.startsWith(lastMonth));
  if (lowMonthsFiltered.length > 0) {
    results.push({
      name: "Low data months",
      status: "warn",
      detail: `Below 50% expected: ${lowMonthsFiltered.join(", ")}`,
    });
  }

  if (missingMonths.length === 0 && lowMonths.length === 0) {
    results.push({
      name: "Monthly completeness",
      status: "ok",
      detail: `All months within expected range (${monthlyRows.length} months)`,
    });
  }

  return results;
}

function checkPayoutCoverage(db: Database): CheckResult[] {
  return [
    zeroCheck(
      db,
      "Payout coverage",
      "SELECT COUNT(*) as cnt FROM races r WHERE weather IS NOT NULL AND NOT EXISTS (SELECT 1 FROM race_payouts rp WHERE rp.race_id = r.id)",
      "All completed races have payouts",
      (n) => `${n} completed races missing payouts`,
      "warn",
    ),
  ];
}

/** Run all integrity checks */
export function checkIntegrity(db?: Database): CheckResult[] {
  const database = db ?? getDatabase();
  return [
    ...checkStructuralIntegrity(database),
    ...checkValueSanity(database),
    ...checkResultCoverage(database),
    ...checkMonthlyCompleteness(database),
    ...checkPayoutCoverage(database),
  ];
}

/** Build expected monthly race counts based on the date range in the DB */
function buildExpectedMonths(
  monthlyRows: { month: string; cnt: number }[],
): { month: string; expected: number }[] {
  if (monthlyRows.length === 0) return [];

  const first = monthlyRows[0].month;
  const last = monthlyRows[monthlyRows.length - 1].month;

  const result: { month: string; expected: number }[] = [];
  let [year, month] = first.split("-").map(Number);
  const [lastYear, lastMonth] = last.split("-").map(Number);

  // Median of existing months as baseline expected count
  const counts = monthlyRows
    .map((r) => r.cnt)
    .filter((c) => c > 100)
    .sort((a, b) => a - b);
  const median =
    counts.length > 0 ? counts[Math.floor(counts.length / 2)] : 4500;

  while (year < lastYear || (year === lastYear && month <= lastMonth)) {
    const m = `${year}-${String(month).padStart(2, "0")}`;
    result.push({ month: m, expected: median });
    month++;
    if (month > 12) {
      month = 1;
      year++;
    }
  }

  return result;
}
