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

/** Run all integrity checks */
export function checkIntegrity(db?: Database): CheckResult[] {
  const database = db ?? getDatabase();
  const results: CheckResult[] = [];

  // --- Structural integrity ---

  const orphanEntries = query(
    database,
    "SELECT COUNT(*) as cnt FROM race_entries re WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = re.race_id)",
  );
  results.push({
    name: "Orphan entries",
    status: orphanEntries === 0 ? "ok" : "error",
    detail:
      orphanEntries === 0
        ? "No orphan race_entries"
        : `${orphanEntries} race_entries reference non-existent races`,
  });

  const orphanPayouts = query(
    database,
    "SELECT COUNT(*) as cnt FROM race_payouts rp WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = rp.race_id)",
  );
  results.push({
    name: "Orphan payouts",
    status: orphanPayouts === 0 ? "ok" : "error",
    detail:
      orphanPayouts === 0
        ? "No orphan race_payouts"
        : `${orphanPayouts} race_payouts reference non-existent races`,
  });

  const duplicateRaces = query(
    database,
    "SELECT COUNT(*) as cnt FROM (SELECT stadium_id, race_date, race_number FROM races GROUP BY stadium_id, race_date, race_number HAVING COUNT(*) > 1)",
  );
  results.push({
    name: "Duplicate races",
    status: duplicateRaces === 0 ? "ok" : "error",
    detail:
      duplicateRaces === 0
        ? "No duplicate races"
        : `${duplicateRaces} duplicate race keys found`,
  });

  const duplicateEntries = query(
    database,
    "SELECT COUNT(*) as cnt FROM (SELECT race_id, boat_number FROM race_entries GROUP BY race_id, boat_number HAVING COUNT(*) > 1)",
  );
  results.push({
    name: "Duplicate entries",
    status: duplicateEntries === 0 ? "ok" : "error",
    detail:
      duplicateEntries === 0
        ? "No duplicate entries"
        : `${duplicateEntries} duplicate entry keys found`,
  });

  // --- Entry count per race ---

  const wrongEntryCount = query(
    database,
    "SELECT COUNT(*) as cnt FROM (SELECT race_id, COUNT(*) as c FROM race_entries GROUP BY race_id HAVING c != 6)",
  );
  results.push({
    name: "Entry count",
    status: wrongEntryCount === 0 ? "ok" : "error",
    detail:
      wrongEntryCount === 0
        ? "All races have 6 entries"
        : `${wrongEntryCount} races have entry count ≠ 6`,
  });

  const racesNoEntries = query(
    database,
    "SELECT COUNT(*) as cnt FROM races r WHERE NOT EXISTS (SELECT 1 FROM race_entries re WHERE re.race_id = r.id)",
  );
  results.push({
    name: "Races without entries",
    status: racesNoEntries === 0 ? "ok" : "error",
    detail:
      racesNoEntries === 0
        ? "All races have entries"
        : `${racesNoEntries} races have no entries`,
  });

  // --- Value sanity ---

  const badBoatNumber = query(
    database,
    "SELECT COUNT(*) as cnt FROM race_entries WHERE boat_number < 1 OR boat_number > 6",
  );
  results.push({
    name: "Boat number range",
    status: badBoatNumber === 0 ? "ok" : "error",
    detail:
      badBoatNumber === 0
        ? "All boat numbers in 1-6"
        : `${badBoatNumber} entries with boat_number outside 1-6`,
  });

  const badFinishPos = query(
    database,
    "SELECT COUNT(*) as cnt FROM race_entries WHERE finish_position IS NOT NULL AND (finish_position < 1 OR finish_position > 6)",
  );
  results.push({
    name: "Finish position range",
    status: badFinishPos === 0 ? "ok" : "error",
    detail:
      badFinishPos === 0
        ? "All finish positions in 1-6 or NULL"
        : `${badFinishPos} entries with finish_position outside 1-6`,
  });

  const badRaceNumber = query(
    database,
    "SELECT COUNT(*) as cnt FROM races WHERE race_number < 1 OR race_number > 12",
  );
  results.push({
    name: "Race number range",
    status: badRaceNumber === 0 ? "ok" : "error",
    detail:
      badRaceNumber === 0
        ? "All race numbers in 1-12"
        : `${badRaceNumber} races with race_number outside 1-12`,
  });

  // --- Result coverage ---

  const totalRaces = query(database, "SELECT COUNT(*) as cnt FROM races");
  const noResultRaces = query(
    database,
    "SELECT COUNT(*) as cnt FROM races WHERE weather IS NULL",
  );
  const noResultPct =
    totalRaces > 0 ? ((noResultRaces / totalRaces) * 100).toFixed(1) : "0";
  results.push({
    name: "Result coverage",
    status: noResultRaces / totalRaces < 0.05 ? "ok" : "warn",
    detail: `${totalRaces - noResultRaces}/${totalRaces} races have results (${noResultRaces} without, ${noResultPct}%)`,
  });

  const totalEntries = query(
    database,
    "SELECT COUNT(*) as cnt FROM race_entries",
  );
  const noFinish = query(
    database,
    "SELECT COUNT(*) as cnt FROM race_entries WHERE finish_position IS NULL",
  );
  const noFinishPct =
    totalEntries > 0 ? ((noFinish / totalEntries) * 100).toFixed(1) : "0";
  results.push({
    name: "Finish position coverage",
    status: noFinish / totalEntries < 0.05 ? "ok" : "warn",
    detail: `${totalEntries - noFinish}/${totalEntries} entries have finish position (${noFinishPct}% NULL)`,
  });

  // --- Monthly completeness ---

  const monthlyRows = database
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

  // --- Payout coverage ---

  const noPayoutRaces = query(
    database,
    "SELECT COUNT(*) as cnt FROM races r WHERE weather IS NOT NULL AND NOT EXISTS (SELECT 1 FROM race_payouts rp WHERE rp.race_id = r.id)",
  );
  results.push({
    name: "Payout coverage",
    status: noPayoutRaces === 0 ? "ok" : "warn",
    detail:
      noPayoutRaces === 0
        ? "All completed races have payouts"
        : `${noPayoutRaces} completed races missing payouts`,
  });

  return results;
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
