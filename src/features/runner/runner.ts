/** P2 prediction daemon: predict → drift → bet → result. Reads data from DB (scrape-daemon writes). */

import { existsSync, mkdirSync, readdirSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import {
  closeDatabase,
  getDatabase,
  initializeDatabase,
  loadSnapshotTrifectaOdds,
} from "@/features/database";
import { STADIUMS } from "@/features/scraper/sources/boatrace/constants";
import type { PurchaseExecutor } from "@/features/teleboat";
import { config, getActiveModelDir } from "@/shared/config";
import { enableFileLog, logger } from "@/shared/logger";
import { pythonCommand } from "@/shared/python";
import {
  type BankrollState,
  loadBankrollState,
  saveBankrollState,
} from "./bankroll-state";
import {
  type BetDecision,
  ODDS_LEAD,
  type RaceSlot,
  allDone,
  buildSchedule,
  getActiveRaces,
} from "./race-scheduler";
import {
  notifyDailySummary,
  notifyError,
  notifyPrediction,
  notifyResult,
  notifyShutdown,
  notifyStartup,
  setSlackWebhook,
} from "./slack";

const POLL_INTERVAL_MS = 10_000;
const POLL_INTERVAL_FAST_MS = 5_000;
const FAST_POLL_WINDOW_MIN = 2;

export interface RunnerOptions {
  dryRun: boolean;
  evThreshold: number;
  gap23Threshold: number;
  top3ConcThreshold: number;
  gap12MinThreshold: number;
  betCap: number;
  unitDivisor: number;
  bankroll: number;
  slackWebhookUrl?: string;
  purchaseExecutor?: PurchaseExecutor | null;
}

// ---------------------------------------------------------------------------
// P2 prediction types
// ---------------------------------------------------------------------------

interface P2Ticket {
  combo: string;
  modelProb: number;
  marketOdds: number;
  ev: number;
}

interface P2Prediction {
  top3Conc: number;
  gap23: number;
  gap12: number;
  tickets: P2Ticket[];
  hasExhibition: boolean;
  bcStatus: string; // "full" | "missing" | "partial:n/6"
}

export interface SkippedPrediction {
  skipReason: string;
  top1Boat?: number; // for not_b1_top
  top3Conc?: number; // for top3_conc_low / no_ev_tickets
  gap23?: number; // for gap23_low / no_ev_tickets
  gap12?: number; // for gap12_low
  stadiumId?: number; // for stadium_excluded
  withdrawnBoats?: number[]; // for withdrawal
  bcStatus?: string;
  // P2 tickets the runner would have bought if all filters had passed.
  // Populated for gap12_low / top3_conc_low / gap23_low / no_ev_tickets so
  // filter-stage SKIP logs can show which tickets got cut. null odds/ev
  // mean the market side was missing at predict time.
  wouldBeTickets?: {
    combo: string;
    modelProb: number;
    marketOdds: number | null;
    ev: number | null;
  }[];
}

/** True if any boat in the dash-separated combo is in the refunded set.
 * Used at result time to detect tickets returned due to flying disqualification
 * or other in-race anomalies. Pure function, exported for unit tests. */
export function ticketContainsRefundedBoat(
  combo: string,
  refundedBoats: Set<number>,
): boolean {
  return combo.split("-").some((b) => refundedBoats.has(Number(b)));
}

/** Decide the tag for a result log line. Exported for unit tests. */
export function resultTag(
  won: boolean,
  refundedCount: number,
  ticketCount: number,
): "WIN" | "REFUND" | "LOSE" {
  if (won) return "WIN";
  // Only mark REFUND when there ARE tickets and they are ALL refunded.
  // ticketCount === 0 means we have no ticket info (e.g. cache lost), which
  // is NOT a refund and should fall through to LOSE.
  if (ticketCount > 0 && refundedCount === ticketCount) return "REFUND";
  return "LOSE";
}

/** Format the would-be ticket list for inline SKIP log display. */
function formatWouldBeTickets(
  tickets: NonNullable<SkippedPrediction["wouldBeTickets"]> | undefined,
): string {
  if (!tickets || tickets.length === 0) return "";
  const parts = tickets.map((t) => {
    if (t.marketOdds == null || t.ev == null) {
      return `${t.combo}(odds=n/a)`;
    }
    return `${t.combo}(EV ${(t.ev * 100).toFixed(0)}% @${t.marketOdds.toFixed(1)})`;
  });
  return ` | cut: ${parts.join(", ")}`;
}

export function formatSkipReason(
  s: SkippedPrediction,
  opts: Pick<
    RunnerOptions,
    "evThreshold" | "gap23Threshold" | "top3ConcThreshold" | "gap12MinThreshold"
  >,
): string {
  const r = s.skipReason;
  const concTh = (opts.top3ConcThreshold * 100).toFixed(0);
  const gapTh = (opts.gap23Threshold * 100).toFixed(1);
  const gap12Th = (opts.gap12MinThreshold * 100).toFixed(1);
  const evTh = (opts.evThreshold * 100).toFixed(0);
  if (r === "not_b1_top" && s.top1Boat != null) {
    return `not_b1_top (top1=${s.top1Boat}号艇)`;
  }
  if (r === "stadium_excluded") {
    return "stadium_excluded";
  }
  if (r === "withdrawal") {
    const boats = s.withdrawnBoats?.length
      ? s.withdrawnBoats.map((b) => `${b}号艇`).join(",")
      : "?";
    return `withdrawal (${boats}欠場)`;
  }
  const cut = formatWouldBeTickets(s.wouldBeTickets);
  if (r === "gap12_low" && s.gap12 != null) {
    return `gap12_low (${(s.gap12 * 100).toFixed(1)}% < th=${gap12Th}%)${cut}`;
  }
  if (r === "top3_conc_low" && s.top3Conc != null) {
    return `top3_conc_low (${(s.top3Conc * 100).toFixed(0)}% < th=${concTh}%)${cut}`;
  }
  if (r === "gap23_low" && s.gap23 != null) {
    return `gap23_low (${(s.gap23 * 100).toFixed(1)}% < th=${gapTh}%)${cut}`;
  }
  if (r === "no_ev_tickets") {
    const parts: string[] = [];
    if (s.top3Conc != null)
      parts.push(`conc=${(s.top3Conc * 100).toFixed(0)}%`);
    if (s.gap23 != null) parts.push(`gap23=${(s.gap23 * 100).toFixed(1)}%`);
    return `no_ev_tickets (${parts.join(", ")}, all EV<${evTh}%)${cut}`;
  }
  return r;
}

type CachedPrediction = P2Prediction | SkippedPrediction;
type PredictionCache = Map<number, CachedPrediction>;

interface RunnerState {
  schedule: RaceSlot[];
  bets: Map<number, BetDecision>;
  results: Map<number, { won: boolean; payout: number }>;
  predictionCache: PredictionCache | null;
  bankroll: number;
  bankrollState: BankrollState; // persisted state (carries across restarts)
  date: string;
  lastStatusLine: string;
  snapshotPath?: string;
  // Counters for daily summary
  skipCounts: {
    not_b1_top: number;
    gap12_low: number;
    top3_conc_low: number;
    gap23_low: number;
    no_ev_tickets: number;
    drift_drop: number;
    stadium_excluded: number;
    withdrawal: number;
  };
  t1DroppedTickets: number;
}

function todayJST(): string {
  return new Date()
    .toLocaleDateString("sv-SE", { timeZone: "Asia/Tokyo" })
    .replace(/\//g, "-");
}

/**
 * Sleep until 07:00 JST. `when="today"` is a no-op if it's already past 7 AM
 * (used at startup); `when="tomorrow"` always targets the next day (used at
 * day rollover). Anchoring day-init to 07:00 sidesteps the midnight watchtower
 * race condition where runner restarts before scraper finishes pulling races.
 */
async function waitUntilJST7am(when: "today" | "tomorrow"): Promise<void> {
  const now = Date.now();
  const jstNow = now + 9 * 3600_000;
  const jstMidnight = jstNow - (jstNow % (24 * 3600_000));
  const offsetDays = when === "tomorrow" ? 1 : 0;
  const targetJst = jstMidnight + (offsetDays * 24 + 7) * 3600_000;
  const sleepMs = targetJst - 9 * 3600_000 - now;

  if (sleepMs <= 0) return;

  const hours = sleepMs / 3600_000;
  logger.info(`Sleeping until ${when} 07:00 JST (${hours.toFixed(1)}h)`);
  await Bun.sleep(sleepMs);
}

// ---------------------------------------------------------------------------
// Stats snapshot
// ---------------------------------------------------------------------------

const SNAPSHOTS_DIR = resolve(config.projectRoot, "data/stats-snapshots");
const SNAPSHOT_RETENTION_DAYS = 30;

async function buildStatsSnapshot(date: string): Promise<string | undefined> {
  const yesterday = new Date(date);
  yesterday.setDate(yesterday.getDate() - 1);
  const throughDate = yesterday.toISOString().slice(0, 10);

  const snapshotPath = resolve(SNAPSHOTS_DIR, `${throughDate}.db`);

  if (existsSync(snapshotPath)) {
    logger.info(`Snapshot already exists: ${throughDate}.db`);
    rotateSnapshots();
    return snapshotPath;
  }

  logger.info(`Building stats snapshot through ${throughDate}...`);
  mkdirSync(SNAPSHOTS_DIR, { recursive: true });

  const { cmd, cwd } = pythonCommand("scripts.build_snapshot", [
    "--through-date",
    throughDate,
    "--db-path",
    config.dbPath,
    "--output",
    snapshotPath,
  ]);

  const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe", cwd });
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const exitCode = await proc.exited;
  if (stderr) logger.debug(stderr.trim());
  if (stdout.trim()) logger.debug(stdout.trim());
  if (exitCode !== 0) {
    logger.error(`Snapshot build failed (exit ${exitCode})`);
    return undefined;
  }

  logger.info(`Snapshot built: ${throughDate}.db`);
  rotateSnapshots();
  return snapshotPath;
}

function rotateSnapshots(): void {
  try {
    const files = readdirSync(SNAPSHOTS_DIR)
      .filter((f) => f.endsWith(".db"))
      .sort();
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - SNAPSHOT_RETENTION_DAYS);
    const cutoffStr = cutoff.toISOString().slice(0, 10);

    for (const f of files) {
      const dateStr = f.replace(".db", "");
      if (dateStr < cutoffStr) {
        unlinkSync(resolve(SNAPSHOTS_DIR, f));
        logger.debug(`Rotated old snapshot: ${f}`);
      }
    }
  } catch {
    // Ignore rotation errors
  }
}

// ---------------------------------------------------------------------------
// Prediction (P2)
// ---------------------------------------------------------------------------

interface PredictionResult {
  predictions: {
    raceId: number;
    top3Conc: number;
    gap23: number;
    gap12: number;
    tickets: P2Ticket[];
    hasExhibition: boolean;
    bcStatus: string;
  }[];
  evaluatedRaceIds: number[];
  skipped: Record<
    number,
    {
      reason: string;
      top1_boat?: number;
      top3_conc?: number;
      gap23?: number;
      gap12?: number;
      stadium_id?: number;
      withdrawn_boats?: number[];
      bc_status?: string;
      would_be_tickets?: {
        combo: string;
        model_prob: number;
        market_odds: number | null;
        ev: number | null;
      }[];
    }
  >;
}

async function runPrediction(
  date: string,
  opts: RunnerOptions,
  snapshotPath?: string,
  raceIds?: number[],
): Promise<PredictionResult> {
  const modelDir = getActiveModelDir();
  const args = [
    "--date",
    date,
    "--model-dir",
    modelDir,
    "--db-path",
    config.dbPath,
  ];
  if (snapshotPath) {
    args.push("--snapshot", snapshotPath);
  }
  if (raceIds && raceIds.length > 0) {
    args.push("--race-ids", raceIds.join(","));
  }
  args.push("--use-snapshots");
  const { cmd, cwd } = pythonCommand("scripts.predict_p2", args);
  const PREDICTION_TIMEOUT_MS = 60_000;
  const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe", cwd });

  const exited = new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => {
      proc.kill(9);
      reject(
        new Error(`predict timed out after ${PREDICTION_TIMEOUT_MS / 1000}s`),
      );
    }, PREDICTION_TIMEOUT_MS);

    (async () => {
      const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]);
      clearTimeout(timer);
      if (stderr) logger.debug(stderr.trim());
      const exitCode = await proc.exited;
      if (exitCode !== 0) {
        reject(new Error(`predict failed (exit ${exitCode}): ${stderr}`));
      } else {
        resolve(stdout);
      }
    })();
  });

  const stdout = await exited;
  const result = JSON.parse(stdout);

  if (result.stats) {
    const s = result.stats;
    logger.debug(
      `P2 stats: ${s.total}R → B1 top ${s.b1_top} → gap12 ${s.gap12_pass ?? "?"} → conc ${s.conc_pass} → gap23 ${s.gap23_pass} → predicted ${s.predicted}`,
    );
  }

  const predictions = result.predictions.map(
    (p: {
      race_id: number;
      top3_conc: number;
      gap23: number;
      gap12: number;
      tickets: {
        combo: string;
        model_prob: number;
        market_odds: number;
        ev: number;
      }[];
      has_exhibition: boolean;
      bc_status: string;
    }) => ({
      raceId: p.race_id,
      top3Conc: p.top3_conc,
      gap23: p.gap23,
      gap12: p.gap12 ?? 0,
      tickets: p.tickets.map((t) => ({
        combo: t.combo,
        modelProb: t.model_prob,
        marketOdds: t.market_odds,
        ev: t.ev,
      })),
      hasExhibition: p.has_exhibition,
      bcStatus: p.bc_status,
    }),
  );

  const evaluatedRaceIds: number[] = result.evaluated_race_ids ?? [];
  const skipped: PredictionResult["skipped"] = result.skipped ?? {};

  return { predictions, evaluatedRaceIds, skipped };
}

/**
 * Calculate bet unit for P2 strategy.
 * Rule: bankroll / unitDivisor, rounded down to ¥100, MIN ¥100, MAX betCap.
 */
export function calcTrifectaUnit(
  bankroll: number,
  betCap: number,
  unitDivisor: number,
): number {
  let unit = Math.floor(bankroll / unitDivisor / 100) * 100;
  unit = Math.max(unit, 100);
  unit = Math.min(unit, betCap);
  if (unit > bankroll) return 0;
  return unit;
}

// ---------------------------------------------------------------------------
// Poll helpers
// ---------------------------------------------------------------------------

function updatePredictionCache(
  state: RunnerState,
  result: PredictionResult,
): void {
  if (!state.predictionCache) {
    state.predictionCache = new Map();
  }
  for (const [ridStr, info] of Object.entries(result.skipped)) {
    const rid = Number(ridStr);
    state.predictionCache.set(rid, {
      skipReason: info.reason,
      top1Boat: info.top1_boat,
      top3Conc: info.top3_conc,
      gap23: info.gap23,
      gap12: info.gap12,
      stadiumId: info.stadium_id,
      withdrawnBoats: info.withdrawn_boats,
      bcStatus: info.bc_status,
      wouldBeTickets: info.would_be_tickets?.map((t) => ({
        combo: t.combo,
        modelProb: t.model_prob,
        marketOdds: t.market_odds,
        ev: t.ev,
      })),
    });
    // Count skip reason for daily summary
    if (info.reason in state.skipCounts) {
      state.skipCounts[info.reason as keyof typeof state.skipCounts]++;
    }
  }
  for (const p of result.predictions) {
    state.predictionCache.set(p.raceId, {
      top3Conc: p.top3Conc,
      gap23: p.gap23,
      gap12: p.gap12,
      tickets: p.tickets,
      hasExhibition: p.hasExhibition,
      bcStatus: p.bcStatus,
    });
  }
  persistRunnerState(state);
  logger.debug(
    `Prediction cache: ${result.predictions.length} qualifying / ${result.evaluatedRaceIds.length} evaluated`,
  );
}

/** Make bet decisions for P2 predictions. */
async function makeBetDecisions(
  slots: RaceSlot[],
  state: RunnerState,
  opts: RunnerOptions,
  bets: Map<number, BetDecision>,
): Promise<void> {
  for (const slot of slots) {
    const cached = state.predictionCache?.get(slot.raceId);
    if (!cached || "skipReason" in cached) {
      continue;
    }

    const label = `${slot.stadiumName} R${slot.raceNumber}`;
    const tickets = cached.tickets;

    // Defensive: T-1 drift in poll() already continues when survivingTickets is
    // empty, so this branch should be unreachable. Kept as a guard in case
    // makeBetDecisions is called from a different path in the future.
    if (tickets.length === 0) {
      logger.info(`[P2] SKIP: ${label} | no tickets after drift`);
      continue;
    }

    const avgEv = tickets.reduce((s, t) => s + t.ev, 0) / tickets.length;
    const ticketStr = tickets
      .map((t) => `${t.combo}(${(t.ev * 100).toFixed(0)}%)`)
      .join(", ");

    const unit = calcTrifectaUnit(
      state.bankroll,
      opts.betCap,
      opts.unitDivisor,
    );
    const totalWager = unit * tickets.length;

    if (unit > 0 && totalWager <= state.bankroll) {
      // Atomic state mutation: record bet, deduct bankroll, persist once.
      // Two separate persists would leave a window where the bet is recorded
      // but the bankroll is not yet deducted; a crash there would over-credit
      // the bankroll on the next result. Single persist closes the window.
      const decision: BetDecision = {
        raceId: slot.raceId,
        stadiumName: slot.stadiumName,
        raceNumber: slot.raceNumber,
        boatNumber: 1, // P2 always predicts boat 1 first
        prob: 0,
        odds: 0,
        ev: avgEv * 100,
        betAmount: totalWager,
        recommend: true,
        tickets: tickets.map((t) => ({
          combo: t.combo,
          modelProb: t.modelProb,
          marketOdds: t.marketOdds,
          ev: t.ev,
        })),
      };
      bets.set(slot.raceId, decision);
      state.bankroll -= totalWager;
      persistRunnerState(state);

      logger.info(
        `[P2] BET: ${label} | gap12=${(cached.gap12 * 100).toFixed(1)}% conc=${(cached.top3Conc * 100).toFixed(0)}% gap23=${(cached.gap23 * 100).toFixed(1)}% | ¥${unit.toLocaleString()}×${tickets.length}=¥${totalWager.toLocaleString()} | ${ticketStr}`,
      );

      await notifyPrediction({
        stadiumName: slot.stadiumName,
        raceNumber: slot.raceNumber,
        deadline: slot.deadline,
        top3Conc: cached.top3Conc,
        gap23: cached.gap23,
        tickets: tickets.map((t) => ({
          combo: t.combo,
          modelProb: t.modelProb,
          marketOdds: t.marketOdds,
          ev: t.ev,
        })),
        unit,
        betAmount: totalWager,
      });
    } else {
      logger.info(
        `[P2] SKIP: ${label} | bankroll insufficient (¥${state.bankroll.toLocaleString()})`,
      );
    }
  }
}

/** Use fast polling when any active race with prediction is near T-1 deadline. */
function getPollInterval(schedule: RaceSlot[]): number {
  const now = Date.now();
  for (const slot of schedule) {
    if (slot.status !== "active") continue;
    const minutesToDeadline = (slot.deadlineMs - now) / 60_000;
    if (minutesToDeadline <= FAST_POLL_WINDOW_MIN && minutesToDeadline > 0) {
      return POLL_INTERVAL_FAST_MS;
    }
  }
  return POLL_INTERVAL_MS;
}

/** Re-read deadlines from DB for not-yet-decided races. */
function refreshDeadlinesFromDb(schedule: RaceSlot[], date: string): void {
  const pending = schedule.filter(
    (s) =>
      s.status !== "decided" &&
      s.status !== "done" &&
      s.status !== "result_pending",
  );
  if (pending.length === 0) return;

  const rows = getDatabase()
    .query(
      "SELECT id, deadline FROM races WHERE race_date = ? AND deadline IS NOT NULL",
    )
    .all(date) as { id: number; deadline: string }[];
  const dbMap = new Map(rows.map((r) => [r.id, r.deadline]));

  for (const slot of pending) {
    const freshDeadline = dbMap.get(slot.raceId);
    if (freshDeadline && freshDeadline !== slot.deadline) {
      const oldDeadline = slot.deadline;
      slot.deadline = freshDeadline;
      slot.deadlineMs = new Date(`${date}T${freshDeadline}:00+09:00`).getTime();
      logger.warn(
        `Deadline updated: ${slot.stadiumName} R${slot.raceNumber} | ${oldDeadline} → ${freshDeadline}`,
      );
    }
  }
}

function hasSnapshot(raceId: number, timing: string): boolean {
  const row = getDatabase()
    .query(
      "SELECT COUNT(*) as cnt FROM race_odds_snapshots WHERE race_id = ? AND timing = ?",
    )
    .get(raceId, timing) as { cnt: number };
  return row.cnt > 0;
}

// ---------------------------------------------------------------------------
// Main poll loop
// ---------------------------------------------------------------------------

async function poll(state: RunnerState, opts: RunnerOptions): Promise<void> {
  const { schedule, bets, results } = state;
  const now = Date.now();
  const {
    activate,
    active,
    results: resultSlots,
  } = getActiveRaces(schedule, now);

  // Status log
  const counts = { waiting: 0, active: 0, decided: 0, done: 0 };
  for (const s of schedule) {
    if (s.status === "waiting") counts.waiting++;
    else if (s.status === "active") counts.active++;
    else if (s.status === "decided" || s.status === "result_pending")
      counts.decided++;
    else if (s.status === "done") counts.done++;
  }
  const statusLine = `${counts.waiting}/${counts.active}/${counts.decided}/${counts.done}`;
  if (statusLine !== state.lastStatusLine) {
    logger.info(
      `Status: ${schedule.length}R | wait:${counts.waiting} active:${counts.active} bet:${counts.decided} done:${counts.done}`,
    );
    state.lastStatusLine = statusLine;
  }

  refreshDeadlinesFromDb(schedule, state.date);

  // 1. Activate races approaching deadline
  for (const slot of activate) {
    slot.status = "active";
  }

  // 2a. Collect races needing prediction (have T-5 odds, no cache yet)
  const toPredictSlots = active.filter((s) => {
    if (state.predictionCache?.has(s.raceId)) return false;
    return hasSnapshot(s.raceId, "T-5");
  });

  if (toPredictSlots.length > 0) {
    try {
      getDatabase().exec("PRAGMA wal_checkpoint(TRUNCATE)");
      const targetRaceIds = toPredictSlots.map((s) => s.raceId);
      logger.info(
        `Running P2 prediction for ${targetRaceIds.length} race(s)...`,
      );
      const result = await runPrediction(
        state.date,
        opts,
        state.snapshotPath,
        targetRaceIds,
      );
      updatePredictionCache(state, result);

      // Log T-5 summary
      for (const slot of toPredictSlots) {
        const cached = state.predictionCache?.get(slot.raceId);
        if (!cached) continue;
        const label = `${slot.stadiumName} R${slot.raceNumber}`;
        const bcTag = cached.bcStatus ? ` | bc=${cached.bcStatus}` : "";
        if ("skipReason" in cached) {
          logger.info(
            `[T-5] ${label} | SKIP: ${formatSkipReason(cached, opts)}${bcTag}`,
          );
        } else {
          const ticketStr = cached.tickets
            .map((t) => `${t.combo}(EV ${(t.ev * 100).toFixed(0)}%)`)
            .join(", ");
          logger.info(
            `[T-5] ${label} | gap12=${(cached.gap12 * 100).toFixed(1)}% conc=${(cached.top3Conc * 100).toFixed(0)}% gap23=${(cached.gap23 * 100).toFixed(1)}% | ${ticketStr}${bcTag}`,
          );
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.error(`Prediction failed: ${msg}`);
      await notifyError("prediction", err).catch(() => {});
    }
  }

  // 2b. T-1 drift: re-evaluate per-ticket EV from T-1 snapshot odds
  for (const slot of active) {
    const minutesToDeadline = (slot.deadlineMs - now) / 60_000;
    if (minutesToDeadline > ODDS_LEAD) continue;

    const cached = state.predictionCache?.get(slot.raceId);
    if (!cached) continue;
    if ("skipReason" in cached) {
      logger.info(
        `[P2] SKIP: ${slot.stadiumName} R${slot.raceNumber} | ${formatSkipReason(cached, opts)}`,
      );
      slot.status = "decided";
      continue;
    }

    // Need T-1 snapshot for drift
    if (!hasSnapshot(slot.raceId, "T-1")) {
      continue; // Not yet available, try next poll
    }

    // P2 drift: look up T-1 odds for each ticket combo directly
    const combos = cached.tickets.map((t) => t.combo);
    const t1Odds = loadSnapshotTrifectaOdds(slot.raceId, "T-1", combos);

    const label = `${slot.stadiumName} R${slot.raceNumber}`;
    const survivingTickets: P2Ticket[] = [];

    for (const ticket of cached.tickets) {
      const newOdds = t1Odds.get(ticket.combo);
      if (!newOdds || newOdds <= 0) {
        logger.info(
          `[DRIFT] ${label} | ${ticket.combo} | no T-1 odds, dropping`,
        );
        state.t1DroppedTickets++;
        continue;
      }
      const oldEv = ticket.ev;
      const newEv = (ticket.modelProb / (1.0 / newOdds)) * 0.75 - 1;

      logger.info(
        `[DRIFT] ${label} | ${ticket.combo} | EV ${(oldEv * 100).toFixed(1)}%→${(newEv * 100).toFixed(1)}% (odds ${ticket.marketOdds}→${newOdds})`,
      );

      if (newEv >= opts.evThreshold) {
        survivingTickets.push({
          ...ticket,
          ev: newEv,
          marketOdds: newOdds,
        });
      } else {
        state.t1DroppedTickets++;
      }
    }

    cached.tickets = survivingTickets;

    if (survivingTickets.length === 0) {
      logger.info(`[P2] SKIP: ${label} | all tickets dropped after drift`);
      state.skipCounts.drift_drop++;
      persistRunnerState(state);
      slot.status = "decided";
      continue;
    }

    // Bet decision (also persists runner state on bet)
    await makeBetDecisions([slot], state, opts, bets);
    slot.status = "decided";
  }

  // 3. Check results from DB
  for (const slot of resultSlots) {
    try {
      const db = getDatabase();
      // Fetch ALL entries (finishers + DSQ'd) so we can detect refunded combos.
      // A boat with NULL finish_position is either DSQ'd mid-race (e.g. flying
      // start) or didn't start; either way, combos containing it are refunded.
      const allEntries = db
        .query(
          `SELECT boat_number, finish_position FROM race_entries
           WHERE race_id = ? ORDER BY boat_number`,
        )
        .all(slot.raceId) as {
        boat_number: number;
        finish_position: number | null;
      }[];

      const finishers = allEntries
        .filter((e) => e.finish_position != null)
        .sort(
          (a, b) =>
            (a.finish_position as number) - (b.finish_position as number),
        );

      if (finishers.length === 0) {
        slot.status = "result_pending";
        continue;
      }

      const bet = bets.get(slot.raceId);
      if (!bet) {
        slot.status = "done";
        continue;
      }

      const refundedBoats = new Set<number>(
        allEntries
          .filter((e) => e.finish_position == null)
          .map((e) => e.boat_number),
      );
      const actual1st = finishers[0]?.boat_number;
      const actual2nd = finishers[1]?.boat_number;
      const actual3rd = finishers[2]?.boat_number;

      // Use tickets from BetDecision (persisted at BET time) so refund and
      // payout logic survives a runner restart that wipes the prediction cache.
      const tickets = bet.tickets;
      const unitStake = tickets.length > 0 ? bet.betAmount / tickets.length : 0;
      const refundedTicketCount = tickets.filter((t) =>
        ticketContainsRefundedBoat(t.combo, refundedBoats),
      ).length;
      const refundedAmount = refundedTicketCount * unitStake;

      const hitCombo =
        actual1st && actual2nd && actual3rd
          ? `${actual1st}-${actual2nd}-${actual3rd}`
          : null;
      // A refunded ticket cannot win even if its combo coincidentally matches
      // hitCombo, because hitCombo is built from non-DSQ finishers and would
      // not include a refunded boat anyway. The check is defensive.
      const won =
        hitCombo != null &&
        tickets.some(
          (t) =>
            t.combo === hitCombo &&
            !ticketContainsRefundedBoat(t.combo, refundedBoats),
        );

      // Official payout (確定オッズ × 100円) from race_payouts
      let officialPayoutPer100 = 0;
      if (hitCombo) {
        const payoutRow = db
          .query(
            `SELECT payout FROM race_payouts
             WHERE race_id = ? AND bet_type = '3連単' AND combination = ?`,
          )
          .get(slot.raceId, hitCombo) as { payout: number } | null;
        officialPayoutPer100 = payoutRow?.payout ?? 0;
      }

      let payout = 0;
      if (won && officialPayoutPer100 > 0) {
        payout = Math.round((unitStake / 100) * officialPayoutPer100);
      }

      // Bankroll already had bet.betAmount deducted at BET time. At result we
      // credit (payout + refundedAmount). Refunded tickets get their stake
      // back without affecting P/L.
      state.bankroll += payout + refundedAmount;
      results.set(slot.raceId, { won, payout });
      persistRunnerState(state);

      const resultStr = hitCombo ?? "N/A";
      const pl = payout + refundedAmount - bet.betAmount;
      const plStr =
        pl >= 0
          ? `+¥${pl.toLocaleString()}`
          : `-¥${Math.abs(pl).toLocaleString()}`;
      const combosLog = tickets.map((t) => t.combo).join(",");
      const refundTag =
        refundedTicketCount > 0
          ? ` | 返還 ${refundedTicketCount}/${tickets.length}券 ¥${refundedAmount.toLocaleString()}`
          : "";
      const tag = resultTag(won, refundedTicketCount, tickets.length);
      logger.info(
        `[P2] ${tag}: ${slot.stadiumName} R${slot.raceNumber} | 買目 ${combosLog} | 結果 ${resultStr} | 確定 ¥${officialPayoutPer100 || "-"}${refundTag} | ${plStr} (残¥${state.bankroll.toLocaleString()})`,
      );

      await notifyResult({
        stadiumName: slot.stadiumName,
        raceNumber: slot.raceNumber,
        won,
        betAmount: bet.betAmount,
        payout,
        bankroll: state.bankroll,
        tickets: tickets.map((t) => ({
          combo: t.combo,
          marketOdds: t.marketOdds,
          ev: t.ev,
        })),
        resultCombo: hitCombo,
        officialPayoutPer100,
      });

      slot.status = "done";
    } catch (err) {
      await notifyError(`result ${slot.stadiumName} R${slot.raceNumber}`, err);
      slot.status = "done";
    }
  }
}

// ---------------------------------------------------------------------------
// Day setup
// ---------------------------------------------------------------------------

const SCHEDULE_RETRY_INTERVAL_MS = 60_000;
const SCHEDULE_MAX_RETRIES = 30;

const stadiumNames = new Map(
  Object.entries(STADIUMS).map(([code, name]) => [
    Number.parseInt(code, 10),
    name,
  ]),
);

/** Snapshot in-memory daily state + bankroll into bankrollState and persist.
 * Called after every state mutation so daily Slack summary can recover from
 * mid-day runner restarts. */
function persistRunnerState(state: RunnerState): void {
  state.bankrollState.bankroll = state.bankroll;
  state.bankrollState.lastUpdate = new Date().toISOString();
  state.bankrollState.today = {
    date: state.date,
    bets: Array.from(state.bets.entries()).map(([raceId, decision]) => ({
      raceId,
      decision: { ...decision },
    })),
    results: Array.from(state.results.entries()).map(([raceId, r]) => ({
      raceId,
      won: r.won,
      payout: r.payout,
    })),
    skipCounts: { ...state.skipCounts },
    t1DroppedTickets: state.t1DroppedTickets,
  };
  saveBankrollState(state.bankrollState);
}

async function setupDay(
  opts: RunnerOptions,
  bankrollState: BankrollState,
): Promise<RunnerState | null> {
  const date = todayJST();

  logger.info(
    `Starting P2 runner v${config.version} for ${date} (${opts.dryRun ? "DRY RUN" : "LIVE"})`,
  );
  logger.info(
    `Strategy: EV>=${(opts.evThreshold * 100).toFixed(0)}% | unit=BR/${opts.unitDivisor} cap=¥${opts.betCap.toLocaleString()}`,
  );

  const db = getDatabase();
  let races: {
    id: number;
    stadium_id: number;
    race_number: number;
    deadline: string | null;
  }[] = [];

  for (let attempt = 0; attempt <= SCHEDULE_MAX_RETRIES; attempt++) {
    races = db
      .query(
        `SELECT id, stadium_id, race_number, deadline FROM races
         WHERE race_date = ? ORDER BY deadline, stadium_id, race_number`,
      )
      .all(date) as typeof races;
    if (races.length > 0) break;
    if (attempt < SCHEDULE_MAX_RETRIES) {
      logger.warn(
        `No races in DB yet (attempt ${attempt + 1}/${SCHEDULE_MAX_RETRIES + 1}), waiting for scrape-daemon...`,
      );
      await Bun.sleep(SCHEDULE_RETRY_INTERVAL_MS);
    }
  }

  if (races.length === 0) {
    logger.warn("No races found in DB after retries");
    return null;
  }

  const schedule = buildSchedule(races, stadiumNames, date);
  if (schedule.length > 0) {
    const venues = new Set(schedule.map((s) => s.stadiumName));
    const first = schedule[0];
    const last = schedule[schedule.length - 1];
    logger.info(
      `Schedule: ${schedule.length}R / ${venues.size} venues | ${first.deadline} ~ ${last.deadline}`,
    );
  } else {
    logger.info("Schedule: 0 races");
  }

  const snapshotPath = await buildStatsSnapshot(date);

  logger.info(
    `Day start bankroll: ¥${bankrollState.bankroll.toLocaleString()} (all-time initial ¥${bankrollState.allTimeInitial.toLocaleString()}, since ${bankrollState.startedAt.slice(0, 10)})`,
  );

  const state: RunnerState = {
    schedule,
    bets: new Map(),
    results: new Map(),
    predictionCache: null,
    bankroll: bankrollState.bankroll,
    bankrollState,
    date,
    lastStatusLine: "",
    snapshotPath,
    skipCounts: {
      not_b1_top: 0,
      gap12_low: 0,
      top3_conc_low: 0,
      gap23_low: 0,
      no_ev_tickets: 0,
      drift_drop: 0,
      stadium_excluded: 0,
      withdrawal: 0,
    },
    t1DroppedTickets: 0,
  };

  // Restore today's snapshot if a same-day persisted state exists.
  // This recovers bets/results/skipCounts after a mid-day runner restart so
  // the daily Slack summary reports the full day's activity.
  const snapshot = bankrollState.today;
  if (snapshot && snapshot.date === date) {
    for (const { raceId, decision } of snapshot.bets) {
      state.bets.set(raceId, decision);
    }
    for (const { raceId, won, payout } of snapshot.results) {
      state.results.set(raceId, { won, payout });
    }
    Object.assign(state.skipCounts, snapshot.skipCounts);
    state.t1DroppedTickets = snapshot.t1DroppedTickets;
    logger.info(
      `Restored daily snapshot: ${state.bets.size} bet(s), ${state.results.size} result(s), ${state.t1DroppedTickets} drift drop(s)`,
    );
  } else if (snapshot && snapshot.date !== date) {
    // Day has rolled over — drop yesterday's snapshot, keep bankroll only.
    bankrollState.today = undefined;
    saveBankrollState(bankrollState);
    logger.info(
      `Discarded stale snapshot from ${snapshot.date} (today is ${date})`,
    );
  }

  await notifyStartup({
    version: config.version,
    date,
    venues: new Set(races.map((r) => r.stadium_id)).size,
    races: schedule.length,
    dryRun: opts.dryRun,
    evThreshold: opts.evThreshold,
  });

  return state;
}

// ---------------------------------------------------------------------------
// Main daemon
// ---------------------------------------------------------------------------

export async function runDaemon(opts: RunnerOptions): Promise<void> {
  setSlackWebhook(opts.slackWebhookUrl);
  enableFileLog(resolve(config.projectRoot, "logs"));
  initializeDatabase();

  const bankrollState = loadBankrollState(opts.bankroll);
  await waitUntilJST7am("today");
  const state = await setupDay(opts, bankrollState);
  if (!state) {
    closeDatabase();
    return;
  }

  function shutdown(): void {
    logger.info("Shutting down...");
    notifyShutdown()
      .catch(() => {})
      .finally(() => {
        closeDatabase();
        process.exit(0);
      });
  }

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  while (true) {
    await poll(state, opts);

    let dayDone = false;
    while (!dayDone) {
      const interval = getPollInterval(state.schedule);
      await Bun.sleep(interval);
      try {
        await poll(state, opts);
        if (allDone(state.schedule)) {
          logger.info("All races done for today");
          dayDone = true;
        }
      } catch (err) {
        await notifyError("poll", err).catch(console.error);
      }
    }

    // Daily summary
    {
      let totalWagered = 0;
      let totalPayout = 0;
      let totalTickets = 0;
      let wins = 0;
      for (const [raceId, bet] of state.bets) {
        totalWagered += bet.betAmount;
        // Recover ticket count from cache (bet doesn't store it directly)
        const cached = state.predictionCache?.get(raceId);
        if (cached && !("skipReason" in cached)) {
          totalTickets += cached.tickets.length;
        }
        const r = state.results.get(raceId);
        if (r) {
          totalPayout += r.payout;
          if (r.won) wins++;
        }
      }
      await notifyDailySummary({
        date: state.date,
        totalRaces: state.schedule.length,
        totalBets: state.bets.size,
        totalTickets,
        wins,
        totalWagered,
        totalPayout,
        bankroll: state.bankroll,
        allTimeInitial: state.bankrollState.allTimeInitial,
        startedAt: state.bankrollState.startedAt,
        skipCounts: state.skipCounts,
        t1DroppedTickets: state.t1DroppedTickets,
      });
    }

    await waitUntilJST7am("tomorrow");

    logger.info("New day starting...");
    closeDatabase();
    initializeDatabase();

    // bankrollState persists across days and restarts
    const newState = await setupDay(opts, state.bankrollState);
    if (!newState) continue;

    Object.assign(state, newState);
  }
}
