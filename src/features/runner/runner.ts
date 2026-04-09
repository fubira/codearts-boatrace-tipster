/** Daemon that automates scrape → predict → notify cycle. */

import { existsSync, mkdirSync, readdirSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import {
  type OddsTiming,
  closeDatabase,
  getDatabase,
  initializeDatabase,
  loadSnapshotWinProbs,
  saveBeforeInfo,
  saveOdds,
  saveOddsSnapshot,
  saveRaceResults,
  saveRaces,
} from "@/features/database";
import {
  disableCacheRead,
  enableCache,
  enableCacheRead,
} from "@/features/scraper/cache-manager";
import { fetchPage } from "@/features/scraper/http-client";
import { getScraper } from "@/features/scraper/registry";
import { fetchAndSaveBoatcast } from "@/features/scraper/sources/boatcast/fetcher";
import {
  type RaceParams,
  STADIUMS,
  beforeInfoUrl,
  odds3TUrl,
  raceResultUrl,
} from "@/features/scraper/sources/boatrace/constants";
import { discoverDateSchedule } from "@/features/scraper/sources/boatrace/discovery";
import { parseOdds3T } from "@/features/scraper/sources/boatrace/odds-parsers";
import {
  parseBeforeInfo,
  parseRaceResult,
} from "@/features/scraper/sources/boatrace/parsers";
import type { PurchaseExecutor } from "@/features/teleboat";
import { config } from "@/shared/config";
import { enableFileLog, logger } from "@/shared/logger";
import { pythonCommand } from "@/shared/python";
import {
  type BetDecision,
  type RaceSlot,
  allDone,
  buildSchedule,
  getActionableRaces,
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

const POLL_INTERVAL_MS = 30_000;

export interface RunnerOptions {
  dryRun: boolean;
  evThreshold: number;
  betCap: number;
  bankroll: number;
  slackWebhookUrl?: string;
  purchaseExecutor?: PurchaseExecutor | null;
}

/** Trifecta prediction: X-allflow (20-ticket fixed 1st, all 2-3) */
interface TrifectaPrediction {
  winnerPick: number;
  b1Prob: number;
  winnerProb: number;
  ev: number;
  rankUsed: number;
  tickets: string[];
  hasExhibition: boolean;
}

/** Prediction evaluated but skipped (b1_win, ev_low, no_odds, etc.) */
interface SkippedPrediction {
  skipReason: string;
  b1Prob: number;
  winnerPick?: number;
  ev?: number;
}

type CachedPrediction = TrifectaPrediction | SkippedPrediction;
type PredictionCache = Map<number, CachedPrediction>;

function formatSkipLabel(skip: SkippedPrediction): string {
  const b1Pct = (skip.b1Prob * 100).toFixed(0);
  const pickTag = skip.winnerPick != null ? ` ${skip.winnerPick}号艇1着` : "";
  const evTag = skip.ev != null ? ` EV=${(skip.ev * 100).toFixed(1)}%` : "";
  return `${skip.skipReason}${pickTag} b1=${b1Pct}%${evTag}`;
}

interface RunnerState {
  schedule: RaceSlot[];
  bets: Map<number, BetDecision>; // raceId → decision
  results: Map<number, { won: boolean; payout: number }>;
  predictionCache: PredictionCache | null; // null = not yet loaded
  oddsTimings: Map<number, Set<OddsTiming>>; // raceId → collected timing labels
  bankroll: number;
  date: string;
  lastStatusLine: string;
  snapshotPath?: string; // stats snapshot for fast inference
}

function todayJST(): string {
  return new Date()
    .toLocaleDateString("sv-SE", { timeZone: "Asia/Tokyo" })
    .replace(/\//g, "-");
}

function todayYYYYMMDD(): string {
  return todayJST().replace(/-/g, "");
}

function padStadiumCode(id: number): string {
  return String(id).padStart(2, "0");
}

// ---------------------------------------------------------------------------
// Odds drift extrapolation coefficients (fitted on T-3 + T-1 → final data)
// final_mp = DRIFT_COEF_T3 * mp_t3 + DRIFT_COEF_T1 * mp_t1 + DRIFT_INTERCEPT
// ---------------------------------------------------------------------------
const DRIFT_COEF_T3 = -0.857;
const DRIFT_COEF_T1 = 1.891;
const DRIFT_INTERCEPT = -0.006;

/**
 * Re-evaluate EV using T-3+T-1 odds extrapolation for races with cached predictions.
 * Updates the prediction cache in-place with extrapolated EV.
 */
function reEvaluateWithExtrapolation(
  slots: RaceSlot[],
  state: RunnerState,
): void {
  if (!state.predictionCache) return;

  let updated = 0;
  for (const slot of slots) {
    const cached = state.predictionCache.get(slot.raceId);
    if (!cached || "skipReason" in cached) continue;

    const mpT3 = loadSnapshotWinProbs(slot.raceId, "T-3");
    const mpT1 = loadSnapshotWinProbs(slot.raceId, "T-1");

    if (mpT3.size === 0 || mpT1.size === 0) {
      // T-3/T-1 missing — replace with SkippedPrediction to prevent bet
      const label = `${slot.stadiumName} R${slot.raceNumber}`;
      logger.warn(`[DRIFT] ${label} | T-3/T-1 snapshot missing, forcing skip`);
      state.predictionCache.set(slot.raceId, {
        skipReason: "no_drift_data",
        b1Prob: cached.b1Prob,
        winnerPick: cached.winnerPick,
        ev: cached.ev,
      });
      continue;
    }

    const boat = cached.winnerPick;
    const t3 = mpT3.get(boat);
    const t1 = mpT1.get(boat);
    if (t3 === undefined || t1 === undefined) continue;

    const extrapolatedMp =
      DRIFT_COEF_T3 * t3 + DRIFT_COEF_T1 * t1 + DRIFT_INTERCEPT;
    if (extrapolatedMp <= 0) continue;

    const oldEv = cached.ev;
    const newEv = (cached.winnerProb / extrapolatedMp) * 0.75 - 1;

    cached.ev = newEv;
    updated++;

    const label = `${slot.stadiumName} R${slot.raceNumber}`;
    const oldPct = (oldEv * 100).toFixed(1);
    const newPct = (newEv * 100).toFixed(1);
    logger.info(
      `[DRIFT] ${label} | ${boat}号艇 | T-5 EV=${oldPct}% → extrap EV=${newPct}% (mp: T3=${(t3 * 100).toFixed(1)}% T1=${(t1 * 100).toFixed(1)}% → ${(extrapolatedMp * 100).toFixed(1)}%)`,
    );
  }

  if (updated > 0) {
    logger.info(`Drift extrapolation: ${updated}/${slots.length} updated`);
  }
}

// ---------------------------------------------------------------------------
// Data fetching helpers
// ---------------------------------------------------------------------------

function scrapeBeforeInfoForRace(slot: RaceSlot, date: string): boolean {
  const params: RaceParams = {
    raceNumber: slot.raceNumber,
    stadiumCode: padStadiumCode(slot.stadiumId),
    date: date.replace(/-/g, ""),
  };

  // Always skip cache — exhibition data changes up to race time
  const page = fetchPage(beforeInfoUrl(params), { skipCache: true });
  if (!page) return false;

  const context = { params, raceDate: date };
  const data = parseBeforeInfo(page.html, context);
  if (!data) return false;

  saveBeforeInfo([data]);
  return true;
}

function scrapeResultForRace(
  slot: RaceSlot,
  date: string,
): {
  entries: { boatNumber: number; finishPosition?: number }[];
  payouts?: { betType: string; combination: string; payout: number }[];
} | null {
  const params: RaceParams = {
    raceNumber: slot.raceNumber,
    stadiumCode: padStadiumCode(slot.stadiumId),
    date: date.replace(/-/g, ""),
  };

  // Always skip cache — result page only available after race
  const page = fetchPage(raceResultUrl(params), { skipCache: true });
  if (!page) return null;

  const context = { params, raceDate: date };
  const data = parseRaceResult(page.html, context);
  if (!data) return null;

  saveRaceResults([data]);

  return data;
}

// ---------------------------------------------------------------------------
// Stats snapshot
// ---------------------------------------------------------------------------

const SNAPSHOTS_DIR = resolve(config.projectRoot, "data/stats-snapshots");
const SNAPSHOT_RETENTION_DAYS = 30;

async function buildStatsSnapshot(date: string): Promise<string | undefined> {
  // through_date = yesterday (exclude today for leak-safe stats)
  const yesterday = new Date(date);
  yesterday.setDate(yesterday.getDate() - 1);
  const throughDate = yesterday.toISOString().slice(0, 10);

  const snapshotPath = resolve(SNAPSHOTS_DIR, `${throughDate}.db`);

  // Skip if already exists (e.g., restart on same day)
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

  try {
    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    const exitCode = await proc.exited;
    if (stdout) logger.debug(stdout.trim());
    if (stderr) logger.debug(stderr.trim());
    if (exitCode !== 0) {
      logger.error(`Snapshot build failed (exit ${exitCode}): ${stderr}`);
      return undefined;
    }
    logger.info(`Snapshot built: ${throughDate}.db`);
    rotateSnapshots();
    return snapshotPath;
  } catch (err) {
    proc.kill(9);
    const msg = err instanceof Error ? err.message : String(err);
    logger.error(`Snapshot build failed: ${msg}`);
    return undefined;
  }
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
// Prediction
// ---------------------------------------------------------------------------

interface SkipInfo {
  b1_prob: number;
  ev?: number;
  pick?: number;
  reason: string;
}

interface PredictionResult {
  predictions: {
    raceId: number;
    winnerPick: number;
    b1Prob: number;
    winnerProb: number;
    ev: number;
    rankUsed: number;
    tickets: string[];
    hasExhibition: boolean;
  }[];
  evaluatedRaceIds: number[];
  skipped: Record<number, SkipInfo>;
}

async function runPrediction(
  date: string,
  opts: RunnerOptions,
  snapshotPath?: string,
  raceIds?: number[],
): Promise<PredictionResult> {
  const modelDir = resolve(config.projectRoot, "ml/models/trifecta_v1");
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
  const { cmd, cwd } = pythonCommand("scripts.predict_trifecta", args);
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
      `Trifecta stats: ${s.total} races, ${s.b1_pass} upset, ${s.has_odds} with odds, ${s.predicted} predicted`,
    );
  }

  const predictions = result.predictions.map(
    (p: {
      race_id: number;
      winner_pick: number;
      b1_prob: number;
      winner_prob: number;
      ev: number;
      rank_used?: number;
      tickets: string[];
      has_exhibition: boolean;
    }) => ({
      raceId: p.race_id,
      winnerPick: p.winner_pick,
      b1Prob: p.b1_prob,
      winnerProb: p.winner_prob,
      ev: p.ev,
      rankUsed: p.rank_used ?? 1,
      tickets: p.tickets,
      hasExhibition: p.has_exhibition,
    }),
  );

  const evaluatedRaceIds: number[] = result.evaluated_race_ids ?? [];
  const skipped: Record<
    number,
    { b1_prob: number; ev?: number; reason: string }
  > = result.skipped ?? {};

  return { predictions, evaluatedRaceIds, skipped };
}

/**
 * Calculate bet unit for trifecta strategy.
 * Rule: bankroll / 800, rounded down to ¥100, MIN ¥100, MAX betCap.
 */
export function calcTrifectaUnit(bankroll: number, betCap: number): number {
  let unit = Math.floor(bankroll / 800 / 100) * 100;
  unit = Math.max(unit, 100);
  unit = Math.min(unit, betCap);
  if (unit > bankroll) return 0;
  return unit;
}

// ---------------------------------------------------------------------------
// Poll cycle
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Poll helpers (extracted from poll() for readability)
// ---------------------------------------------------------------------------

/**
 * Fetch trifecta odds and save as snapshot.
 * Pre-race timings (T-5/T-3/T-1) save to race_odds_snapshots ONLY.
 * "final" saves to both race_odds (confirmed) and race_odds_snapshots.
 */
function fetchTrifectaOdds(
  slots: RaceSlot[],
  date: string,
  timing: OddsTiming,
  oddsTimings: Map<number, Set<OddsTiming>>,
): number {
  let fetched = 0;
  const isConfirmed = timing === "final";

  for (const slot of slots) {
    // Skip if this timing already collected for this race
    if (oddsTimings.get(slot.raceId)?.has(timing)) continue;

    const stadiumCode = padStadiumCode(slot.stadiumId);
    const params: RaceParams = {
      raceNumber: slot.raceNumber,
      stadiumCode,
      date: date.replace(/-/g, ""),
    };
    try {
      const page = fetchPage(odds3TUrl(params), { skipCache: true });
      if (page) {
        const entries = parseOdds3T(page.html);
        if (entries.length > 0) {
          const oddsEntries = entries.map((e) => ({
            betType: e.betType,
            combination: e.combination,
            odds: e.odds,
          }));

          // Confirmed odds → race_odds table (the source of truth)
          if (isConfirmed) {
            saveOdds([
              {
                stadiumId: slot.stadiumId,
                raceDate: date,
                raceNumber: slot.raceNumber,
                entries: oddsEntries,
              },
            ]);
          }

          // All timings → snapshots (for drift analysis)
          saveOddsSnapshot(slot.raceId, timing, oddsEntries);
          if (!oddsTimings.has(slot.raceId)) {
            oddsTimings.set(slot.raceId, new Set());
          }
          oddsTimings.get(slot.raceId)?.add(timing);

          fetched++;
        }
      }
    } catch {
      // Odds may not be available yet for early races
    }
  }
  return fetched;
}

/** Determine which races need prediction based on cache state. */
function getRacesToPredict(
  slots: RaceSlot[],
  cache: PredictionCache | null,
  oddsFetched: number,
): RaceSlot[] {
  const uncached = slots.filter((s) => !cache?.has(s.raceId));
  const exhibitionUpdated = slots.filter((s) => {
    if (!cache?.has(s.raceId)) return false;
    const cached = cache.get(s.raceId);
    if (!cached) return false;
    if ("skipReason" in cached) return true;
    return !cached.hasExhibition;
  });
  const oddsUpdated =
    oddsFetched > 0 ? slots.filter((s) => cache?.has(s.raceId)) : [];
  return [
    ...new Map(
      [...uncached, ...exhibitionUpdated, ...oddsUpdated].map((s) => [
        s.raceId,
        s,
      ]),
    ).values(),
  ];
}

/** Update prediction cache from ML result. */
function updatePredictionCache(
  state: RunnerState,
  result: PredictionResult,
): void {
  if (!state.predictionCache) {
    state.predictionCache = new Map();
  }
  for (const [ridStr, info] of Object.entries(result.skipped)) {
    const rid = Number(ridStr);
    const skipped: SkippedPrediction = {
      skipReason: info.reason,
      b1Prob: info.b1_prob,
      winnerPick: info.pick,
      ev: info.ev,
    };
    state.predictionCache.set(rid, skipped);
  }
  for (const p of result.predictions) {
    state.predictionCache.set(p.raceId, {
      winnerPick: p.winnerPick,
      b1Prob: p.b1Prob,
      winnerProb: p.winnerProb,
      ev: p.ev,
      rankUsed: p.rankUsed,
      tickets: p.tickets,
      hasExhibition: p.hasExhibition,
    });
  }
  logger.debug(
    `Prediction cache: ${result.predictions.length} qualifying / ${result.evaluatedRaceIds.length} evaluated`,
  );
}

/** Predict and make bet decisions for actionable races. Odds saved as T-5 snapshot. */
async function pollPredict(
  state: RunnerState,
  slots: RaceSlot[],
  opts: RunnerOptions,
  bets: Map<number, BetDecision>,
): Promise<void> {
  // Re-scrape before-info and BOATCAST if exhibition data is missing
  const db = getDatabase();
  let rescraped = 0;
  for (const slot of slots) {
    const counts = db
      .query(
        `SELECT
           SUM(CASE WHEN exhibition_time IS NOT NULL THEN 1 ELSE 0 END) as exh,
           SUM(CASE WHEN bc_lap_time IS NOT NULL THEN 1 ELSE 0 END) as bc
         FROM race_entries WHERE race_id = ?`,
      )
      .get(slot.raceId) as { exh: number; bc: number };
    if (counts.exh === 0) {
      try {
        scrapeBeforeInfoForRace(slot, state.date);
        rescraped++;
      } catch {
        // Prediction will proceed without exhibition data
      }
    }
    if (counts.bc === 0) {
      try {
        await fetchAndSaveBoatcast(slot.stadiumId, state.date, slot.raceNumber);
      } catch {
        // Non-critical
      }
    }
  }
  if (rescraped > 0) {
    logger.info(`Re-scraped exhibition data for ${rescraped} race(s)`);
  }

  const oddsFetched = fetchTrifectaOdds(
    slots,
    state.date,
    "T-5",
    state.oddsTimings,
  );
  logger.info(`Trifecta odds (T-5): ${oddsFetched}/${slots.length} fetched`);

  const racesToPredict = getRacesToPredict(
    slots,
    state.predictionCache,
    oddsFetched,
  );

  try {
    if (racesToPredict.length > 0) {
      getDatabase().exec("PRAGMA wal_checkpoint(TRUNCATE)");

      const targetRaceIds = racesToPredict.map((s) => s.raceId);
      logger.info(
        `Running trifecta prediction for ${targetRaceIds.length} race(s)...`,
      );
      const result = await runPrediction(
        state.date,
        opts,
        state.snapshotPath,
        targetRaceIds,
      );
      updatePredictionCache(state, result);
    }

    // Log T-5 EV summary (bet decision deferred to T-1 re-evaluation)
    for (const slot of slots) {
      const cached = state.predictionCache?.get(slot.raceId);
      if (!cached) {
        logger.info(
          `[T-5] ${slot.stadiumName} R${slot.raceNumber} | no prediction returned`,
        );
        slot.status = "predicted";
        continue;
      }
      const label = `${slot.stadiumName} R${slot.raceNumber}`;
      if ("skipReason" in cached) {
        logger.info(`[T-5] ${label} | ${formatSkipLabel(cached)}`);
      } else {
        const evPct = (cached.ev * 100).toFixed(1);
        const rankTag = cached.rankUsed === 2 ? "(r2)" : "";
        logger.info(
          `[T-5] ${label} | ${cached.winnerPick}号艇1着${rankTag} | b1=${(cached.b1Prob * 100).toFixed(0)}% EV=${evPct}%`,
        );
      }
      slot.status = "predicted";
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error(`Prediction failed: ${msg}`);
    await notifyError("prediction", err).catch(() => {});
    for (const slot of slots) {
      slot.status = "predicted";
    }
  }
}

/** Make bet decisions for each slot based on cached predictions. */
async function makeBetDecisions(
  slots: RaceSlot[],
  state: RunnerState,
  opts: RunnerOptions,
  bets: Map<number, BetDecision>,
): Promise<void> {
  const evThresholdPct = opts.evThreshold * 100;
  for (const slot of slots) {
    const cached = state.predictionCache?.get(slot.raceId);
    if (!cached || "skipReason" in cached) {
      const reason = cached ? formatSkipLabel(cached) : "unknown";
      logger.info(
        `[TRI] SKIP: ${slot.stadiumName} R${slot.raceNumber} | ${reason}`,
      );
      slot.status = "predicted";
      continue;
    }

    if (!cached.hasExhibition) {
      logger.info(
        `[TRI] WAIT: ${slot.stadiumName} R${slot.raceNumber} | exhibition data missing, retrying next poll`,
      );
      slot.status = "before_info";
      continue;
    }

    const label = `${slot.stadiumName} R${slot.raceNumber}`;
    const evFrac = cached.ev;
    const evPct = evFrac * 100;
    const isBet = evFrac >= opts.evThreshold;

    let thresholdTag = "";
    if (!isBet) {
      const wouldPass = [10, 20, 30].filter((t) => evPct >= t);
      thresholdTag =
        wouldPass.length > 0
          ? ` (would buy ≥${wouldPass[wouldPass.length - 1]}%)`
          : "";
    }

    const rankTag = cached.rankUsed === 2 ? "(r2)" : "";
    const base = `${label} | ${cached.winnerPick}号艇1着${rankTag} | b1=${(cached.b1Prob * 100).toFixed(0)}% EV=${evPct.toFixed(1)}% | ${cached.tickets.length}pt`;

    if (isBet) {
      const unit = calcTrifectaUnit(state.bankroll, opts.betCap);
      const totalWager = unit * cached.tickets.length;

      if (unit > 0 && totalWager <= state.bankroll) {
        const decision: BetDecision = {
          raceId: slot.raceId,
          stadiumName: slot.stadiumName,
          raceNumber: slot.raceNumber,
          boatNumber: cached.winnerPick,
          prob: cached.winnerProb,
          odds: 0,
          ev: evPct,
          betAmount: totalWager,
          recommend: true,
        };
        bets.set(slot.raceId, decision);

        const ticketStr = cached.tickets.slice(0, 3).join(", ");
        const moreStr =
          cached.tickets.length > 3 ? ` +${cached.tickets.length - 3}` : "";
        logger.info(
          `[TRI] BET: ${base} | ¥${unit}×${cached.tickets.length}=¥${totalWager.toLocaleString()} | ${ticketStr}${moreStr}`,
        );

        await notifyPrediction({
          stadiumName: slot.stadiumName,
          raceNumber: slot.raceNumber,
          deadline: slot.deadline,
          prob: cached.winnerProb,
          odds: 0,
          ev: evPct,
          betAmount: totalWager,
        });

        // TODO: teleboat purchase for trifecta (Phase 2)

        state.bankroll -= totalWager;
      } else {
        logger.info(`[TRI] SKIP: ${base} | bankroll insufficient`);
      }
    } else {
      logger.info(
        `[TRI] ---: ${base} | <${evThresholdPct.toFixed(0)}%${thresholdTag}`,
      );
    }

    slot.status = "predicted";
  }
}

// ---------------------------------------------------------------------------
// Main poll loop
// ---------------------------------------------------------------------------

async function poll(state: RunnerState, opts: RunnerOptions): Promise<void> {
  const { schedule, bets, results } = state;
  const now = Date.now();
  const actionable = getActionableRaces(schedule, now);

  // Status log
  const counts = {
    waiting: 0,
    before_info: 0,
    predicted: 0,
    decided: 0,
    result_pending: 0,
    done: 0,
  };
  for (const s of schedule) counts[s.status]++;
  const active =
    actionable.beforeInfo.length +
    actionable.predict.length +
    actionable.oddsT3.length +
    actionable.odds.length +
    actionable.results.length;
  const statusLine = `${counts.waiting}/${counts.before_info}/${counts.predicted + counts.decided + counts.result_pending}/${counts.done}`;
  if (statusLine !== state.lastStatusLine) {
    logger.info(
      `Status: ${schedule.length}R | wait:${counts.waiting} exh:${counts.before_info} pred:${counts.predicted} bet:${counts.decided + counts.result_pending} done:${counts.done} | action:${active}`,
    );
    state.lastStatusLine = statusLine;
  }

  // 1. Scrape before-info for races approaching deadline
  if (actionable.beforeInfo.length > 0) {
    let scraped = 0;
    let bcFetched = 0;
    for (const slot of actionable.beforeInfo) {
      try {
        const ok = scrapeBeforeInfoForRace(slot, state.date);
        if (ok) {
          slot.status = "before_info";
          scraped++;
        }
      } catch (err) {
        await notifyError(
          `before-info ${slot.stadiumName} R${slot.raceNumber}`,
          err,
        );
      }
      // Fetch BOATCAST exhibition data (oriten + stt)
      try {
        const bc = await fetchAndSaveBoatcast(
          slot.stadiumId,
          state.date,
          slot.raceNumber,
        );
        if (bc) bcFetched++;
      } catch {
        // Non-critical: prediction proceeds without BOATCAST data
      }
    }
    logger.info(
      `Before-info: ${scraped}/${actionable.beforeInfo.length} scraped, BOATCAST: ${bcFetched}`,
    );
  }

  // 2. Predict for races near deadline (trifecta strategy)
  if (actionable.predict.length > 0) {
    await pollPredict(state, actionable.predict, opts, bets);
  }

  // 3a. Fetch T-3 odds snapshot (no status change)
  if (actionable.oddsT3.length > 0) {
    const t3Fetched = fetchTrifectaOdds(
      actionable.oddsT3,
      state.date,
      "T-3",
      state.oddsTimings,
    );
    if (t3Fetched > 0) {
      logger.info(
        `Odds snapshot (T-3): ${t3Fetched}/${actionable.oddsT3.length} fetched`,
      );
    }
  }

  // 3b. Fetch T-1 odds snapshot, re-evaluate EV with drift extrapolation, and make bet decisions
  if (actionable.odds.length > 0) {
    const t1Fetched = fetchTrifectaOdds(
      actionable.odds,
      state.date,
      "T-1",
      state.oddsTimings,
    );
    if (t1Fetched > 0) {
      logger.info(
        `Odds snapshot (T-1): ${t1Fetched}/${actionable.odds.length} fetched`,
      );
    }

    // Re-evaluate EV using T-3+T-1 extrapolation
    reEvaluateWithExtrapolation(actionable.odds, state);

    // Now make bet decisions with extrapolated EV
    await makeBetDecisions(actionable.odds, state, opts, bets);
  }
  for (const slot of actionable.odds) {
    slot.status = "decided";
  }

  // 4. Check results for finished races (always scrape, even without bets)
  // Fetch final (confirmed) odds for all results races
  if (actionable.results.length > 0) {
    const finalFetched = fetchTrifectaOdds(
      actionable.results,
      state.date,
      "final",
      state.oddsTimings,
    );
    if (finalFetched > 0) {
      logger.info(
        `Odds snapshot (final): ${finalFetched}/${actionable.results.length} fetched`,
      );
    }
  }

  for (const slot of actionable.results) {
    try {
      const raceResult = scrapeResultForRace(slot, state.date);

      if (raceResult === null) {
        slot.status = "result_pending";
        continue;
      }

      const bet = bets.get(slot.raceId);
      if (!bet) {
        slot.status = "done";
        continue;
      }

      // Find actual finishing order
      const entries = raceResult.entries
        .filter((e) => e.finishPosition != null)
        .sort((a, b) => (a.finishPosition ?? 99) - (b.finishPosition ?? 99));

      const actual1st = entries[0]?.boatNumber;
      const actual2nd = entries[1]?.boatNumber;
      const actual3rd = entries[2]?.boatNumber;

      // Check if any of our trifecta tickets hit
      const cached = state.predictionCache?.get(slot.raceId);
      const tickets = cached && !("skipReason" in cached) ? cached.tickets : [];
      const hitCombo =
        actual1st && actual2nd && actual3rd
          ? `${actual1st}-${actual2nd}-${actual3rd}`
          : null;
      const won = hitCombo != null && tickets.includes(hitCombo);

      // Get trifecta payout from race_payouts
      let payout = 0;
      if (won) {
        const trifectaPayout = raceResult.payouts?.find(
          (p) => p.betType === "3連単" && p.combination === hitCombo,
        );
        if (trifectaPayout) {
          // betAmount = unit × nTickets. unit = betAmount / nTickets
          const unit = tickets.length > 0 ? bet.betAmount / tickets.length : 0;
          payout = Math.round((unit / 100) * trifectaPayout.payout);
        }
      }

      state.bankroll += payout;
      results.set(slot.raceId, { won, payout });

      // Log result with finishing order detail
      const resultStr = hitCombo ?? "N/A";
      const pickHit = actual1st === bet.boatNumber ? "1着○" : "1着×";
      const pl = payout - bet.betAmount;
      const plStr =
        pl >= 0
          ? `+¥${pl.toLocaleString()}`
          : `-¥${Math.abs(pl).toLocaleString()}`;
      logger.info(
        `[TRI] ${won ? "WIN" : "LOSE"}: ${slot.stadiumName} R${slot.raceNumber} | 結果${resultStr} | 予想${bet.boatNumber}号艇${pickHit} | ${plStr} (残¥${state.bankroll.toLocaleString()})`,
      );

      await notifyResult({
        stadiumName: slot.stadiumName,
        raceNumber: slot.raceNumber,
        won,
        betAmount: bet.betAmount,
        payout,
        bankroll: state.bankroll,
      });

      slot.status = "done";
    } catch (err) {
      await notifyError(`result ${slot.stadiumName} R${slot.raceNumber}`, err);
      slot.status = "done";
    }
  }
}

// ---------------------------------------------------------------------------
// Day setup (shared by initial startup and daily restart)
// ---------------------------------------------------------------------------

const DISCOVER_RETRY_INTERVAL_MS = 5 * 60_000;
const DISCOVER_MAX_RETRIES = 6;

const stadiumNames = new Map(
  Object.entries(STADIUMS).map(([code, name]) => [
    Number.parseInt(code, 10),
    name,
  ]),
);

/** Discover venues, scrape race lists, build schedule and snapshot.
 *  Returns null if no venues found. */
async function setupDay(opts: RunnerOptions): Promise<RunnerState | null> {
  const date = todayJST();
  const yyyymmdd = todayYYYYMMDD();

  logger.info(
    `Starting runner v${config.version} for ${date} (${opts.dryRun ? "DRY RUN" : "LIVE"})`,
  );

  // 1. Discover venues (retry up to 30 min if schedule not yet published)
  let venueCodes: { stadiumCode: string; date: string }[] = [];
  for (let attempt = 0; attempt <= DISCOVER_MAX_RETRIES; attempt++) {
    if (attempt > 0) disableCacheRead();
    venueCodes = discoverDateSchedule(yyyymmdd);
    if (attempt > 0) enableCacheRead();
    if (venueCodes.length > 0) break;
    if (attempt < DISCOVER_MAX_RETRIES) {
      logger.warn(
        `No venues found yet (attempt ${attempt + 1}/${DISCOVER_MAX_RETRIES + 1}), retrying in 5 min...`,
      );
      await Bun.sleep(DISCOVER_RETRY_INTERVAL_MS);
    }
  }

  if (venueCodes.length === 0) {
    logger.warn("No venues found after retries — no races today?");
    return null;
  }

  // 2. Scrape race lists
  logger.info(`Found ${venueCodes.length} venues, scraping race lists...`);
  const scraper = getScraper("boatrace");
  if (!scraper) {
    logger.error("Scraper 'boatrace' not found");
    return null;
  }

  let totalScraped = 0;
  scraper.scrape({
    date: yyyymmdd,
    skipResults: true,
    onBatchComplete: (batch) => {
      if (batch.races.length > 0) saveRaces(batch.races);
      if (batch.results.length > 0) saveRaceResults(batch.results);
      if (batch.beforeInfo.length > 0) saveBeforeInfo(batch.beforeInfo);
      totalScraped += batch.races.length;
    },
  });
  logger.info(`Scraped ${totalScraped} races`);

  // 3. Build schedule from DB
  const db = getDatabase();
  const races = db
    .query(
      `SELECT id, stadium_id, race_number, deadline FROM races
       WHERE race_date = ? ORDER BY deadline, stadium_id, race_number`,
    )
    .all(date) as {
    id: number;
    stadium_id: number;
    race_number: number;
    deadline: string | null;
  }[];

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

  // 4. Build stats snapshot for fast inference
  const snapshotPath = await buildStatsSnapshot(date);

  const state: RunnerState = {
    schedule,
    bets: new Map(),
    results: new Map(),
    predictionCache: null,
    oddsTimings: new Map(),
    bankroll: opts.bankroll,
    date,
    lastStatusLine: "",
    snapshotPath,
  };

  await notifyStartup({
    version: config.version,
    date,
    venues: venueCodes.length,
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
  enableCache();
  initializeDatabase();

  const state = await setupDay(opts);
  if (!state) {
    closeDatabase();
    return;
  }

  // 3. Polling loop
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

  // Main loop: run daily, sleep until next day when done
  while (true) {
    // Immediate first poll
    await poll(state, opts);

    // Polling loop for the day
    let dayDone = false;
    while (!dayDone) {
      await Bun.sleep(POLL_INTERVAL_MS);
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

    // Daily summary + cleanup
    if (state.bets.size > 0) {
      let totalWagered = 0;
      let totalPayout = 0;
      let wins = 0;
      for (const [raceId, bet] of state.bets) {
        totalWagered += bet.betAmount;
        const r = state.results.get(raceId);
        if (r) {
          totalPayout += r.payout;
          if (r.won) wins++;
        }
      }
      await notifyDailySummary({
        date: state.date,
        totalBets: state.bets.size,
        wins,
        totalWagered,
        totalPayout,
        bankroll: state.bankroll,
      });
    }

    // Sleep until next day 7:00 JST
    const now = Date.now();
    const jstNow = now + 9 * 3600_000; // UTC ms → JST ms
    const jstMidnight = jstNow - (jstNow % (24 * 3600_000)); // JST midnight (in JST ms)
    const jstHourMs = jstNow - jstMidnight;
    const jst7am = 7 * 3600_000;
    const target =
      jstHourMs >= jst7am
        ? jstMidnight + 24 * 3600_000 + jst7am // tomorrow 7:00 JST
        : jstMidnight + jst7am; // today 7:00 JST
    const sleepMs = target - 9 * 3600_000 - now; // JST ms → UTC ms

    logger.info(
      `Sleeping until tomorrow 07:00 JST (${Math.round(sleepMs / 3600_000)}h)`,
    );
    await Bun.sleep(sleepMs);

    // Restart for next day
    logger.info("New day starting...");
    closeDatabase();
    initializeDatabase();

    const newState = await setupDay(opts);
    if (!newState) continue;

    Object.assign(state, newState);
  }
}
