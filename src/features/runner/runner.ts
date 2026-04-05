/** Daemon that automates scrape → predict → notify cycle. */

import { existsSync, mkdirSync, readdirSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import {
  closeDatabase,
  getDatabase,
  initializeDatabase,
  saveBeforeInfo,
  saveOdds,
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
  tickets: string[];
  hasExhibition: boolean;
}

// string value = skip reason (e.g. "no_odds", "ev_low:-0.123")
type PredictionCache = Map<number, TrifectaPrediction | string>;

interface RunnerState {
  schedule: RaceSlot[];
  bets: Map<number, BetDecision>; // raceId → decision
  results: Map<number, { won: boolean; payout: number }>;
  predictionCache: PredictionCache | null; // null = not yet loaded
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

  const SNAPSHOT_TIMEOUT_MS = 120_000;
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
  } catch {
    proc.kill(9);
    logger.error(
      `Snapshot build timed out after ${SNAPSHOT_TIMEOUT_MS / 1000}s`,
    );
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
    "--ev-threshold",
    "0", // Return all EV>0 candidates; runner filters by opts.evThreshold
  ];
  if (snapshotPath) {
    args.push("--snapshot", snapshotPath);
  }
  if (raceIds && raceIds.length > 0) {
    args.push("--race-ids", raceIds.join(","));
  }
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
      `Trifecta stats: ${s.total} races, ${s.b1_pass} upset, ${s.has_odds} with odds, ${s.ev_pass} EV-pass`,
    );
  }

  const predictions = result.predictions.map(
    (p: {
      race_id: number;
      winner_pick: number;
      b1_prob: number;
      winner_prob: number;
      ev: number;
      tickets: string[];
      has_exhibition: boolean;
    }) => ({
      raceId: p.race_id,
      winnerPick: p.winner_pick,
      b1Prob: p.b1_prob,
      winnerProb: p.winner_prob,
      ev: p.ev,
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

/** Fetch trifecta odds for actionable races. Returns count of fetched. */
function fetchTrifectaOdds(slots: RaceSlot[], date: string): number {
  let fetched = 0;
  for (const slot of slots) {
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
          saveOdds([
            {
              stadiumId: slot.stadiumId,
              raceDate: date,
              raceNumber: slot.raceNumber,
              entries: entries.map((e) => ({
                betType: e.betType,
                combination: e.combination,
                odds: e.odds,
              })),
            },
          ]);
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
    if (typeof cached === "string") return true;
    return cached !== undefined && !cached.hasExhibition;
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
    const b1Pct = (info.b1_prob * 100).toFixed(0);
    const pickTag = info.pick ? ` ${info.pick}号艇1着` : "";
    const skipStr =
      info.ev !== undefined
        ? `${info.reason}${pickTag} b1=${b1Pct}% EV=${(info.ev * 100).toFixed(1)}%`
        : `${info.reason} b1=${b1Pct}%`;
    state.predictionCache.set(rid, skipStr);
  }
  for (const p of result.predictions) {
    state.predictionCache.set(p.raceId, {
      winnerPick: p.winnerPick,
      b1Prob: p.b1Prob,
      winnerProb: p.winnerProb,
      ev: p.ev,
      tickets: p.tickets,
      hasExhibition: p.hasExhibition,
    });
  }
  logger.debug(
    `Prediction cache: ${result.predictions.length} qualifying / ${result.evaluatedRaceIds.length} evaluated`,
  );
}

/** Predict and make bet decisions for actionable races. */
async function pollPredict(
  state: RunnerState,
  slots: RaceSlot[],
  opts: RunnerOptions,
  bets: Map<number, BetDecision>,
): Promise<void> {
  // Re-scrape before-info if exhibition data is missing
  const db = getDatabase();
  let rescraped = 0;
  for (const slot of slots) {
    const hasExh = db
      .query(
        `SELECT COUNT(*) as cnt FROM race_entries
         WHERE race_id = ? AND exhibition_time IS NOT NULL`,
      )
      .get(slot.raceId) as { cnt: number };
    if (hasExh.cnt === 0) {
      try {
        scrapeBeforeInfoForRace(slot, state.date);
        rescraped++;
      } catch {
        // Prediction will proceed without exhibition data
      }
    }
  }
  if (rescraped > 0) {
    logger.info(`Re-scraped exhibition data for ${rescraped} race(s)`);
  }

  const oddsFetched = fetchTrifectaOdds(slots, state.date);
  logger.info(`Trifecta odds: ${oddsFetched}/${slots.length} fetched`);

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

    await makeBetDecisions(slots, state, opts, bets);
  } catch (err) {
    await notifyError("prediction", err);
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
    if (!cached || typeof cached === "string") {
      logger.info(
        `[TRI] SKIP: ${slot.stadiumName} R${slot.raceNumber} | ${cached ?? "unknown"}`,
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

    const base = `${label} | ${cached.winnerPick}号艇1着 | b1=${(cached.b1Prob * 100).toFixed(0)}% EV=+${evPct.toFixed(1)}% | ${cached.tickets.length}pt`;

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
    }
    logger.info(
      `Before-info: ${scraped}/${actionable.beforeInfo.length} scraped`,
    );
  }

  // 2. Predict for races near deadline (trifecta strategy)
  if (actionable.predict.length > 0) {
    await pollPredict(state, actionable.predict, opts, bets);
  }

  // 3. Skip oddsTf (T-1) — no longer used in trifecta strategy
  for (const slot of actionable.odds) {
    slot.status = "decided";
  }

  // 4. Check results for finished races (trifecta)
  for (const slot of actionable.results) {
    try {
      const bet = bets.get(slot.raceId);
      if (!bet) {
        slot.status = "done";
        continue;
      }

      const raceResult = scrapeResultForRace(slot, state.date);

      if (raceResult === null) {
        slot.status = "result_pending";
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
      const tickets =
        cached && typeof cached !== "string" ? cached.tickets : [];
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
// Main daemon
// ---------------------------------------------------------------------------

export async function runDaemon(opts: RunnerOptions): Promise<void> {
  setSlackWebhook(opts.slackWebhookUrl);
  enableFileLog(resolve(config.projectRoot, "logs"));
  enableCache();
  initializeDatabase();

  const date = todayJST();
  const yyyymmdd = todayYYYYMMDD();

  logger.info(
    `Starting runner v${config.version} for ${date} (${opts.dryRun ? "DRY RUN" : "LIVE"})`,
  );

  // 1. Discover venues (retry up to 30 min if schedule not yet published)
  const DISCOVER_RETRY_INTERVAL_MS = 5 * 60_000;
  const DISCOVER_MAX_RETRIES = 6;
  let venueCodes: { stadiumCode: string; date: string }[] = [];

  for (let attempt = 0; attempt <= DISCOVER_MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      // Bypass cache on retry — previous empty response may be cached
      disableCacheRead();
    }
    venueCodes = discoverDateSchedule(yyyymmdd);
    if (attempt > 0) {
      enableCacheRead();
    }
    if (venueCodes.length > 0) break;

    if (attempt < DISCOVER_MAX_RETRIES) {
      logger.warn(
        `No venues found yet (attempt ${attempt + 1}/${DISCOVER_MAX_RETRIES + 1}), retrying in 5 min...`,
      );
      await Bun.sleep(DISCOVER_RETRY_INTERVAL_MS);
    }
  }

  if (venueCodes.length === 0) {
    logger.error("No venues found after retries — no races today?");
    closeDatabase();
    return;
  }

  logger.info(`Found ${venueCodes.length} venues, scraping race lists...`);

  const scraper = getScraper("boatrace");
  if (!scraper) {
    logger.error("Scraper 'boatrace' not found");
    closeDatabase();
    return;
  }

  let totalScraped = 0;
  scraper.scrape({
    date: yyyymmdd,
    skipResults: true, // Don't cache empty result pages for today's races
    onBatchComplete: (batch) => {
      if (batch.races.length > 0) saveRaces(batch.races);
      if (batch.results.length > 0) saveRaceResults(batch.results);
      if (batch.beforeInfo.length > 0) saveBeforeInfo(batch.beforeInfo);
      totalScraped += batch.races.length;
    },
  });

  logger.info(`Scraped ${totalScraped} races`);

  // 2. Build schedule from DB
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

  const stadiumNames = new Map(
    Object.entries(STADIUMS).map(([code, name]) => [
      Number.parseInt(code, 10),
      name,
    ]),
  );

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

  // Build stats snapshot for fast inference
  const snapshotPath = await buildStatsSnapshot(date);

  // Prediction cache is built on-demand when races reach "predict" state
  // (after exhibition data has been scraped)
  const predictionCache: PredictionCache | null = null;

  const state: RunnerState = {
    schedule,
    bets: new Map(),
    results: new Map(),
    predictionCache,
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
        if (allDone(schedule)) {
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

    const newDate = todayJST();
    const newYYYYMMDD = todayYYYYMMDD();

    logger.info(
      `Starting runner v${config.version} for ${newDate} (${opts.dryRun ? "DRY RUN" : "LIVE"})`,
    );

    // Re-discover venues
    let newVenueCodes: { stadiumCode: string; date: string }[] = [];
    for (let attempt = 0; attempt <= 6; attempt++) {
      if (attempt > 0) disableCacheRead();
      newVenueCodes = discoverDateSchedule(newYYYYMMDD);
      if (attempt > 0) enableCacheRead();
      if (newVenueCodes.length > 0) break;
      if (attempt < 6) {
        logger.warn(
          `No venues found (attempt ${attempt + 1}/7), retrying in 5 min...`,
        );
        await Bun.sleep(5 * 60_000);
      }
    }

    if (newVenueCodes.length === 0) {
      logger.warn("No venues found — no races today? Sleeping until tomorrow.");
      continue;
    }

    logger.info(`Found ${newVenueCodes.length} venues, scraping race lists...`);
    const newScraper = getScraper("boatrace");
    if (newScraper) {
      newScraper.scrape({
        date: newYYYYMMDD,
        skipResults: true,
        onBatchComplete: (batch) => {
          if (batch.races.length > 0) saveRaces(batch.races);
          if (batch.results.length > 0) saveRaceResults(batch.results);
          if (batch.beforeInfo.length > 0) saveBeforeInfo(batch.beforeInfo);
        },
      });
    }

    const newDb = getDatabase();
    const newRaces = newDb
      .query(
        `SELECT id, stadium_id, race_number, deadline FROM races
         WHERE race_date = ? ORDER BY deadline, stadium_id, race_number`,
      )
      .all(newDate) as {
      id: number;
      stadium_id: number;
      race_number: number;
      deadline: string | null;
    }[];

    const newSchedule = buildSchedule(newRaces, stadiumNames, newDate);
    if (newSchedule.length > 0) {
      const newVenues = new Set(newSchedule.map((s) => s.stadiumName));
      const newFirst = newSchedule[0];
      const newLast = newSchedule[newSchedule.length - 1];
      logger.info(
        `Schedule: ${newSchedule.length}R / ${newVenues.size} venues | ${newFirst.deadline} ~ ${newLast.deadline}`,
      );
    }

    // Build stats snapshot for new day
    const newSnapshotPath = await buildStatsSnapshot(newDate);

    // Reset state for new day
    state.schedule = newSchedule;
    state.bets = new Map();
    state.results = new Map();
    state.predictionCache = null;
    state.date = newDate;
    state.lastStatusLine = "";
    state.snapshotPath = newSnapshotPath;

    await notifyStartup({
      version: config.version,
      date: newDate,
      venues: newVenueCodes.length,
      races: newSchedule.length,
      dryRun: opts.dryRun,
      evThreshold: opts.evThreshold,
    });
  }
}
