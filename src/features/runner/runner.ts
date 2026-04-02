/** Daemon that automates scrape → predict → notify cycle. */

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
import type { OddsData } from "@/features/database";
import { enableCache } from "@/features/scraper/cache-manager";
import { fetchPage } from "@/features/scraper/http-client";
import { getScraper } from "@/features/scraper/registry";
import {
  type RaceParams,
  STADIUMS,
  beforeInfoUrl,
  oddsTfUrl,
  raceResultUrl,
} from "@/features/scraper/sources/boatrace/constants";
import { discoverDateSchedule } from "@/features/scraper/sources/boatrace/discovery";
import { parseOddsTf } from "@/features/scraper/sources/boatrace/odds-parsers";
import {
  parseBeforeInfo,
  parseRaceResult,
} from "@/features/scraper/sources/boatrace/parsers";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import {
  type ActionableRaces,
  type BetDecision,
  type RaceSlot,
  allDone,
  buildSchedule,
  getActionableRaces,
} from "./race-scheduler";
import {
  type DailySummaryInfo,
  notifyDailySummary,
  notifyError,
  notifyPrediction,
  notifyResult,
  notifyShutdown,
  notifyStartup,
  setSlackWebhook,
} from "./slack";

const POLL_INTERVAL_MS = 60_000;

export interface RunnerOptions {
  dryRun: boolean;
  evThreshold: number;
  betCap: number;
  kellyFraction: number;
  bankroll: number;
  slackWebhookUrl?: string;
}

type PredictionCache = Map<
  number,
  { prob: number; odds: number | null; ev: number | null }
>;

interface RunnerState {
  schedule: RaceSlot[];
  bets: Map<number, BetDecision>; // raceId → decision
  results: Map<number, { won: boolean; payout: number }>;
  predictionCache: PredictionCache | null; // null = not yet loaded
  bankroll: number;
  date: string;
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

  const page = fetchPage(beforeInfoUrl(params));
  if (!page) return false;

  const context = { params, raceDate: date };
  const data = parseBeforeInfo(page.html, context);
  if (!data) return false;

  saveBeforeInfo([data]);
  return true;
}

function scrapeTanshoOddsForRace(slot: RaceSlot, date: string): boolean {
  const params: RaceParams = {
    raceNumber: slot.raceNumber,
    stadiumCode: padStadiumCode(slot.stadiumId),
    date: date.replace(/-/g, ""),
  };

  const page = fetchPage(oddsTfUrl(params));
  if (!page) return false;

  const entries = parseOddsTf(page.html);
  if (entries.length === 0) return false;

  const oddsData: OddsData = {
    stadiumId: slot.stadiumId,
    raceDate: date,
    raceNumber: slot.raceNumber,
    entries: entries.map((e) => ({
      betType: e.betType,
      combination: e.combination,
      odds: e.odds,
    })),
  };

  saveOdds([oddsData]);
  return true;
}

function scrapeResultForRace(
  slot: RaceSlot,
  date: string,
): { won: boolean; payout: number } | null {
  const params: RaceParams = {
    raceNumber: slot.raceNumber,
    stadiumCode: padStadiumCode(slot.stadiumId),
    date: date.replace(/-/g, ""),
  };

  const page = fetchPage(raceResultUrl(params));
  if (!page) return null;

  const context = { params, raceDate: date };
  const data = parseRaceResult(page.html, context);
  if (!data) return null;

  saveRaceResults([data]);

  // Check if boat 1 won
  const boat1 = data.entries.find((e) => e.boatNumber === 1);
  const won = boat1?.finishPosition === 1;

  // Get tansho payout for boat 1
  const tansho = data.payouts?.find(
    (p) => p.betType === "単勝" && p.combination === "1",
  );
  const payout = won && tansho ? tansho.payout : 0;

  return { won, payout };
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

async function runPrediction(
  date: string,
  opts: RunnerOptions,
): Promise<
  {
    raceId: number;
    prob: number;
    odds: number | null;
    ev: number | null;
    recommend: boolean;
  }[]
> {
  const modelDir = resolve(config.projectRoot, "ml/models/boat1");
  const proc = Bun.spawn(
    [
      "uv",
      "run",
      "--directory",
      resolve(config.projectRoot, "ml"),
      "python",
      "-m",
      "scripts.predict_boat1",
      "--date",
      date,
      "--model-dir",
      modelDir,
      "--db-path",
      config.dbPath,
    ],
    { stdout: "pipe", stderr: "pipe" },
  );

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  if (stderr) logger.debug(stderr.trim());

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new Error(`predict failed (exit ${exitCode}): ${stderr}`);
  }

  const result = JSON.parse(stdout);
  return result.predictions.map(
    (p: {
      race_id: number;
      prob: number;
      tansho_odds: number | null;
      ev: number | null;
      recommend: boolean;
    }) => ({
      raceId: p.race_id,
      prob: p.prob,
      odds: p.tansho_odds,
      ev: p.ev,
      recommend: p.recommend,
    }),
  );
}

export function calcKellyBet(
  prob: number,
  odds: number,
  bankroll: number,
  kellyFraction: number,
  betCap: number,
): number {
  const edge = prob * odds - 1;
  if (edge <= 0 || odds <= 1) return 0;
  const kellyFull = edge / (odds - 1);
  const frac = Math.min(kellyFull * kellyFraction, 0.05);
  let bet = Math.floor((bankroll * frac) / 100) * 100;
  bet = Math.max(bet, 100);
  bet = Math.min(bet, betCap);
  if (bet > bankroll) bet = Math.floor(bankroll / 100) * 100;
  return bet < 100 ? 0 : bet;
}

// ---------------------------------------------------------------------------
// Poll cycle
// ---------------------------------------------------------------------------

async function poll(state: RunnerState, opts: RunnerOptions): Promise<void> {
  const { schedule, bets, results } = state;
  const now = Date.now();
  const actionable = getActionableRaces(schedule, now);

  // 1. Scrape before-info for races approaching deadline
  for (const slot of actionable.beforeInfo) {
    try {
      logger.info(
        `Scraping before-info: ${slot.stadiumName} R${slot.raceNumber}`,
      );
      const ok = scrapeBeforeInfoForRace(slot, state.date);
      if (ok) {
        slot.status = "before_info";
      }
    } catch (err) {
      await notifyError(
        `before-info ${slot.stadiumName} R${slot.raceNumber}`,
        err,
      );
    }
  }

  // 2. Scrape odds + predict for races near deadline
  if (actionable.predict.length > 0) {
    // Scrape tansho odds for each race
    for (const slot of actionable.predict) {
      try {
        logger.info(`Scraping odds: ${slot.stadiumName} R${slot.raceNumber}`);
        scrapeTanshoOddsForRace(slot, state.date);
      } catch (err) {
        await notifyError(`odds ${slot.stadiumName} R${slot.raceNumber}`, err);
      }
    }

    // Load predictions (cached: run Python only once per day)
    try {
      if (!state.predictionCache) {
        logger.info("Running prediction (first time today)...");
        const predictions = await runPrediction(state.date, opts);
        state.predictionCache = new Map();
        for (const p of predictions) {
          state.predictionCache.set(p.raceId, {
            prob: p.prob,
            odds: p.odds,
            ev: p.ev,
          });
        }
        logger.info(`Cached ${state.predictionCache.size} predictions`);
      }

      for (const slot of actionable.predict) {
        const cached = state.predictionCache.get(slot.raceId);
        if (!cached) {
          slot.status = "predicted";
          continue;
        }

        // Re-read latest odds from DB (may differ from cached odds)
        const db = getDatabase();
        const oddsRow = db
          .query(
            `SELECT odds FROM race_odds
             WHERE race_id = ? AND bet_type = '単勝' AND combination = '1'`,
          )
          .get(slot.raceId) as { odds: number } | null;

        const latestOdds = oddsRow?.odds ?? cached.odds;
        const evPct =
          latestOdds !== null
            ? (cached.prob * latestOdds - 1) * 100
            : Number.NEGATIVE_INFINITY;
        const isRecommended = evPct >= opts.evThreshold;

        if (isRecommended && latestOdds !== null) {
          const betAmount = calcKellyBet(
            cached.prob,
            latestOdds,
            state.bankroll,
            opts.kellyFraction,
            opts.betCap,
          );

          if (betAmount > 0) {
            const decision: BetDecision = {
              raceId: slot.raceId,
              stadiumName: slot.stadiumName,
              raceNumber: slot.raceNumber,
              prob: cached.prob,
              odds: latestOdds,
              ev: evPct,
              betAmount,
              recommend: true,
            };
            bets.set(slot.raceId, decision);

            await notifyPrediction({
              stadiumName: slot.stadiumName,
              raceNumber: slot.raceNumber,
              deadline: slot.deadline,
              prob: cached.prob,
              odds: latestOdds,
              ev: evPct,
              betAmount,
            });

            // DRY_RUN: deduct from virtual bankroll
            state.bankroll -= betAmount;
          }
        }

        slot.status = "predicted";
      }
    } catch (err) {
      await notifyError("prediction", err);
      for (const slot of actionable.predict) {
        slot.status = "predicted";
      }
    }
  }

  // 3. Check results for finished races
  for (const slot of actionable.results) {
    try {
      const bet = bets.get(slot.raceId);
      if (!bet) {
        slot.status = "done";
        continue;
      }

      logger.info(`Checking result: ${slot.stadiumName} R${slot.raceNumber}`);
      const result = scrapeResultForRace(slot, state.date);

      if (result === null) {
        // Result not yet available, try again later
        slot.status = "result_pending";
        continue;
      }

      // result.payout is per 100 yen; scale to actual bet amount
      const payout = result.won
        ? Math.round((bet.betAmount / 100) * result.payout)
        : 0;
      state.bankroll += payout;
      results.set(slot.raceId, { won: result.won, payout });

      await notifyResult({
        stadiumName: slot.stadiumName,
        raceNumber: slot.raceNumber,
        won: result.won,
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
  enableCache();
  initializeDatabase();

  const date = todayJST();
  const yyyymmdd = todayYYYYMMDD();

  logger.info(
    `Starting runner for ${date} (${opts.dryRun ? "DRY RUN" : "LIVE"})`,
  );

  // 1. Discover venues and scrape race lists
  const venueCodes = discoverDateSchedule(yyyymmdd);
  if (venueCodes.length === 0) {
    logger.error("No venues found for today");
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
  logger.info(`Scheduled ${schedule.length} races`);

  // 3. Pre-load predictions (run Python once at startup)
  let predictionCache: PredictionCache | null = null;
  try {
    logger.info("Running prediction for today...");
    const predictions = await runPrediction(date, opts);
    predictionCache = new Map();
    for (const p of predictions) {
      predictionCache.set(p.raceId, { prob: p.prob, odds: p.odds, ev: p.ev });
    }
    logger.info(`Cached ${predictionCache.size} predictions`);
  } catch (err) {
    logger.error(`Prediction failed at startup: ${err}`);
    await notifyError("startup prediction", err);
    // Continue without cache — poll will retry if needed
  }

  const state: RunnerState = {
    schedule,
    bets: new Map(),
    results: new Map(),
    predictionCache,
    bankroll: opts.bankroll,
    date,
  };

  await notifyStartup({
    date,
    venues: venueCodes.length,
    races: schedule.length,
    dryRun: opts.dryRun,
    evThreshold: opts.evThreshold,
  });

  // 3. Polling loop
  let timer: ReturnType<typeof setInterval> | null = null;
  let shuttingDown = false;

  async function shutdown(): Promise<void> {
    if (shuttingDown) return;
    shuttingDown = true;

    logger.info("Shutting down...");
    if (timer) clearInterval(timer);

    // Send daily summary if we have bets
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

    await notifyShutdown();
    closeDatabase();
    process.exit(0);
  }

  process.on("SIGINT", () => {
    shutdown().catch(console.error);
  });
  process.on("SIGTERM", () => {
    shutdown().catch(console.error);
  });

  // Immediate first poll
  await poll(state, opts);

  timer = setInterval(async () => {
    try {
      await poll(state, opts);

      if (allDone(schedule)) {
        logger.info("All races done for today");
        await shutdown();
      }
    } catch (err) {
      await notifyError("poll", err).catch(console.error);
    }
  }, POLL_INTERVAL_MS);

  logger.info(`Polling every ${POLL_INTERVAL_MS / 1000}s — Ctrl+C to stop`);
}
