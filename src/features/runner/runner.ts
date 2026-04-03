/** Daemon that automates scrape → predict → notify cycle. */

import { resolve } from "node:path";
import {
  closeDatabase,
  getDatabase,
  initializeDatabase,
  saveBeforeInfo,
  saveOdds,
  savePurchaseRecord,
  saveRaceResults,
  saveRaces,
} from "@/features/database";
import type { OddsData } from "@/features/database";
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
  oddsTfUrl,
  raceResultUrl,
} from "@/features/scraper/sources/boatrace/constants";
import { discoverDateSchedule } from "@/features/scraper/sources/boatrace/discovery";
import { parseOddsTf } from "@/features/scraper/sources/boatrace/odds-parsers";
import {
  parseBeforeInfo,
  parseRaceResult,
} from "@/features/scraper/sources/boatrace/parsers";
import {
  calcMaxBetForPool,
  fetchTanshoPool,
} from "@/features/scraper/sources/boatrace/pool-size";
import type { PurchaseExecutor } from "@/features/teleboat";
import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { pythonCommand } from "@/shared/python";
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

const POLL_INTERVAL_MS = 30_000;

export interface RunnerOptions {
  dryRun: boolean;
  evThreshold: number;
  betCap: number;
  kellyFraction: number;
  bankroll: number;
  slackWebhookUrl?: string;
  purchaseExecutor?: PurchaseExecutor | null;
}

interface AntiFavorite {
  boatNumber: number;
  rankProb: number;
}

type PredictionCache = Map<
  number,
  {
    prob: number;
    odds: number | null;
    ev: number | null;
    hasExhibition: boolean;
    antiFavorite: AntiFavorite | null;
  }
>;

interface RunnerState {
  schedule: RaceSlot[];
  bets: Map<number, BetDecision>; // raceId → decision
  results: Map<number, { won: boolean; payout: number }>;
  predictionCache: PredictionCache | null; // null = not yet loaded
  bankroll: number;
  date: string;
  lastStatusLine: string;
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
    hasExhibition: boolean;
    antiFavorite: AntiFavorite | null;
  }[]
> {
  const modelDir = resolve(config.projectRoot, "ml/models/boat1");
  const rankingModelDir = resolve(config.projectRoot, "ml/models/ranking");
  const { cmd, cwd } = pythonCommand("scripts.predict_boat1", [
    "--date",
    date,
    "--model-dir",
    modelDir,
    "--ranking-model-dir",
    rankingModelDir,
    "--db-path",
    config.dbPath,
  ]);
  const PREDICTION_TIMEOUT_MS = 30_000;
  const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe", cwd });

  const exited = new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => {
      proc.kill(9); // SIGKILL to ensure termination
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
  return result.predictions.map(
    (p: {
      race_id: number;
      prob: number;
      tansho_odds: number | null;
      ev: number | null;
      recommend: boolean;
      has_exhibition: boolean;
      anti_favorite?: { boat_number: number; rank_prob: number };
    }) => ({
      raceId: p.race_id,
      prob: p.prob,
      odds: p.tansho_odds,
      ev: p.ev,
      recommend: p.recommend,
      hasExhibition: p.has_exhibition,
      antiFavorite: p.anti_favorite
        ? {
            boatNumber: p.anti_favorite.boat_number,
            rankProb: p.anti_favorite.rank_prob,
          }
        : null,
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
    // Re-scrape before-info if exhibition data is missing
    for (const slot of actionable.predict) {
      const db = getDatabase();
      const hasExh = db
        .query(
          `SELECT COUNT(*) as cnt FROM race_entries
           WHERE race_id = ? AND exhibition_time IS NOT NULL`,
        )
        .get(slot.raceId) as { cnt: number };
      if (hasExh.cnt === 0) {
        logger.info(
          `Re-scraping before-info (exhibition missing): ${slot.stadiumName} R${slot.raceNumber}`,
        );
        try {
          scrapeBeforeInfoForRace(slot, state.date);
        } catch (err) {
          logger.warn(
            `Failed to re-scrape before-info: ${slot.stadiumName} R${slot.raceNumber}: ${err}`,
          );
        }
      }
    }

    // Rebuild prediction cache (ML only — no odds needed)
    const needsRebuild = actionable.predict.some(
      (s) =>
        !state.predictionCache?.has(s.raceId) ||
        !state.predictionCache?.get(s.raceId)?.hasExhibition,
    );
    try {
      if (needsRebuild) {
        logger.info("Running prediction (new exhibition data available)...");
        const predictions = await runPrediction(state.date, opts);
        state.predictionCache = new Map();
        for (const p of predictions) {
          state.predictionCache.set(p.raceId, {
            prob: p.prob,
            odds: p.odds,
            ev: p.ev,
            hasExhibition: p.hasExhibition,
            antiFavorite: p.antiFavorite,
          });
        }
        logger.info(`Predicted ${state.predictionCache.size} races`);
      }

      for (const slot of actionable.predict) {
        const cached = state.predictionCache?.get(slot.raceId);
        const exhTag = cached?.hasExhibition ? "" : " [no-exh]";
        const probPct = cached ? (cached.prob * 100).toFixed(1) : "N/A";
        logger.info(
          `Predicted: ${slot.stadiumName} R${slot.raceNumber} | prob=${probPct}%${exhTag}`,
        );
        slot.status = "predicted";
      }
    } catch (err) {
      await notifyError("prediction", err);
      for (const slot of actionable.predict) {
        slot.status = "predicted";
      }
    }
  }

  // 3. Fetch live odds + EV decision + purchase (as late as possible, T-1min)
  for (const slot of actionable.odds) {
    const cached = state.predictionCache?.get(slot.raceId);
    if (!cached) {
      slot.status = "decided";
      continue;
    }

    const label = `${slot.stadiumName} R${slot.raceNumber}`;
    const exhTag = cached.hasExhibition ? "" : " [no-exh]";
    const b1ProbPct = (cached.prob * 100).toFixed(1);
    const stadiumCode = padStadiumCode(slot.stadiumId);

    // Determine strategy: anti-favorite or boat1
    const af = cached.antiFavorite;
    const targetBoat = af ? af.boatNumber : 1;
    const targetProb = af ? af.rankProb : cached.prob;
    const strategyTag = af ? `[AF boat${targetBoat}]` : "[B1]";

    try {
      // Fetch live odds from boatrace.jp (real-time)
      const params: RaceParams = {
        raceNumber: slot.raceNumber,
        stadiumCode,
        date: state.date.replace(/-/g, ""),
      };

      logger.info(`Scraping odds (T-1min): ${label}`);
      const page = fetchPage(oddsTfUrl(params), { skipCache: true });
      if (!page) {
        logger.info(
          `EV判定: ${label} ${strategyTag} | b1=${b1ProbPct}% → SKIP (fetch failed)${exhTag}`,
        );
        slot.status = "decided";
        continue;
      }

      const entries = parseOddsTf(page.html);
      if (entries.length > 0) {
        saveOdds([
          {
            stadiumId: slot.stadiumId,
            raceDate: state.date,
            raceNumber: slot.raceNumber,
            entries: entries.map((e) => ({
              betType: e.betType,
              combination: e.combination,
              odds: e.odds,
            })),
          },
        ]);
      }

      // Read tansho odds for target boat
      const db = getDatabase();
      const oddsRow = db
        .query(
          `SELECT odds FROM race_odds
           WHERE race_id = ? AND bet_type = '単勝' AND combination = ?`,
        )
        .get(slot.raceId, String(targetBoat)) as { odds: number } | null;

      const latestOdds = oddsRow?.odds ?? null;
      if (latestOdds === null || latestOdds <= 0) {
        logger.info(
          `EV判定: ${label} ${strategyTag} | b1=${b1ProbPct}% odds=N/A → SKIP${exhTag}`,
        );
        slot.status = "decided";
        continue;
      }

      const evPct = (targetProb * latestOdds - 1) * 100;
      const isRecommended = evPct >= opts.evThreshold;
      const probPct = (targetProb * 100).toFixed(1);
      const oddsStr = latestOdds.toFixed(1);
      const evStr = `${evPct >= 0 ? "+" : ""}${evPct.toFixed(1)}%`;

      if (isRecommended) {
        let betAmount = calcKellyBet(
          targetProb,
          latestOdds,
          state.bankroll,
          opts.kellyFraction,
          opts.betCap,
        );

        // Market impact 制限 (skip for high-odds anti-favorite — pool impact is negligible)
        let poolTag = "";
        if (betAmount > 0 && !af) {
          const pool = await fetchTanshoPool(
            stadiumCode,
            state.date,
            slot.raceNumber,
          );
          if (pool && pool.totalVotes > 0) {
            const { maxBet } = calcMaxBetForPool(
              targetProb,
              latestOdds,
              pool,
              opts.betCap,
            );
            poolTag = ` pool=${pool.totalVotes}票`;
            if (maxBet < betAmount) {
              poolTag += ` cap=${maxBet}`;
              betAmount = maxBet;
            }
          } else {
            const fallback = Math.floor(opts.betCap / 2 / 100) * 100;
            if (fallback < betAmount) {
              betAmount = fallback;
              poolTag = " pool=N/A";
            }
          }
        }

        if (betAmount > 0) {
          const decision: BetDecision = {
            raceId: slot.raceId,
            stadiumName: slot.stadiumName,
            raceNumber: slot.raceNumber,
            boatNumber: targetBoat,
            prob: targetProb,
            odds: latestOdds,
            ev: evPct,
            betAmount,
            recommend: true,
          };
          bets.set(slot.raceId, decision);

          logger.info(
            `EV判定: ${label} ${strategyTag} | b1=${b1ProbPct}% prob=${probPct}% odds=${oddsStr} EV=${evStr} → BET ¥${betAmount.toLocaleString()}${exhTag}${poolTag}`,
          );

          await notifyPrediction({
            stadiumName: slot.stadiumName,
            raceNumber: slot.raceNumber,
            deadline: slot.deadline,
            prob: targetProb,
            odds: latestOdds,
            ev: evPct,
            betAmount,
          });

          // Execute purchase
          if (opts.purchaseExecutor?.isConfigured()) {
            const purchaseResult = await opts.purchaseExecutor.execute({
              stadiumCode,
              stadiumName: slot.stadiumName,
              raceNumber: slot.raceNumber,
              boatNumber: targetBoat,
              betType: "tansho",
              amount: betAmount,
            });
            savePurchaseRecord({
              raceId: slot.raceId,
              stadiumName: slot.stadiumName,
              raceNumber: slot.raceNumber,
              raceDate: state.date,
              boatNumber: targetBoat,
              betType: "単勝",
              amount: betAmount,
              dryRun: purchaseResult.dryRun,
              success: purchaseResult.success,
              error: purchaseResult.error,
              screenshotPath: purchaseResult.screenshotPath,
            });
            if (!purchaseResult.success) {
              await notifyError(
                `purchase ${slot.stadiumName} R${slot.raceNumber}`,
                purchaseResult.error,
              );
            }
          }

          state.bankroll -= betAmount;
        } else {
          logger.info(
            `EV判定: ${label} ${strategyTag} | b1=${b1ProbPct}% prob=${probPct}% odds=${oddsStr} EV=${evStr} → SKIP (bet=0)${exhTag}`,
          );
        }
      } else {
        logger.info(
          `EV判定: ${label} ${strategyTag} | b1=${b1ProbPct}% prob=${probPct}% odds=${oddsStr} EV=${evStr} → SKIP${exhTag}`,
        );
      }
    } catch (err) {
      await notifyError(`odds ${label}`, err);
    }

    slot.status = "decided";
  }

  // 4. Check results for finished races
  for (const slot of actionable.results) {
    try {
      const bet = bets.get(slot.raceId);
      if (!bet) {
        slot.status = "done";
        continue;
      }

      logger.info(`Checking result: ${slot.stadiumName} R${slot.raceNumber}`);
      const raceResult = scrapeResultForRace(slot, state.date);

      if (raceResult === null) {
        // Result not yet available, try again later
        slot.status = "result_pending";
        continue;
      }

      // Check if bet target boat won
      const targetEntry = raceResult.entries.find(
        (e) => e.boatNumber === bet.boatNumber,
      );
      const won = targetEntry?.finishPosition === 1;

      // Get tansho payout for target boat
      const tansho = raceResult.payouts?.find(
        (p) => p.betType === "単勝" && p.combination === String(bet.boatNumber),
      );
      const payoutPer100 = won && tansho ? tansho.payout : 0;
      const payout = won ? Math.round((bet.betAmount / 100) * payoutPer100) : 0;
      state.bankroll += payout;
      results.set(slot.raceId, { won, payout });

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

  // Log today's schedule overview
  const byStadium = new Map<
    string,
    { first: string; last: string; count: number }
  >();
  for (const s of schedule) {
    const entry = byStadium.get(s.stadiumName);
    if (!entry) {
      byStadium.set(s.stadiumName, {
        first: s.deadline,
        last: s.deadline,
        count: 1,
      });
    } else {
      entry.last = s.deadline;
      entry.count++;
    }
  }
  logger.info("Today's schedule:");
  for (const [name, info] of byStadium) {
    logger.info(`  ${name}: ${info.count}R (${info.first} ~ ${info.last})`);
  }
  if (schedule.length > 0) {
    logger.info(
      `  First deadline: ${schedule[0].deadline} (${schedule[0].stadiumName} R${schedule[0].raceNumber})`,
    );
    const last = schedule[schedule.length - 1];
    logger.info(
      `  Last deadline:  ${last.deadline} (${last.stadiumName} R${last.raceNumber})`,
    );
  }

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
