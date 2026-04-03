/**
 * boatcast.jp から単勝プールサイズ（票数）を取得する。
 *
 * URL: https://race.boatcast.jp/txt/{場コード}/bc_api_hyousu1_{日付}_{場コード}_{レース番号}.txt
 *
 * レスポンス形式（券種ブロック順）:
 *   3連単データ行 / 合計行
 *   3連複データ行 / 合計行
 *   単勝データ行（6艇の票数） / 合計行（合計, 返還, 流れ）
 *   複勝データ行 / 合計行
 *   選手名行
 */

import { logger } from "@/shared/logger";

const BOATCAST_BASE = "https://race.boatcast.jp/txt";
const FETCH_TIMEOUT = 5_000;

/** パリミュチュエル控除率（25%） */
const TAKEOUT = 0.75;
/** 3分前→確定でプールが約2倍になる補正 */
const POOL_MULTIPLIER = 2.0;

export interface TanshoPoolInfo {
  /** 単勝の総票数 */
  totalVotes: number;
  /** 単勝の総金額（票数 × 100円） */
  poolSize: number;
  /** 各艇の票数 [1号艇, 2号艇, ..., 6号艇] */
  votesByBoat: number[];
}

/**
 * 単勝プールサイズを取得する。
 * 取得失敗時は null を返す（呼び出し側でフォールバック）。
 */
export async function fetchTanshoPool(
  stadiumCode: string,
  date: string,
  raceNumber: number,
): Promise<TanshoPoolInfo | null> {
  const dateStr = date.replace(/-/g, "");
  const raceStr = String(raceNumber).padStart(2, "0");
  const url = `${BOATCAST_BASE}/${stadiumCode}/bc_api_hyousu1_${dateStr}_${stadiumCode}_${raceStr}.txt`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);

    if (!res.ok) {
      logger.debug(`Pool size fetch failed: ${res.status} ${url}`);
      return null;
    }

    const text = await res.text();
    return parseTanshoPool(text);
  } catch (err) {
    logger.debug(`Pool size fetch error: ${err}`);
    return null;
  }
}

/**
 * hyousu1 テキストから単勝プール情報をパースする。
 *
 * 構造: 各券種ブロックが「データ行 / 合計行」のペア。
 * 合計行は "数値\t数値\t数値" の3列（合計, 返還, 流れ）。
 * 単勝ブロックは3番目（0-indexed: index 2）。
 */
export function parseTanshoPool(text: string): TanshoPoolInfo | null {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith("data="));

  // Skip first line (always "1" or "2")
  const dataLines = lines.slice(1);

  let summaryIndex = 0;
  for (let i = 0; i < dataLines.length; i++) {
    const cols = dataLines[i].split("\t").filter((c) => c.length > 0);

    if (cols.length === 3 && cols.every((c) => /^\d+$/.test(c))) {
      summaryIndex++;

      // 3rd summary = 単勝
      if (summaryIndex === 3) {
        const totalVotes = Number.parseInt(cols[0], 10);
        const dataLine = dataLines[i - 1];
        const voteCols = dataLine
          .split("\t")
          .filter((c) => c.length > 0)
          .slice(0, 6)
          .map((v) => Number.parseInt(v, 10) || 0);

        return {
          totalVotes,
          poolSize: totalVotes * 100,
          votesByBoat: voteCols,
        };
      }
    }
  }

  return null;
}

/**
 * post-bet EV > 0 を維持する最大 bet 額を計算する。
 *
 * パリミュチュエル方式: 配当は最終プールで決まる。
 * 3分前のプール × POOL_MULTIPLIER で最終プールを推定。
 *
 * odds_after = TAKEOUT × (finalPool + X) / (boat1Pool + X)
 * EV_after = prob × odds_after - 1 > 0
 * → X < (TAKEOUT × prob × finalPool - boat1Pool) / (1 - TAKEOUT × prob)
 */
export function calcMaxBetForPool(
  prob: number,
  odds: number,
  pool: TanshoPoolInfo,
  betCap: number,
): { maxBet: number; estimatedPool: number; boat1Pool: number } {
  const estimatedPool = pool.poolSize * POOL_MULTIPLIER;
  let boat1Pool = (TAKEOUT * estimatedPool) / odds;
  if (boat1Pool > estimatedPool) {
    boat1Pool = estimatedPool * 0.9;
  }

  const denom = 1 - TAKEOUT * prob;
  let maxBet: number;
  if (denom <= 0) {
    // prob so high that any bet maintains positive EV
    maxBet = betCap;
  } else {
    maxBet = (TAKEOUT * prob * estimatedPool - boat1Pool) / denom;
  }

  maxBet = Math.max(0, Math.floor(maxBet / 100) * 100);
  maxBet = Math.min(maxBet, betCap);

  return { maxBet, estimatedPool, boat1Pool: Math.round(boat1Pool) };
}
