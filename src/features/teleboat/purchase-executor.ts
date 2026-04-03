/**
 * テレボート購入実行器
 *
 * one-shot セッション（起動→ログイン→購入→終了）で1件ずつ処理。
 * リトライ1回。dry-run 時は確認画面でキャンセル。
 */
import { logger } from "@/shared/logger";
import { launchTelebotBrowser } from "./browser";
import { createTelebotClient } from "./teleboat-client";
import type {
  TelebotBetOrder,
  TelebotBetResult,
  TelebotCredentials,
} from "./types";

const MAX_RETRIES = 1;
const CLOSE_TIMEOUT = 5_000;

export interface PurchaseExecutor {
  execute(order: TelebotBetOrder): Promise<TelebotBetResult>;
  isConfigured(): boolean;
}

export function createPurchaseExecutor(options: {
  credentials: TelebotCredentials;
  dryRun: boolean;
}): PurchaseExecutor {
  const { credentials, dryRun } = options;

  async function execute(order: TelebotBetOrder): Promise<TelebotBetResult> {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (attempt > 0) {
        logger.warn(
          `Teleboat: Retry ${attempt}/${MAX_RETRIES} for ${order.stadiumName} R${order.raceNumber}`,
        );
      }

      const result = await executeOnce(order);
      if (result.success) return result;

      if (attempt === MAX_RETRIES) return result;
    }

    // Unreachable, but TypeScript needs it
    throw new Error("Unexpected: exceeded retry loop");
  }

  async function executeOnce(
    order: TelebotBetOrder,
  ): Promise<TelebotBetResult> {
    let browser: Awaited<ReturnType<typeof launchTelebotBrowser>> | null = null;

    try {
      browser = await launchTelebotBrowser();
      const client = createTelebotClient(browser);

      await client.login(credentials);

      // 残高チェック
      const balance = await client.getBalance();
      if (!dryRun && balance.availableBalance < order.amount) {
        logger.warn(
          `Teleboat: Insufficient balance ¥${balance.availableBalance} < ¥${order.amount}`,
        );
        return {
          order,
          success: false,
          error: `残高不足: ¥${balance.availableBalance} < ¥${order.amount}`,
          completedAt: new Date().toISOString(),
          dryRun,
        };
      }

      return await client.placeBet(order, dryRun);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger.error(`Teleboat: Execute failed: ${errorMsg}`);
      return {
        order,
        success: false,
        error: errorMsg,
        completedAt: new Date().toISOString(),
        dryRun,
      };
    } finally {
      if (browser) {
        const closePromise = browser.close();
        const timeout = new Promise<void>((resolve) =>
          setTimeout(resolve, CLOSE_TIMEOUT),
        );
        await Promise.race([closePromise, timeout]);
      }
    }
  }

  return {
    execute,
    isConfigured: () => true,
  };
}
