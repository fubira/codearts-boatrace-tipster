/**
 * テレボートクライアント（SP版）
 *
 * Playwright でテレボート SP 版を操作し、単勝舟券を購入する。
 * tateyamakun-trader-jra の ipat-client.ts をボートレース向けに踏襲。
 */
import { mkdirSync } from "node:fs";
import { logger } from "@/shared/logger";
import type { TelebotBrowser } from "./browser";
import {
  BET_SELECTORS,
  CONFIRM_SELECTORS,
  LOGIN_SELECTORS,
  MENU_SELECTORS,
  RACE_SELECTORS,
  STADIUM_SELECTORS,
  TELEBOAT_URL,
} from "./selectors";
import type {
  TelebotBalance,
  TelebotBetOrder,
  TelebotBetResult,
  TelebotCredentials,
} from "./types";

const SCREENSHOT_DIR = "tmp/teleboat-screenshots";
const NAV_TIMEOUT = 15_000;
const CLICK_TIMEOUT = 10_000;
const BET_TIMEOUT = 30_000;

export interface TelebotClient {
  login(credentials: TelebotCredentials): Promise<void>;
  getBalance(): Promise<TelebotBalance>;
  placeBet(order: TelebotBetOrder, dryRun: boolean): Promise<TelebotBetResult>;
  close(): Promise<void>;
}

export function createTelebotClient(browser: TelebotBrowser): TelebotClient {
  const { page } = browser;
  mkdirSync(SCREENSHOT_DIR, { recursive: true });

  async function screenshotWithTimestamp(label: string): Promise<string> {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `${SCREENSHOT_DIR}/${ts}_${label}.png`;
    await browser.screenshot(path);
    return path;
  }

  async function login(credentials: TelebotCredentials): Promise<void> {
    logger.info("Teleboat: Login started");
    await page.goto(TELEBOAT_URL);
    await page.waitForSelector(LOGIN_SELECTORS.subscriberNumber, {
      timeout: CLICK_TIMEOUT,
    });

    await page.fill(
      LOGIN_SELECTORS.subscriberNumber,
      credentials.subscriberNumber,
    );
    await page.fill(LOGIN_SELECTORS.pin, credentials.pin);
    await page.fill(LOGIN_SELECTORS.password, credentials.password);

    await page.click(LOGIN_SELECTORS.loginButton);
    await page.waitForSelector(MENU_SELECTORS.balance, {
      timeout: NAV_TIMEOUT,
    });

    logger.info("Teleboat: Login completed");
    await screenshotWithTimestamp("login-success");
  }

  async function getBalance(): Promise<TelebotBalance> {
    const text = await page.textContent(MENU_SELECTORS.balance);
    const numStr = (text ?? "0").replace(/[,，円\s]/g, "");
    const availableBalance = Number.parseInt(numStr, 10) || 0;

    return {
      availableBalance,
      queriedAt: new Date().toISOString(),
    };
  }

  async function placeBet(
    order: TelebotBetOrder,
    dryRun: boolean,
  ): Promise<TelebotBetResult> {
    const label = `${order.stadiumName}${order.raceNumber}R-${order.boatNumber}号艇`;
    logger.info(
      `Teleboat: Bet started ${label} ¥${order.amount}${dryRun ? " (dry-run)" : ""}`,
    );

    const timeout = new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new Error(`placeBet timeout: ${BET_TIMEOUT}ms`)),
        BET_TIMEOUT,
      ),
    );

    return Promise.race([placeBetInternal(order, label, dryRun), timeout]);
  }

  async function placeBetInternal(
    order: TelebotBetOrder,
    label: string,
    dryRun: boolean,
  ): Promise<TelebotBetResult> {
    try {
      // 1. 投票メニューへ
      logger.info(`Teleboat step 1: purchase menu ${label}`);
      await page.click(MENU_SELECTORS.purchaseMenu);
      await page.waitForSelector(STADIUM_SELECTORS.stadiumList, {
        timeout: CLICK_TIMEOUT,
      });

      // 2. 会場を選択
      logger.info(`Teleboat step 2: stadium select ${label}`);
      const stadiumItems = await page.$$(STADIUM_SELECTORS.stadiumItem);
      let stadiumFound = false;
      for (const item of stadiumItems) {
        const text = await item.textContent();
        if (text?.includes(order.stadiumName)) {
          await item.click();
          stadiumFound = true;
          break;
        }
      }
      if (!stadiumFound) {
        throw new Error(`会場「${order.stadiumName}」が見つかりません`);
      }

      // 3. レースを選択
      logger.info(`Teleboat step 3: race select ${label}`);
      await page.waitForSelector(RACE_SELECTORS.raceItem, {
        timeout: CLICK_TIMEOUT,
      });
      const raceItems = await page.$$(RACE_SELECTORS.raceItem);
      let raceFound = false;
      for (const item of raceItems) {
        const numEl = await item.$(RACE_SELECTORS.raceNum);
        const numText = await numEl?.textContent();
        if (numText?.trim() === `${order.raceNumber}R`) {
          await item.click();
          raceFound = true;
          break;
        }
      }
      if (!raceFound) {
        throw new Error(`${order.raceNumber}Rが見つかりません`);
      }

      // 4. 単勝を選択
      logger.info(`Teleboat step 4: bet type select ${label}`);
      await page.waitForSelector(BET_SELECTORS.tanshoTab, {
        timeout: CLICK_TIMEOUT,
      });
      await page.click(BET_SELECTORS.tanshoTab);

      // 5. 艇番を選択
      logger.info(`Teleboat step 5: boat select ${label}`);
      await page.waitForSelector(BET_SELECTORS.boatNumber, {
        timeout: CLICK_TIMEOUT,
      });
      await page.click(BET_SELECTORS.boatNumber);

      // 6. 金額入力
      if (order.amount % 100 !== 0 || order.amount <= 0) {
        throw new Error(
          `金額は100円単位の正数で指定してください: ¥${order.amount}`,
        );
      }
      const units = order.amount / 100;
      const amountInput = await page.waitForSelector(
        BET_SELECTORS.amountInput,
        {
          timeout: CLICK_TIMEOUT,
        },
      );
      await amountInput.fill(String(units));
      await page.click(BET_SELECTORS.setButton);

      // 7. 確認画面
      logger.info(`Teleboat step 7: confirm screen ${label}`);
      await page.waitForSelector(CONFIRM_SELECTORS.voteList, {
        timeout: CLICK_TIMEOUT,
      });
      await screenshotWithTimestamp(`confirm-${label}`);

      // 確認画面のバリデーション
      const voteItems = await page.$$(CONFIRM_SELECTORS.voteList);
      if (voteItems.length === 0) {
        throw new Error("投票内容が確認画面に表示されていません");
      }

      // 8. dry-run: キャンセル / live: 投票実行
      if (dryRun) {
        logger.info(`Teleboat: Dry-run, cancelling ${label}`);
        await page.click(CONFIRM_SELECTORS.cancelButton);
        const screenshotPath = await screenshotWithTimestamp(
          `dryrun-cancel-${label}`,
        );
        return {
          order,
          success: true,
          screenshotPath,
          completedAt: new Date().toISOString(),
          dryRun: true,
        };
      }

      // LIVE: 合計金額入力 → 投票
      logger.info(`Teleboat step 8: submit vote ${label}`);
      await page.fill(CONFIRM_SELECTORS.totalInput, String(order.amount));

      const [dialog] = await Promise.all([
        page.waitForEvent("dialog", { timeout: CLICK_TIMEOUT }),
        page.click(CONFIRM_SELECTORS.submitButton),
      ]);
      if (dialog.type() !== "confirm") {
        await dialog.dismiss();
        throw new Error(
          `予期しないダイアログ: type=${dialog.type()} message="${dialog.message()}"`,
        );
      }
      await dialog.accept();
      await page.waitForLoadState("networkidle", { timeout: NAV_TIMEOUT });
      const screenshotPath = await screenshotWithTimestamp(`complete-${label}`);

      logger.info(`Teleboat: Bet completed ${label}`);
      return {
        order,
        success: true,
        screenshotPath,
        completedAt: new Date().toISOString(),
        dryRun: false,
      };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      logger.error(`Teleboat: Bet failed ${label}: ${errorMsg}`);
      await screenshotWithTimestamp(`error-${label}`);

      return {
        order,
        success: false,
        error: errorMsg,
        completedAt: new Date().toISOString(),
        dryRun,
      };
    }
  }

  return {
    login,
    getBalance,
    placeBet,
    close: () => browser.close(),
  };
}
