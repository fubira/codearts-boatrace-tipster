/**
 * テレボートクライアント（PC版）
 *
 * Playwright で BOAT RACE インターネット投票を操作し、舟券を購入する。
 * tateyamakun-trader-jra の ipat-client.ts をボートレース向けに踏襲。
 *
 * 投票フロー:
 *   トップ → 会場クリック → 投票画面 → 賭式タブ → [フォーメーション] → 艇番選択
 *   → 金額入力 → ベットリスト追加 → 投票入力完了 → 確認画面 → 投票実行
 */
import { mkdirSync } from "node:fs";
import { logger } from "@/shared/logger";
import type { TelebotBrowser } from "./browser";
import {
  BETLIST_SELECTORS,
  BET_SELECTORS,
  BET_URL_PATTERN,
  CHARGE_SELECTORS,
  CONFIRM_SELECTORS,
  LOGIN_SELECTORS,
  MENU_SELECTORS,
  RACE_SELECTORS,
  STADIUM_SELECTORS,
  TELEBOAT_URL,
  TOP_URL_PATTERN,
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
const BET_TIMEOUT = 60_000;

export interface TelebotClient {
  login(credentials: TelebotCredentials): Promise<void>;
  getBalance(): Promise<TelebotBalance>;
  deposit(amount: number, betPassword: string): Promise<void>;
  placeBet(order: TelebotBetOrder, dryRun: boolean): Promise<TelebotBetResult>;
  close(): Promise<void>;
}

export function createTelebotClient(browser: TelebotBrowser): TelebotClient {
  let page = browser.page;
  mkdirSync(SCREENSHOT_DIR, { recursive: true });

  async function screenshotWithTimestamp(label: string): Promise<string> {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `${SCREENSHOT_DIR}/${ts}_${label}.png`;
    await page.screenshot({ path, fullPage: true });
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

    // ログインボタンで別ウインドウが開く
    const [betPage] = await Promise.all([
      page.context().waitForEvent("page", { timeout: NAV_TIMEOUT }),
      page.click(LOGIN_SELECTORS.loginButton),
    ]);
    page = betPage;
    await page.waitForLoadState("networkidle", { timeout: NAV_TIMEOUT });

    // 「特別なお知らせ」モーダルが表示されていたら既読にして全て閉じる
    for (;;) {
      const isVisible = await page
        .locator(LOGIN_SELECTORS.noticeCloseButton)
        .isVisible();
      if (!isVisible) break;
      logger.info("Teleboat: Dismissing notice modal");
      const allRead = await page.$(LOGIN_SELECTORS.noticeAllRead);
      if (allRead) await allRead.check();
      await page.click(LOGIN_SELECTORS.noticeCloseButton);
      await page.waitForTimeout(500);
    }

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

  async function deposit(amount: number, betPassword: string): Promise<void> {
    if (amount <= 0 || amount % 1000 !== 0) {
      throw new Error(`入金額は1000円単位の正数で指定してください: ¥${amount}`);
    }
    const units = amount / 1000;
    logger.info(`Teleboat: Deposit ¥${amount} (${units}千円)`);

    // 入金メニューを開く
    await page.click(MENU_SELECTORS.charge);
    await page.waitForSelector(CHARGE_SELECTORS.amountInput, {
      timeout: CLICK_TIMEOUT,
    });

    // 金額と投票用パスワードを入力
    await page.fill(CHARGE_SELECTORS.amountInput, String(units));
    await page.fill(CHARGE_SELECTORS.betPassword, betPassword);
    await screenshotWithTimestamp("deposit-before");

    // 入金実行
    await page.click(CHARGE_SELECTORS.executeButton);

    // 完了画面を待って閉じる
    await page.waitForSelector(CHARGE_SELECTORS.closeCompButton, {
      timeout: NAV_TIMEOUT,
    });
    await screenshotWithTimestamp("deposit-complete");
    await page.click(CHARGE_SELECTORS.closeCompButton);

    logger.info(`Teleboat: Deposit completed ¥${amount}`);
  }

  async function placeBet(
    order: TelebotBetOrder,
    dryRun: boolean,
  ): Promise<TelebotBetResult> {
    const boatLabel =
      order.betType === "sanrentan"
        ? `${order.boats1st.join("")}-${(order.boats2nd ?? []).join("")}-${(order.boats3rd ?? []).join("")}`
        : `${order.boats1st.join("")}号艇`;
    const label = `${order.stadiumName}${order.raceNumber}R-${order.betType}-${boatLabel}`;
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

  /** フォーメーション投票で艇番セルをクリック */
  async function selectFormationBoats(
    boats: number[],
    column: number,
  ): Promise<void> {
    for (const boat of boats) {
      const selector = `${BET_SELECTORS.formationBoatCell}.x${boat}.y${column}`;
      await page.waitForSelector(selector, { timeout: CLICK_TIMEOUT });
      await page.click(selector);
    }
  }

  /** 通常投票で艇番ボタンをクリック */
  async function selectNormalBoats(
    boats: number[],
    column: number,
  ): Promise<void> {
    for (const boat of boats) {
      const selector = `${BET_SELECTORS.normalBoatPrefix}${boat}_${column}`;
      await page.waitForSelector(selector, { timeout: CLICK_TIMEOUT });
      await page.click(selector);
    }
  }

  async function placeBetInternal(
    order: TelebotBetOrder,
    label: string,
    dryRun: boolean,
  ): Promise<TelebotBetResult> {
    try {
      // 1. トップページの会場をクリック → 投票画面へ遷移
      logger.info(`Teleboat step 1: stadium click ${label}`);
      const stadiumSelector = `${STADIUM_SELECTORS.stadiumIdPrefix}${order.stadiumCode} a`;
      await page.waitForSelector(stadiumSelector, { timeout: CLICK_TIMEOUT });
      await page.click(stadiumSelector);
      await page.waitForURL(BET_URL_PATTERN, { timeout: NAV_TIMEOUT });

      // 2. レースタブをクリック
      const raceNo = String(order.raceNumber).padStart(2, "0");
      const raceSelector = `${RACE_SELECTORS.raceTabPrefix}${raceNo}`;
      logger.info(`Teleboat step 2: race select ${label}`);
      await page.waitForSelector(raceSelector, { timeout: CLICK_TIMEOUT });
      await page.click(raceSelector);

      // 3. 賭式タブをクリック
      const betTypeTab =
        order.betType === "sanrentan"
          ? BET_SELECTORS.sanrentanTab
          : BET_SELECTORS.tanshoTab;
      logger.info(`Teleboat step 3: ${order.betType} tab ${label}`);
      await page.waitForSelector(betTypeTab, { timeout: CLICK_TIMEOUT });
      await page.click(betTypeTab);

      // 3.5 フォーメーション投票タブをクリック（3連単時）
      if (order.betType === "sanrentan") {
        logger.info(`Teleboat step 3.5: formation tab ${label}`);
        await page.waitForSelector(BET_SELECTORS.formationBetWay, {
          timeout: CLICK_TIMEOUT,
        });
        await page.click(BET_SELECTORS.formationBetWay);
      }

      // 4. 艇番を選択
      logger.info(`Teleboat step 4: boat select ${label}`);
      if (order.betType === "sanrentan") {
        if (!order.boats2nd?.length || !order.boats3rd?.length) {
          throw new Error("3連単は2着・3着の指定が必要です");
        }
        await selectFormationBoats(order.boats1st, 1);
        await selectFormationBoats(order.boats2nd, 2);
        await selectFormationBoats(order.boats3rd, 3);
      } else {
        await selectNormalBoats(order.boats1st, 1);
      }

      // 5. 金額入力（100円単位 — 入力値 × 100 = 実金額）
      if (order.amount % 100 !== 0 || order.amount <= 0) {
        throw new Error(
          `金額は100円単位の正数で指定してください: ¥${order.amount}`,
        );
      }
      const units = order.amount / 100;
      logger.info(`Teleboat step 5: amount ${units} units ${label}`);
      const amountInput = await page.waitForSelector(
        BET_SELECTORS.amountInput,
        {
          timeout: CLICK_TIMEOUT,
        },
      );
      await amountInput.fill(String(units));

      // 6. ベットリストに追加
      const addButton =
        order.betType === "sanrentan"
          ? BET_SELECTORS.formationAddToBetList
          : BET_SELECTORS.normalAddToBetList;
      logger.info(`Teleboat step 6: add to bet list ${label}`);
      await page.click(addButton);

      // 7. 投票入力完了 → 確認画面へ
      logger.info(`Teleboat step 7: submit bet list ${label}`);
      await page.waitForSelector(BETLIST_SELECTORS.submitButton, {
        timeout: CLICK_TIMEOUT,
      });
      await screenshotWithTimestamp(`betlist-${label}`);
      await page.click(BETLIST_SELECTORS.submitButton);

      // 8. 確認画面
      // TODO: 確認画面のセレクタが埋まったら以下を実装
      // - 投票内容のバリデーション
      // - dry-run: キャンセル / live: 投票用パスワード入力 → 投票実行
      logger.info(`Teleboat step 8: confirm screen ${label}`);
      await screenshotWithTimestamp(`confirm-${label}`);

      if (dryRun) {
        logger.info(`Teleboat: Dry-run completed ${label}`);
        // TODO: 確認画面でキャンセルクリック
        // await page.click(CONFIRM_SELECTORS.cancelButton);
        return {
          order,
          success: true,
          completedAt: new Date().toISOString(),
          dryRun: true,
        };
      }

      // LIVE: TODO — 確認画面セレクタ実装後に有効化
      throw new Error(
        "LIVE mode not yet implemented — confirm screen selectors needed",
      );
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
    deposit,
    placeBet,
    close: () => browser.close(),
  };
}
