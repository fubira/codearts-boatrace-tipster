/**
 * テレボート即時購入スクリプト
 *
 * ログイン → 入金（必要時） → 3連単フォーメーション投票 → ベットリスト追加まで自動実行。
 * 「投票入力完了」はマスターが手動で押す。
 *
 * Usage:
 *   bun run scripts/teleboat-buy.ts --stadium 02 --race 3 --boat 2 --amount 100
 *
 * Options:
 *   --stadium  場コード（2桁: 01=桐生, 02=戸田, ... 24=大村）
 *   --race     レース番号（1-12）
 *   --boat     1着予測の艇番（2-6、1号艇は除外対象）
 *   --amount   1点あたりの金額（100円単位、デフォルト: 100）
 */
import { parseArgs } from "node:util";
import { launchTelebotBrowser } from "@/features/teleboat/browser";
import {
  BETLIST_SELECTORS,
  BET_SELECTORS,
  CHARGE_SELECTORS,
  LOGIN_SELECTORS,
  MENU_SELECTORS,
  RACE_SELECTORS,
  STADIUM_CODES,
  STADIUM_SELECTORS,
  TELEBOAT_URL,
} from "@/features/teleboat/selectors";

const { values } = parseArgs({
  options: {
    stadium: { type: "string", short: "s" },
    race: { type: "string", short: "r" },
    boat: { type: "string", short: "b" },
    amount: { type: "string", short: "a", default: "100" },
  },
});

const stadiumCode = values.stadium ?? "";
const raceNumber = Number(values.race);
const boat1st = Number(values.boat);
const amountPerBet = Number(values.amount);

// バリデーション
if (!stadiumCode || !STADIUM_CODES[stadiumCode]) {
  console.error(`無効な場コード: ${stadiumCode}`);
  console.error(
    "有効な場コード:",
    Object.entries(STADIUM_CODES)
      .map(([k, v]) => `${k}=${v}`)
      .join(", "),
  );
  process.exit(1);
}
if (raceNumber < 1 || raceNumber > 12 || !Number.isInteger(raceNumber)) {
  console.error(`無効なレース番号: ${values.race}`);
  process.exit(1);
}
if (boat1st < 2 || boat1st > 6 || !Number.isInteger(boat1st)) {
  console.error(`無効な艇番: ${values.boat}（2-6を指定、1号艇は除外対象）`);
  process.exit(1);
}
if (amountPerBet < 100 || amountPerBet % 100 !== 0) {
  console.error(`無効な金額: ${values.amount}（100円単位）`);
  process.exit(1);
}

const stadiumName = STADIUM_CODES[stadiumCode];
const boats2nd = [2, 3, 4, 5, 6].filter((b) => b !== boat1st);
const totalAmount = amountPerBet * 12;
const depositNeeded = Math.ceil(totalAmount / 1000) * 1000;

console.log("=== テレボート購入 ===");
console.log(`場: ${stadiumName}(${stadiumCode}) ${raceNumber}R`);
console.log(
  `3連単フォーメーション: ${boat1st}-${boats2nd.join("")}-${boats2nd.join("")}`,
);
console.log(`金額: ¥${amountPerBet} × 12点 = ¥${totalAmount}`);
console.log();

const credentials = {
  subscriberNumber: process.env.TELEBOAT_SUBSCRIBER_NUMBER ?? "",
  pin: process.env.TELEBOAT_PIN ?? "",
  password: process.env.TELEBOAT_PASSWORD ?? "",
  betPassword: process.env.TELEBOAT_BET_PASSWORD ?? "",
};

if (
  !credentials.subscriberNumber ||
  !credentials.pin ||
  !credentials.password ||
  !credentials.betPassword
) {
  console.error(
    "TELEBOAT_SUBSCRIBER_NUMBER, TELEBOAT_PIN, TELEBOAT_PASSWORD, TELEBOAT_BET_PASSWORD を .env に設定してください",
  );
  process.exit(1);
}

const browser = await launchTelebotBrowser({ headless: false });
const { page: loginPage } = browser;

try {
  // 1. ログイン
  console.log("[1/6] ログイン中...");
  await loginPage.goto(TELEBOAT_URL);
  await loginPage.waitForSelector(LOGIN_SELECTORS.subscriberNumber, {
    timeout: 10_000,
  });
  await loginPage.fill(
    LOGIN_SELECTORS.subscriberNumber,
    credentials.subscriberNumber,
  );
  await loginPage.fill(LOGIN_SELECTORS.pin, credentials.pin);
  await loginPage.fill(LOGIN_SELECTORS.password, credentials.password);

  const [betPage] = await Promise.all([
    loginPage.context().waitForEvent("page", { timeout: 30_000 }),
    loginPage.click(LOGIN_SELECTORS.loginButton),
  ]);
  await betPage.waitForLoadState("networkidle", { timeout: 30_000 });

  // お知らせモーダルをスキップ
  for (;;) {
    const isVisible = await betPage
      .locator(LOGIN_SELECTORS.noticeCloseButton)
      .isVisible();
    if (!isVisible) break;
    console.log("  お知らせモーダルをスキップ...");
    const allRead = await betPage.$(LOGIN_SELECTORS.noticeAllRead);
    if (allRead) await allRead.check();
    await betPage.click(LOGIN_SELECTORS.noticeCloseButton);
    await betPage.waitForTimeout(500);
  }

  await betPage.waitForSelector(MENU_SELECTORS.balance, { timeout: 15_000 });
  const balanceText = await betPage.textContent(MENU_SELECTORS.balance);
  const balance = Number.parseInt(
    (balanceText ?? "0").replace(/[,，円\s]/g, ""),
    10,
  );
  console.log(`  ログイン成功 残高: ¥${balance.toLocaleString()}`);

  // 2. 入金（残高不足時）
  if (balance < totalAmount) {
    console.log(
      `[2/6] 入金中... ¥${depositNeeded.toLocaleString()}（残高 ¥${balance.toLocaleString()} < 必要額 ¥${totalAmount.toLocaleString()}）`,
    );
    await betPage.click(MENU_SELECTORS.charge);
    await betPage.waitForSelector(CHARGE_SELECTORS.amountInput, {
      timeout: 10_000,
    });
    await betPage.fill(
      CHARGE_SELECTORS.amountInput,
      String(depositNeeded / 1000),
    );
    await betPage.fill(CHARGE_SELECTORS.betPassword, credentials.betPassword);
    await betPage.click(CHARGE_SELECTORS.executeButton);
    await betPage.waitForSelector(CHARGE_SELECTORS.closeCompButton, {
      timeout: 30_000,
    });
    console.log(`  入金完了 ¥${depositNeeded.toLocaleString()}`);
    await betPage.click(CHARGE_SELECTORS.closeCompButton);
    await betPage.waitForTimeout(500);
  } else {
    console.log(
      `[2/6] 入金不要（残高 ¥${balance.toLocaleString()} >= 必要額 ¥${totalAmount.toLocaleString()}）`,
    );
  }

  // 3. 会場選択
  console.log(`[3/6] ${stadiumName} を選択...`);
  const stadiumSelector = `${STADIUM_SELECTORS.stadiumIdPrefix}${stadiumCode} a`;
  await betPage.waitForSelector(stadiumSelector, { timeout: 10_000 });
  await betPage.click(stadiumSelector);
  await betPage.waitForURL("**/service/bet/betcom/**", { timeout: 15_000 });

  // 4. レース選択 + 3連単フォーメーション
  const raceNo = String(raceNumber).padStart(2, "0");
  console.log(`[4/6] ${raceNumber}R → 3連単フォーメーション...`);
  await betPage.waitForSelector(`${RACE_SELECTORS.raceTabPrefix}${raceNo}`, {
    timeout: 10_000,
  });
  await betPage.click(`${RACE_SELECTORS.raceTabPrefix}${raceNo}`);

  // 3連単タブ
  await betPage.waitForSelector(BET_SELECTORS.sanrentanTab, {
    timeout: 10_000,
  });
  await betPage.click(BET_SELECTORS.sanrentanTab);

  // フォーメーションタブ
  await betPage.waitForSelector(BET_SELECTORS.formationBetWay, {
    timeout: 10_000,
  });
  await betPage.click(BET_SELECTORS.formationBetWay);

  // 5. 艇番選択
  console.log(
    `[5/6] 艇番選択: 1着=${boat1st} 2着=${boats2nd.join(",")} 3着=${boats2nd.join(",")}...`,
  );

  // 1着
  const sel1st = `${BET_SELECTORS.formationBoatCell}.x${boat1st}.y1`;
  await betPage.waitForSelector(sel1st, { timeout: 10_000 });
  await betPage.click(sel1st);

  // 2着・3着
  for (const boat of boats2nd) {
    await betPage.click(`${BET_SELECTORS.formationBoatCell}.x${boat}.y2`);
    await betPage.click(`${BET_SELECTORS.formationBoatCell}.x${boat}.y3`);
  }

  // ベット数確認
  const betCount = await betPage.textContent(BET_SELECTORS.formationBetCount);
  console.log(`  組合せ数: ${betCount}`);

  // 金額入力
  const units = amountPerBet / 100;
  await betPage.fill(BET_SELECTORS.amountInput, String(units));

  // ベットリストに追加
  console.log("[6/6] ベットリストに追加...");
  await betPage.click(BET_SELECTORS.formationAddToBetList);

  // ベットリスト反映を確認
  await betPage.waitForTimeout(1000);
  const totalAmountText = await betPage.textContent(
    BETLIST_SELECTORS.totalAmount,
  );
  console.log(`  総購入金額: ${totalAmountText}円`);

  console.log();
  console.log("===========================================");
  console.log("  ベットリスト追加完了！");
  console.log("  ブラウザで内容を確認し「投票入力完了」を押してください");
  console.log("===========================================");
  console.log();
  console.log("終了するには Ctrl+C を押してください");

  // ブラウザが閉じられるまで待機
  await new Promise(() => {});
} catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
  console.error(`エラー: ${msg}`);
  await browser.close();
  process.exit(1);
}
