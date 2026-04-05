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
import { STADIUM_CODES } from "@/features/teleboat/selectors";
import { createTelebotClient } from "@/features/teleboat/teleboat-client";
import type { TelebotCredentials } from "@/features/teleboat/types";

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

const credentials: TelebotCredentials = {
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
const client = createTelebotClient(browser);

try {
  // 1. ログイン
  console.log("[1/4] ログイン中...");
  await client.login(credentials);
  const { availableBalance } = await client.getBalance();
  console.log(`  残高: ¥${availableBalance.toLocaleString()}`);

  // 2. 入金（残高不足時）
  if (availableBalance < totalAmount) {
    console.log(
      `[2/4] 入金中... ¥${depositNeeded.toLocaleString()}（残高 ¥${availableBalance.toLocaleString()} < 必要額 ¥${totalAmount.toLocaleString()}）`,
    );
    await client.deposit(depositNeeded, credentials.betPassword);
    const { availableBalance: newBalance } = await client.getBalance();
    console.log(`  残高反映: ¥${newBalance.toLocaleString()}`);
  } else {
    console.log(
      `[2/4] 入金不要（残高 ¥${availableBalance.toLocaleString()} >= 必要額 ¥${totalAmount.toLocaleString()}）`,
    );
  }

  // 3. 投票（ベットリスト追加まで）
  console.log("[3/4] 投票セット中...");
  const result = await client.placeBet(
    {
      stadiumCode,
      stadiumName,
      raceNumber,
      betType: "sanrentan",
      boats1st: [boat1st],
      boats2nd,
      boats3rd: boats2nd,
      amount: amountPerBet,
    },
    true, // dry-run: ベットリスト追加まで
  );

  if (result.success) {
    console.log();
    console.log("===========================================");
    console.log("  [4/4] ベットリスト追加完了！");
    console.log("  ブラウザで内容を確認し「投票入力完了」を押してください");
    console.log("===========================================");
  } else {
    console.error(`投票失敗: ${result.error}`);
  }

  console.log();
  console.log("終了するには Ctrl+C を押してください");
  await new Promise(() => {});
} catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
  console.error(`エラー: ${msg}`);
  await browser.close();
  process.exit(1);
}
