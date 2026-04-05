/**
 * テレボート入金テストスクリプト
 *
 * ログイン → 残高確認 → 入金 → 残高確認 で入金フローを検証。
 *
 * Usage:
 *   bun run scripts/teleboat-deposit-test.ts --amount 1000
 */
import { parseArgs } from "node:util";
import { launchTelebotBrowser } from "@/features/teleboat/browser";
import { createTelebotClient } from "@/features/teleboat/teleboat-client";
import type { TelebotCredentials } from "@/features/teleboat/types";

const { values } = parseArgs({
  options: {
    amount: { type: "string", short: "a", default: "1000" },
  },
});

const amount = Number(values.amount);
if (amount < 1000 || amount % 1000 !== 0) {
  console.error("入金額は1000円単位で指定してください");
  process.exit(1);
}

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
  console.log("[1/3] ログイン中...");
  await client.login(credentials);
  const { availableBalance: before } = await client.getBalance();
  console.log(`  残高: ¥${before.toLocaleString()}`);

  // 2. 入金
  console.log(`[2/3] 入金 ¥${amount.toLocaleString()}...`);
  await client.deposit(amount, credentials.betPassword);

  // 3. 残高確認
  console.log("[3/3] 残高確認...");
  const { availableBalance: after } = await client.getBalance();
  console.log(
    `  残高: ¥${before.toLocaleString()} → ¥${after.toLocaleString()}`,
  );

  console.log();
  console.log("=== 入金テスト完了 ===");
  console.log("終了するには Ctrl+C を押してください");
  await new Promise(() => {});
} catch (error) {
  const msg = error instanceof Error ? error.message : String(error);
  console.error(`エラー: ${msg}`);
  await browser.close();
  process.exit(1);
}
