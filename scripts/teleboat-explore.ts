/**
 * テレボート調査スクリプト
 *
 * ヘッドフルモードでログイン → 投票画面 → 確認画面まで手動操作し、
 * 確認画面の HTML を取得する。
 *
 * Usage: bun run scripts/teleboat-explore.ts
 */
import { launchTelebotBrowser } from "@/features/teleboat/browser";
import {
  BET_SELECTORS,
  LOGIN_SELECTORS,
  MENU_SELECTORS,
  TELEBOAT_URL,
} from "@/features/teleboat/selectors";

const credentials = {
  subscriberNumber: process.env.TELEBOAT_SUBSCRIBER_NUMBER ?? "",
  pin: process.env.TELEBOAT_PIN ?? "",
  password: process.env.TELEBOAT_PASSWORD ?? "",
};

if (
  !credentials.subscriberNumber ||
  !credentials.pin ||
  !credentials.password
) {
  console.error(
    "TELEBOAT_SUBSCRIBER_NUMBER, TELEBOAT_PIN, TELEBOAT_PASSWORD を .env に設定してください",
  );
  process.exit(1);
}

const browser = await launchTelebotBrowser({ headless: false });
const { page } = browser;

// ログイン
console.log("Logging in...");
await page.goto(TELEBOAT_URL);
await page.waitForSelector(LOGIN_SELECTORS.subscriberNumber, {
  timeout: 10_000,
});
await page.fill(LOGIN_SELECTORS.subscriberNumber, credentials.subscriberNumber);
await page.fill(LOGIN_SELECTORS.pin, credentials.pin);
await page.fill(LOGIN_SELECTORS.password, credentials.password);
// ログインボタンクリックで別ウインドウが開く
const [newPage] = await Promise.all([
  browser.page.context().waitForEvent("page", { timeout: 30_000 }),
  page.click(LOGIN_SELECTORS.loginButton),
]);

// 新ウインドウに切り替え
const betPage = newPage;
await betPage.waitForLoadState("networkidle", { timeout: 30_000 });
console.log("URL:", betPage.url());

// 「特別なお知らせ」モーダルを既読にして全て閉じる
for (;;) {
  const isVisible = await betPage
    .locator(LOGIN_SELECTORS.noticeCloseButton)
    .isVisible();
  if (!isVisible) break;
  console.log("Dismissing notice modal...");
  const allRead = await betPage.$(LOGIN_SELECTORS.noticeAllRead);
  if (allRead) await allRead.check();
  await betPage.click(LOGIN_SELECTORS.noticeCloseButton);
  await betPage.waitForTimeout(500);
}

// balance 確認
const balanceEl = await betPage.$(MENU_SELECTORS.balance);
if (balanceEl) {
  const balance = await balanceEl.textContent();
  console.log(`Login success. Balance: ${balance}`);
} else {
  await Bun.write("tmp/teleboat-after-login.html", await betPage.content());
  await betPage.screenshot({
    path: "tmp/teleboat-after-login.png",
    fullPage: true,
  });
  console.log(
    "Balance not visible — check tmp/teleboat-after-login.{html,png}",
  );
}

// 確認画面のHTMLをキャプチャするリスナー
betPage.on("load", async () => {
  const url = betPage.url();
  if (url.includes("betconf")) {
    console.log("\n=== CONFIRM PAGE DETECTED ===");
    console.log("URL:", url);
    const html = await betPage.content();
    await Bun.write("tmp/teleboat-confirm.html", html);
    console.log("HTML saved to tmp/teleboat-confirm.html");
    await betPage.screenshot({
      path: "tmp/teleboat-confirm.png",
      fullPage: true,
    });
    console.log("Screenshot saved to tmp/teleboat-confirm.png");
  }
});

console.log("\n--- ブラウザを手動操作してください ---");
console.log("1. 会場を選択");
console.log("2. レースを選択");
console.log("3. 3連単タブを選択");
console.log("4. 艇番を選択して金額入力");
console.log("5. 投票入力完了を押す");
console.log("→ 確認画面のHTMLが自動保存されます");
console.log("\n終了するには Ctrl+C を押してください\n");

// ブラウザが閉じられるまで待機
await new Promise(() => {});
