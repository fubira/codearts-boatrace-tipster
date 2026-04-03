import { logger } from "@/shared/logger";
import {
  type Browser,
  type BrowserContext,
  type Page,
  chromium,
} from "playwright";

export interface TelebotBrowser {
  page: Page;
  close: () => Promise<void>;
  screenshot: (path: string) => Promise<void>;
}

export async function launchTelebotBrowser({
  headless = true,
}: { headless?: boolean } = {}): Promise<TelebotBrowser> {
  let browser: Browser | null = null;
  let context: BrowserContext | null = null;

  try {
    browser = await chromium.launch({ headless });
    context = await browser.newContext({
      locale: "ja-JP",
      timezoneId: "Asia/Tokyo",
      viewport: { width: 375, height: 812 },
    });
    const page = await context.newPage();

    return {
      page,
      async close() {
        await context?.close();
        await browser?.close();
        logger.info("Teleboat browser closed");
      },
      async screenshot(path: string) {
        await page.screenshot({ path, fullPage: true });
      },
    };
  } catch (error) {
    await context?.close();
    await browser?.close();
    throw error;
  }
}
