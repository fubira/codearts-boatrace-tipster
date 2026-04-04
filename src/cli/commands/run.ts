import { runDaemon } from "@/features/runner/runner";
import { createPurchaseExecutor } from "@/features/teleboat";
import { loadTelebotCredentials } from "@/shared/config";
import { Command } from "commander";

export const runCommand = new Command("run")
  .description("Run prediction daemon (scrape → predict → notify)")
  .option("--dry-run", "DRY RUN mode (no real purchases)", true)
  .option("--live", "LIVE mode (execute real purchases)")
  .option(
    "--ev-threshold <n>",
    "EV threshold as fraction (e.g. 0.33 = 33%)",
    (v: string) => Number(v),
    0.33,
  )
  .option(
    "--bet-cap <n>",
    "max unit per ticket (¥)",
    (v: string) => Number(v),
    2000,
  )
  .option("--bankroll <n>", "initial bankroll", (v: string) => Number(v), 70000)
  .action(async (opts) => {
    const dryRun = !opts.live;
    const slackWebhookUrl = process.env.SLACK_WEBHOOK_URL;

    if (!slackWebhookUrl) {
      console.warn(
        "SLACK_WEBHOOK_URL not set — notifications will be logged to console only",
      );
    }

    const credentials = loadTelebotCredentials();
    const purchaseExecutor = credentials
      ? createPurchaseExecutor({ credentials, dryRun })
      : null;

    if (credentials) {
      console.log(`Teleboat: configured (${dryRun ? "DRY RUN" : "LIVE"})`);
    }

    await runDaemon({
      dryRun,
      evThreshold: opts.evThreshold,
      betCap: opts.betCap,
      bankroll: opts.bankroll,
      slackWebhookUrl,
      purchaseExecutor,
    });
  });
