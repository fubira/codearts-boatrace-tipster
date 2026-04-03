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
    "EV threshold for bets (pre-deadline odds are approximate)",
    (v: string) => Number(v),
    0,
  )
  .option("--bet-cap <n>", "max bet per race", (v: string) => Number(v), 4000)
  .option("--kelly <f>", "Kelly fraction", (v: string) => Number(v), 0.25)
  .option("--bankroll <n>", "initial bankroll", (v: string) => Number(v), 50000)
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
      kellyFraction: opts.kelly,
      bankroll: opts.bankroll,
      slackWebhookUrl,
      purchaseExecutor,
    });
  });
