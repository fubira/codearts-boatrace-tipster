import { runDaemon } from "@/features/runner/runner";
import { createPurchaseExecutor } from "@/features/teleboat";
import { loadModelStrategy, loadTelebotCredentials } from "@/shared/config";
import { Command } from "commander";

const strategy = loadModelStrategy();

export const runCommand = new Command("run")
  .description("Run P2 prediction daemon (predict → drift → notify)")
  .option("--dry-run", "DRY RUN mode (no real purchases)", true)
  .option("--live", "LIVE mode (execute real purchases)")
  .option(
    "--ev-threshold <n>",
    `EV threshold as fraction (default: ${strategy.evThreshold})`,
    (v: string) => Number(v),
    strategy.evThreshold,
  )
  .option(
    "--bet-cap <n>",
    `max unit per ticket ¥ (default: ${strategy.betCap})`,
    (v: string) => Number(v),
    strategy.betCap,
  )
  .option(
    "--unit-divisor <n>",
    `unit = bankroll / divisor (default: ${strategy.unitDivisor})`,
    (v: string) => Number(v),
    strategy.unitDivisor,
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
      unitDivisor: opts.unitDivisor,
      bankroll: opts.bankroll,
      slackWebhookUrl,
      purchaseExecutor,
    });
  });
