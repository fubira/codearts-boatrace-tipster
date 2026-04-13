import { runDaemon } from "@/features/runner/runner";
import { createPurchaseExecutor } from "@/features/teleboat";
import { loadModelStrategy, loadTelebotCredentials } from "@/shared/config";
import { Command } from "commander";

// NOTE: loadModelStrategy() is called lazily inside .action() below.
// Calling it at module top-level would read ml/models/active.json during
// import, which breaks the scraper container (no ml/models mount) when
// commander loads every command module at startup.

export const runCommand = new Command("run")
  .description("Run P2 prediction daemon (predict → drift → notify)")
  .option("--dry-run", "DRY RUN mode (no real purchases)", true)
  .option("--live", "LIVE mode (execute real purchases)")
  .option(
    "--ev-threshold <n>",
    "EV threshold as fraction (default: from active model)",
    (v: string) => Number(v),
  )
  .option(
    "--bet-cap <n>",
    "max unit per ticket ¥ (default: from active model)",
    (v: string) => Number(v),
  )
  .option(
    "--unit-divisor <n>",
    "unit = bankroll / divisor (default: from active model)",
    (v: string) => Number(v),
  )
  .option("--bankroll <n>", "initial bankroll", (v: string) => Number(v), 70000)
  .action(async (opts) => {
    const strategy = loadModelStrategy();
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
      evThreshold: opts.evThreshold ?? strategy.evThreshold,
      gap23Threshold: strategy.gap23Threshold,
      top3ConcThreshold: strategy.top3ConcThreshold,
      gap12MinThreshold: strategy.gap12MinThreshold,
      betCap: opts.betCap ?? strategy.betCap,
      unitDivisor: opts.unitDivisor ?? strategy.unitDivisor,
      bankroll: opts.bankroll,
      slackWebhookUrl,
      purchaseExecutor,
    });
  });
