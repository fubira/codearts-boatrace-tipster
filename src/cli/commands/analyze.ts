import { config } from "@/shared/config";
import { pythonCommand } from "@/shared/python";
import { Command } from "commander";

interface DailyEntry {
  date: string;
  n_races: number;
  n_tickets: number;
  n_wins: number;
  payout: number;
  pl: number;
  cum_pl: number;
}

interface BacktestResult {
  from_date: string;
  to_date: string;
  days: number;
  races: number;
  tickets: number;
  wins: number;
  payout: number;
  roi: number;
  daily: DailyEntry[];
  error?: string;
}

export const analyzeCommand = new Command("analyze")
  .description("Backtest trifecta X-noB1-noB1 strategy over a date range")
  .requiredOption("--from <date>", "start date (YYYY-MM-DD)")
  .requiredOption("--to <date>", "end date exclusive (YYYY-MM-DD)")
  .option(
    "--b1-threshold <n>",
    "b1 probability threshold",
    (v: string) => Number(v),
    0.4,
  )
  .option(
    "--ev-threshold <n>",
    "EV threshold as fraction (e.g. 0.33)",
    (v: string) => Number(v),
    0.33,
  )
  .option("--json", "output raw JSON")
  .action(async (opts) => {
    for (const d of [opts.from, opts.to]) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) {
        console.error(`Error: dates must be YYYY-MM-DD (got: ${d})`);
        process.exit(1);
      }
    }

    const { cmd, cwd } = pythonCommand("scripts.backtest_trifecta", [
      "--from",
      opts.from,
      "--to",
      opts.to,
      "--b1-threshold",
      String(opts.b1Threshold),
      "--ev-threshold",
      String(opts.evThreshold),
      "--db-path",
      config.dbPath,
      ...(opts.json ? ["--json"] : []),
    ]);
    const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe", cwd });

    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);

    if (stderr) process.stderr.write(stderr);

    const exitCode = await proc.exited;
    if (exitCode !== 0) {
      console.error(`Analysis failed (exit code ${exitCode})`);
      process.exit(1);
    }

    if (opts.json) {
      // backtest_trifecta outputs JSON directly when --json is passed
      process.stdout.write(stdout);
      return;
    }

    // Without --json, backtest_trifecta outputs formatted text directly
    process.stdout.write(stdout);
  });
