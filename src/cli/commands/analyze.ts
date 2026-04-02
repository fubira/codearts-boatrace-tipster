import { resolve } from "node:path";
import { config } from "@/shared/config";
import { Command } from "commander";

interface EvEntry {
  ev_threshold: number;
  bets: number;
  hit_rate: number;
  roi: number;
  profit_at_100: number;
}

interface DailyEntry {
  date: string;
  bets: number;
  wins: number;
  wagered: number;
  payout: number;
  pl: number;
  bankroll: number;
}

interface KellySummary {
  initial_bankroll: number;
  bet_cap: number;
  kelly_fraction: number;
  final_bankroll: number;
  profit: number;
  total_wagered: number;
  total_payout: number;
  roi: number;
  n_bets: number;
  min_bankroll: number;
  max_bankroll: number;
}

interface AnalyzeResult {
  from_date: string;
  to_date: string;
  n_races: number;
  boat1_win_rate: number;
  val_auc: number | null;
  ev_summary: EvEntry[];
  kelly: KellySummary;
  daily: DailyEntry[];
  error?: string;
}

export const analyzeCommand = new Command("analyze")
  .description("Backtest boat 1 EV strategy over a date range")
  .requiredOption("--from <date>", "start date (YYYY-MM-DD)")
  .requiredOption("--to <date>", "end date exclusive (YYYY-MM-DD)")
  .option("--bankroll <n>", "initial bankroll", (v: string) => Number(v), 50000)
  .option("--bet-cap <n>", "max bet per race", (v: string) => Number(v), 4000)
  .option("--kelly <f>", "Kelly fraction", (v: string) => Number(v), 0.25)
  .option("--json", "output raw JSON")
  .action(async (opts) => {
    for (const d of [opts.from, opts.to]) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) {
        console.error(`Error: dates must be YYYY-MM-DD (got: ${d})`);
        process.exit(1);
      }
    }

    const proc = Bun.spawn(
      [
        "uv",
        "run",
        "--directory",
        resolve(config.projectRoot, "ml"),
        "python",
        "-m",
        "scripts.backtest_boat1",
        "--from",
        opts.from,
        "--to",
        opts.to,
        "--db-path",
        config.dbPath,
        "--bankroll",
        String(opts.bankroll),
        "--bet-cap",
        String(opts.betCap),
        "--kelly",
        String(opts.kelly),
      ],
      { stdout: "pipe", stderr: "pipe" },
    );

    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);

    // Show progress from Python
    if (stderr) process.stderr.write(stderr);

    const exitCode = await proc.exited;
    if (exitCode !== 0) {
      console.error(`Analysis failed (exit code ${exitCode})`);
      process.exit(1);
    }

    const result: AnalyzeResult = JSON.parse(stdout);

    if (result.error) {
      console.error(`Error: ${result.error}`);
      process.exit(1);
    }

    if (opts.json) {
      console.log(JSON.stringify(result, null, 2));
      return;
    }

    formatReport(result);
  });

function yen(n: number): string {
  const prefix = n >= 0 ? "+" : "-";
  return `${prefix}¥${Math.abs(n).toLocaleString()}`;
}

function formatReport(r: AnalyzeResult): void {
  const k = r.kelly;

  console.log(`\nBacktest: ${r.from_date} ~ ${r.to_date}`);
  console.log(
    `Races: ${r.n_races} | Boat1 win rate: ${(r.boat1_win_rate * 100).toFixed(1)}% | AUC: ${r.val_auc?.toFixed(4) ?? "N/A"}`,
  );

  // EV summary
  console.log("\n=== EV Strategy (fixed ¥100) ===");
  console.log(
    `  ${"EV≥".padStart(5)}  ${"Bets".padStart(5)}  ${"Hit".padStart(6)}  ${"ROI".padStart(7)}  ${"Profit".padStart(10)}`,
  );
  console.log(`  ${"─".repeat(40)}`);
  for (const e of r.ev_summary) {
    const thr = `+${e.ev_threshold}`.padStart(5);
    const bets = String(e.bets).padStart(5);
    const hit = `${(e.hit_rate * 100).toFixed(1)}%`.padStart(6);
    const roi = `${(e.roi * 100).toFixed(1)}%`.padStart(7);
    const profit = yen(e.profit_at_100).padStart(10);
    console.log(`  ${thr}  ${bets}  ${hit}  ${roi}  ${profit}`);
  }

  // Kelly simulation
  console.log(
    `\n=== Kelly ${k.kelly_fraction} + cap ¥${k.bet_cap.toLocaleString()} (bankroll ¥${k.initial_bankroll.toLocaleString()}) ===`,
  );
  console.log(
    `  ${"日付".padEnd(12)} ${"BET".padStart(4)} ${"的中".padStart(5)}  ${"投入".padStart(9)}  ${"払戻".padStart(9)}  ${"損益".padStart(10)}  ${"残高".padStart(10)}`,
  );
  console.log(`  ${"─".repeat(68)}`);

  for (const d of r.daily) {
    const date = d.date.padEnd(12);
    const bets = String(d.bets).padStart(4);
    const wins = `${d.wins}/${d.bets}`.padStart(5);
    const wagered = `¥${d.wagered.toLocaleString()}`.padStart(9);
    const payout = `¥${d.payout.toLocaleString()}`.padStart(9);
    const pl = yen(d.pl).padStart(10);
    const bank = `¥${d.bankroll.toLocaleString()}`.padStart(10);
    console.log(
      `  ${date} ${bets} ${wins}  ${wagered}  ${payout}  ${pl}  ${bank}`,
    );
  }

  console.log(`  ${"─".repeat(68)}`);
  console.log(
    `  Final: ¥${k.final_bankroll.toLocaleString()} (${yen(k.profit)}) | ROI: ${(k.roi * 100).toFixed(1)}% | ${k.n_bets} bets`,
  );
  console.log(
    `  Min: ¥${k.min_bankroll.toLocaleString()} | Max: ¥${k.max_bankroll.toLocaleString()}`,
  );
}
