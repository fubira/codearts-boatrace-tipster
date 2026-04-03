import { resolve } from "node:path";
import { config } from "@/shared/config";
import { pythonCommand } from "@/shared/python";
import { Command } from "commander";

interface Prediction {
  race_id: number;
  race_date: string;
  stadium_id: number;
  stadium_name: string;
  race_number: number;
  prob: number;
  tansho_odds: number | null;
  ev: number | null;
  recommend: boolean;
  has_exhibition: boolean;
}

interface PredictResult {
  date: string;
  model_dir: string;
  model_meta: { training?: { val_auc?: number; date_range?: string } } | null;
  n_races: number;
  predictions: Prediction[];
  error?: string;
}

export const predictCommand = new Command("predict")
  .description("Predict boat 1 win probability and bet recommendations")
  .option("-d, --date <date>", "target date (YYYY-MM-DD)")
  .option("--json", "output raw JSON")
  .option("--all", "show all races (not just recommendations)")
  .option("--model-dir <dir>", "model directory", "ml/models/boat1")
  .action(async (opts) => {
    const date = opts.date ?? new Date().toISOString().slice(0, 10);

    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      console.error("Error: --date must be YYYY-MM-DD");
      process.exit(1);
    }

    const modelDir = resolve(config.projectRoot, opts.modelDir);
    const modelPath = resolve(modelDir, "model.pkl");
    const modelFile = Bun.file(modelPath);
    if (!(await modelFile.exists())) {
      console.error(`Error: No trained model at ${modelPath}`);
      console.error(
        "Run: uv run --directory ml python -m scripts.train_boat1_binary --save",
      );
      process.exit(1);
    }

    const { cmd, cwd } = pythonCommand("scripts.predict_boat1", [
      "--date",
      date,
      "--model-dir",
      modelDir,
      "--db-path",
      config.dbPath,
    ]);
    const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "pipe", cwd });

    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);

    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      if (stderr) process.stderr.write(stderr);
      console.error(`Prediction failed (exit code ${exitCode})`);
      process.exit(1);
    }

    const result: PredictResult = JSON.parse(stdout);

    if (result.error) {
      console.error(`Error: ${result.error}`);
      process.exit(1);
    }

    if (opts.json) {
      console.log(JSON.stringify(result, null, 2));
      return;
    }

    formatTable(result, opts.all ?? false);
  });

function formatTable(result: PredictResult, showAll: boolean): void {
  const { predictions } = result;
  const recs = predictions.filter((p) => p.recommend);
  const withOdds = predictions.filter((p) => p.tansho_odds !== null);

  // Header
  const meta = result.model_meta;
  const trainInfo = meta?.training?.date_range ?? "unknown";
  console.log(`Boat 1 Predictions: ${result.date}`);
  console.log(`Model: boat1 (${trainInfo})`);
  console.log(
    `Races: ${result.n_races} | With odds: ${withOdds.length} | Bets (EV>0): ${recs.length}`,
  );
  console.log();

  // Select rows to display
  const rows = showAll ? predictions : recs;
  if (rows.length === 0) {
    console.log("No recommendations.");
    return;
  }

  // Sort by EV descending (nulls last)
  const sorted = [...rows].sort(
    (a, b) =>
      (b.ev ?? Number.NEGATIVE_INFINITY) - (a.ev ?? Number.NEGATIVE_INFINITY),
  );

  // Table
  console.log(
    `  ${"場".padEnd(8)} ${"R#".padStart(3)}  ${"予測".padStart(5)}  ${"odds".padStart(5)}  ${"EV".padStart(7)}  推奨`,
  );
  console.log(`  ${"─".repeat(42)}`);

  for (const p of sorted) {
    const stadium = p.stadium_name.padEnd(6);
    const rn = String(p.race_number).padStart(3);
    const prob = `${(p.prob * 100).toFixed(1)}%`.padStart(5);
    const odds =
      p.tansho_odds !== null ? p.tansho_odds.toFixed(1).padStart(5) : "  N/A";
    const ev =
      p.ev !== null
        ? `${p.ev >= 0 ? "+" : ""}${p.ev.toFixed(1)}%`.padStart(7)
        : "    N/A";
    const rec = p.recommend ? " ◎" : " -";
    const exh = p.has_exhibition ? "" : " *";
    console.log(`  ${stadium} ${rn}  ${prob}  ${odds}  ${ev} ${rec}${exh}`);
  }

  // Footer notes
  const noExhCount = sorted.filter((p) => !p.has_exhibition).length;
  if (noExhCount > 0) {
    console.log(
      `\n  * ${noExhCount} race(s) without exhibition data (prediction based on historical stats only)`,
    );
  }

  // Summary
  if (recs.length > 0) {
    const avgEv = recs.reduce((s, p) => s + (p.ev ?? 0), 0) / recs.length;
    const totalBet = recs.length * 100;
    console.log();
    console.log(
      `${recs.length} bets | Avg EV: +${avgEv.toFixed(1)}% | Total: ¥${totalBet.toLocaleString()} (@¥100)`,
    );
  }
}
