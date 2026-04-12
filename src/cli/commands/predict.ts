import { resolve } from "node:path";
import { config } from "@/shared/config";
import { pythonCommand } from "@/shared/python";
import { Command } from "commander";

interface P2Ticket {
  combo: string;
  model_prob: number;
  market_odds: number;
  ev: number;
}

interface P2Prediction {
  race_id: number;
  race_date: string;
  stadium_id: number;
  stadium_name: string;
  race_number: number;
  top3_conc: number;
  gap23: number;
  tickets: P2Ticket[];
  has_exhibition: boolean;
}

interface PredictP2Result {
  date: string;
  model_dir: string;
  strategy: string;
  gap23_threshold: number;
  top3_conc_threshold: number;
  ev_threshold: number;
  n_races: number;
  predictions: P2Prediction[];
  evaluated_race_ids: number[];
  skipped: Record<
    number,
    { reason: string; top1_boat?: number; top3_conc?: number; gap23?: number }
  >;
  stats?: {
    total: number;
    b1_top: number;
    conc_pass: number;
    gap23_pass: number;
    predicted: number;
  };
  error?: string;
}

export const predictCommand = new Command("predict")
  .description("Predict P2 trifecta strategy (1-(2,3)-(2,3) adaptive)")
  .option("-d, --date <date>", "target date (YYYY-MM-DD)")
  .option("--json", "output raw JSON")
  .option("--model-dir <dir>", "model directory", "ml/models/p2_v1")
  .option("--use-snapshots", "read odds from race_odds_snapshots")
  .action(async (opts) => {
    const date = opts.date ?? new Date().toISOString().slice(0, 10);

    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      console.error("Error: --date must be YYYY-MM-DD");
      process.exit(1);
    }

    const modelDir = resolve(config.projectRoot, opts.modelDir);
    const modelFile = Bun.file(resolve(modelDir, "ranking/model.pkl"));
    if (!(await modelFile.exists())) {
      console.error(`Error: No ranking model at ${modelDir}/ranking/model.pkl`);
      process.exit(1);
    }

    const args = [
      "--date",
      date,
      "--model-dir",
      modelDir,
      "--db-path",
      config.dbPath,
    ];
    if (opts.useSnapshots) {
      args.push("--use-snapshots");
    }

    const { cmd, cwd } = pythonCommand("scripts.predict_p2", args);
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

    const result: PredictP2Result = JSON.parse(stdout);

    if (result.error) {
      console.error(`Error: ${result.error}`);
      process.exit(1);
    }

    if (opts.json) {
      console.log(JSON.stringify(result, null, 2));
      return;
    }

    formatP2Table(result);
  });

function formatP2Table(result: PredictP2Result): void {
  const { predictions, stats } = result;
  const concTh = (result.top3_conc_threshold * 100).toFixed(0);
  const gapTh = (result.gap23_threshold * 100).toFixed(1);
  const evTh = (result.ev_threshold * 100).toFixed(0);

  console.log(`P2 Predictions: ${result.date}`);
  console.log(
    `Model: ${result.model_dir.split("/").slice(-1)[0]} | conc>=${concTh}% gap23>=${gapTh}% EV>=${evTh}%`,
  );
  if (stats) {
    console.log(
      `Funnel: ${stats.total}R → B1 top ${stats.b1_top} → conc ${stats.conc_pass} → gap23 ${stats.gap23_pass} → predicted ${stats.predicted}`,
    );
  }
  console.log();

  if (predictions.length === 0) {
    console.log("No qualifying races.");
    return;
  }

  // Sort by best ticket EV desc
  const sorted = [...predictions].sort((a, b) => {
    const maxA = Math.max(...a.tickets.map((t) => t.ev));
    const maxB = Math.max(...b.tickets.map((t) => t.ev));
    return maxB - maxA;
  });

  console.log(
    `  ${"場".padEnd(8)} ${"R#".padStart(3)}  ${"conc".padStart(5)}  ${"gap23".padStart(6)}  ${"pt".padStart(2)}  買い目 (odds / EV)`,
  );
  console.log(`  ${"─".repeat(72)}`);

  for (const p of sorted) {
    const stadium = p.stadium_name.padEnd(6);
    const rn = String(p.race_number).padStart(3);
    const conc = `${(p.top3_conc * 100).toFixed(0)}%`.padStart(5);
    const gap = `${(p.gap23 * 100).toFixed(1)}%`.padStart(6);
    const pt = String(p.tickets.length).padStart(2);
    const ticketStr = p.tickets
      .map(
        (t) =>
          `${t.combo} (${t.market_odds.toFixed(1)}x / +${(t.ev * 100).toFixed(0)}%)`,
      )
      .join(" ");
    console.log(`  ${stadium} ${rn}  ${conc}  ${gap}  ${pt}  ${ticketStr}`);
  }

  const totalTickets = predictions.reduce((s, p) => s + p.tickets.length, 0);
  const avgEv =
    predictions.reduce(
      (s, p) =>
        s + p.tickets.reduce((ss, t) => ss + t.ev, 0) / p.tickets.length,
      0,
    ) / predictions.length;
  console.log();
  console.log(
    `${predictions.length} races | ${totalTickets} tickets | Avg EV: +${(avgEv * 100).toFixed(1)}% | @¥100: ¥${(totalTickets * 100).toLocaleString()}`,
  );
}
