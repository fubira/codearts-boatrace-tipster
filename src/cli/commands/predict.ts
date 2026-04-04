import { resolve } from "node:path";
import { config } from "@/shared/config";
import { pythonCommand } from "@/shared/python";
import { Command } from "commander";

interface TrifectaPrediction {
  race_id: number;
  race_date: string;
  stadium_id: number;
  stadium_name: string;
  race_number: number;
  winner_pick: number;
  b1_prob: number;
  winner_prob: number;
  ev: number;
  tickets: string[];
  has_exhibition: boolean;
}

interface PredictResult {
  date: string;
  model_dir: string;
  b1_threshold: number;
  ev_threshold: number;
  n_races: number;
  predictions: TrifectaPrediction[];
  error?: string;
}

export const predictCommand = new Command("predict")
  .description("Predict trifecta X-noB1-noB1 strategy")
  .option("-d, --date <date>", "target date (YYYY-MM-DD)")
  .option("--json", "output raw JSON")
  .option("--model-dir <dir>", "model directory", "ml/models/trifecta_v1")
  .option(
    "--ev-threshold <n>",
    "EV threshold as fraction (e.g. 0.33 = 33%)",
    (v: string) => Number(v),
    0.33,
  )
  .action(async (opts) => {
    const date = opts.date ?? new Date().toISOString().slice(0, 10);

    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      console.error("Error: --date must be YYYY-MM-DD");
      process.exit(1);
    }

    const modelDir = resolve(config.projectRoot, opts.modelDir);
    const b1ModelPath = resolve(modelDir, "boat1/model.pkl");
    const b1File = Bun.file(b1ModelPath);
    if (!(await b1File.exists())) {
      console.error(`Error: No trained model at ${b1ModelPath}`);
      process.exit(1);
    }

    const { cmd, cwd } = pythonCommand("scripts.predict_trifecta", [
      "--date",
      date,
      "--model-dir",
      modelDir,
      "--db-path",
      config.dbPath,
      "--ev-threshold",
      String(opts.evThreshold),
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

    formatTrifectaTable(result);
  });

function formatTrifectaTable(result: PredictResult): void {
  const { predictions } = result;

  console.log(`Trifecta Predictions: ${result.date}`);
  console.log(
    `Model: trifecta_v1 | b1<${(result.b1_threshold * 100).toFixed(0)}% EV>=${(result.ev_threshold * 100).toFixed(0)}%`,
  );
  console.log(`Qualifying races: ${result.n_races}`);
  console.log();

  if (predictions.length === 0) {
    console.log("No qualifying races.");
    return;
  }

  const sorted = [...predictions].sort((a, b) => b.ev - a.ev);

  console.log(
    `  ${"場".padEnd(8)} ${"R#".padStart(3)}  ${"1着".padStart(4)}  ${"b1%".padStart(5)}  ${"EV".padStart(7)}  ${"pt".padStart(3)}  買い目`,
  );
  console.log(`  ${"─".repeat(60)}`);

  for (const p of sorted) {
    const stadium = p.stadium_name.padEnd(6);
    const rn = String(p.race_number).padStart(3);
    const winner = String(p.winner_pick).padStart(4);
    const b1 = `${(p.b1_prob * 100).toFixed(0)}%`.padStart(5);
    const ev = `+${(p.ev * 100).toFixed(1)}%`.padStart(7);
    const pt = String(p.tickets.length).padStart(3);
    const ticketStr = p.tickets.slice(0, 3).join(" ");
    const more = p.tickets.length > 3 ? ` +${p.tickets.length - 3}` : "";
    const exh = p.has_exhibition ? "" : " *";
    console.log(
      `  ${stadium} ${rn}  ${winner}  ${b1}  ${ev}  ${pt}  ${ticketStr}${more}${exh}`,
    );
  }

  const noExhCount = sorted.filter((p) => !p.has_exhibition).length;
  if (noExhCount > 0) {
    console.log(`\n  * ${noExhCount} race(s) without exhibition data`);
  }

  const totalTickets = predictions.reduce((s, p) => s + p.tickets.length, 0);
  const avgEv = predictions.reduce((s, p) => s + p.ev, 0) / predictions.length;
  console.log();
  console.log(
    `${predictions.length} races | ${totalTickets} tickets | Avg EV: +${(avgEv * 100).toFixed(1)}% | Total: ¥${(totalTickets * 100).toLocaleString()} (@¥100)`,
  );
}
