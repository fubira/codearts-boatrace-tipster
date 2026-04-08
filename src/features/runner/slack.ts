/** Slack notification via Incoming Webhook (Block Kit). */

import { logger } from "@/shared/logger";

interface SlackBlock {
  type: "header" | "section" | "divider";
  text?: { type: "plain_text" | "mrkdwn"; text: string };
  fields?: { type: "mrkdwn"; text: string }[];
}

interface SlackPayload {
  text?: string;
  blocks: SlackBlock[];
}

let webhookUrl: string | undefined;

export function setSlackWebhook(url: string | undefined): void {
  webhookUrl = url;
}

async function send(payload: SlackPayload): Promise<void> {
  if (!webhookUrl) {
    // No webhook configured — log to console instead
    const text =
      payload.text ??
      payload.blocks
        .filter((b) => b.text)
        .map((b) => b.text?.text)
        .join(" | ");
    logger.info(`[Slack] ${text}`);
    return;
  }

  const res = await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const body = await res.text();
    const preview = payload.text ?? "unknown";
    logger.error(`Slack send failed (${res.status}): ${body} [${preview}]`);
  }
}

// ---------------------------------------------------------------------------
// Notification helpers
// ---------------------------------------------------------------------------

export interface StartupInfo {
  version: string;
  date: string;
  venues: number;
  races: number;
  dryRun: boolean;
  evThreshold: number;
}

export async function notifyStartup(info: StartupInfo): Promise<void> {
  const mode = info.dryRun ? "DRY RUN" : "LIVE";
  await send({
    text: `[boatrace] ${mode} started: ${info.date} (v${info.version})`,
    blocks: [
      {
        type: "header",
        text: {
          type: "plain_text",
          text: `🏁 boatrace-tipster v${info.version} ${mode}`,
        },
      },
      {
        type: "section",
        fields: [
          { type: "mrkdwn", text: `*Date:* ${info.date}` },
          { type: "mrkdwn", text: `*Venues:* ${info.venues}` },
          { type: "mrkdwn", text: `*Races:* ${info.races}` },
          {
            type: "mrkdwn",
            text: `*EV≥:* +${(info.evThreshold * 100).toFixed(0)}%`,
          },
        ],
      },
    ],
  });
}

export interface PredictionInfo {
  stadiumName: string;
  raceNumber: number;
  deadline: string;
  prob: number;
  odds: number;
  ev: number;
  betAmount: number;
}

export async function notifyPrediction(p: PredictionInfo): Promise<void> {
  await send({
    text: `[boatrace] BET: ${p.stadiumName}${p.raceNumber}R EV+${p.ev.toFixed(1)}%`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text:
            `*${p.stadiumName} ${p.raceNumber}R* (締切 ${p.deadline})\n` +
            `EV *+${p.ev.toFixed(1)}%* | prob *${(p.prob * 100).toFixed(1)}%*\n` +
            `→ 3連単 *¥${p.betAmount.toLocaleString()}*`,
        },
      },
    ],
  });
}

export interface RaceResultInfo {
  stadiumName: string;
  raceNumber: number;
  won: boolean;
  betAmount: number;
  payout: number;
  bankroll: number;
}

export async function notifyResult(r: RaceResultInfo): Promise<void> {
  const icon = r.won ? "✅" : "❌";
  const pl = r.payout - r.betAmount;
  const plStr =
    pl >= 0 ? `+¥${pl.toLocaleString()}` : `-¥${Math.abs(pl).toLocaleString()}`;
  await send({
    text: `[boatrace] ${icon} ${r.stadiumName}${r.raceNumber}R ${plStr}`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `${icon} *${r.stadiumName} ${r.raceNumber}R* ${plStr} (残高 ¥${r.bankroll.toLocaleString()})`,
        },
      },
    ],
  });
}

export interface DailySummaryInfo {
  date: string;
  totalBets: number;
  wins: number;
  totalWagered: number;
  totalPayout: number;
  bankroll: number;
}

export async function notifyDailySummary(s: DailySummaryInfo): Promise<void> {
  const pl = s.totalPayout - s.totalWagered;
  const plStr =
    pl >= 0 ? `+¥${pl.toLocaleString()}` : `-¥${Math.abs(pl).toLocaleString()}`;
  const roi =
    s.totalWagered > 0
      ? ((s.totalPayout / s.totalWagered) * 100).toFixed(1)
      : "N/A";
  await send({
    text: `[boatrace] Daily: ${plStr} ROI ${roi}%`,
    blocks: [
      {
        type: "header",
        text: { type: "plain_text", text: `📊 Daily Summary ${s.date}` },
      },
      {
        type: "section",
        fields: [
          { type: "mrkdwn", text: `*Bets:* ${s.totalBets}` },
          { type: "mrkdwn", text: `*Wins:* ${s.wins}/${s.totalBets}` },
          { type: "mrkdwn", text: `*P&L:* ${plStr}` },
          { type: "mrkdwn", text: `*ROI:* ${roi}%` },
          {
            type: "mrkdwn",
            text: `*Wagered:* ¥${s.totalWagered.toLocaleString()}`,
          },
          {
            type: "mrkdwn",
            text: `*Bankroll:* ¥${s.bankroll.toLocaleString()}`,
          },
        ],
      },
    ],
  });
}

export async function notifyError(
  context: string,
  error: unknown,
): Promise<void> {
  const msg = error instanceof Error ? error.message : String(error);
  await send({
    text: `[boatrace] ERROR: ${context}: ${msg}`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `⚠️ *Error:* ${context}\n\`\`\`${msg}\`\`\``,
        },
      },
    ],
  });
}

export async function notifyShutdown(): Promise<void> {
  await send({
    text: "[boatrace] Shutdown",
    blocks: [
      {
        type: "section",
        text: { type: "mrkdwn", text: "🛑 *boatrace-tipster stopped*" },
      },
    ],
  });
}
