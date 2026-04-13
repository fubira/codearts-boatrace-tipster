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

export interface P2TicketInfo {
  combo: string;
  modelProb: number;
  marketOdds: number;
  ev: number;
}

export interface PredictionInfo {
  stadiumName: string;
  raceNumber: number;
  deadline: string;
  top3Conc: number;
  gap23: number;
  tickets: P2TicketInfo[];
  unit: number;
  betAmount: number;
}

export async function notifyPrediction(p: PredictionInfo): Promise<void> {
  const ticketLines = p.tickets
    .map(
      (t) =>
        `• \`${t.combo}\` @ ${t.marketOdds.toFixed(1)}倍 | EV *+${(t.ev * 100).toFixed(0)}%* (prob ${(t.modelProb * 100).toFixed(2)}%)`,
    )
    .join("\n");
  const avgEv = p.tickets.reduce((s, t) => s + t.ev, 0) / p.tickets.length;

  await send({
    text: `[boatrace] BET: ${p.stadiumName}${p.raceNumber}R ${p.tickets.length}点 ¥${p.betAmount.toLocaleString()}`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text:
            `*${p.stadiumName} ${p.raceNumber}R* (締切 ${p.deadline})\n` +
            `conc *${(p.top3Conc * 100).toFixed(0)}%* / gap23 *${(p.gap23 * 100).toFixed(1)}%* / avg EV *+${(avgEv * 100).toFixed(0)}%*\n` +
            `${ticketLines}\n` +
            `→ ¥${p.unit.toLocaleString()} × ${p.tickets.length}点 = *¥${p.betAmount.toLocaleString()}*`,
        },
      },
    ],
  });
}

export interface ResultTicketInfo {
  combo: string;
  marketOdds: number; // T-1 odds used for bet decision
  ev: number;
}

export interface RaceResultInfo {
  stadiumName: string;
  raceNumber: number;
  won: boolean;
  betAmount: number;
  payout: number;
  bankroll: number;
  tickets: ResultTicketInfo[];
  resultCombo: string | null; // Actual 1-2-3 combo
  officialPayoutPer100: number; // Official 3連単 payout (per 100 yen bet)
}

export async function notifyResult(r: RaceResultInfo): Promise<void> {
  const icon = r.won ? "✅" : "❌";
  const pl = r.payout - r.betAmount;
  const plStr =
    pl >= 0 ? `+¥${pl.toLocaleString()}` : `-¥${Math.abs(pl).toLocaleString()}`;

  // Unit per ticket (uniform)
  const unit = r.tickets.length > 0 ? r.betAmount / r.tickets.length : 0;

  const ticketLines = r.tickets
    .map((t) => {
      const isHit = t.combo === r.resultCombo && r.officialPayoutPer100 > 0;
      if (isHit) {
        const multiplier = r.officialPayoutPer100 / 100;
        const ticketPayout = (unit / 100) * r.officialPayoutPer100;
        const ticketProfit = ticketPayout - unit;
        return (
          `• ⭕ \`${t.combo}\` 確定 *${multiplier.toFixed(1)}倍* × ¥${unit.toLocaleString()} = ` +
          `*¥${ticketPayout.toLocaleString()}* (+¥${ticketProfit.toLocaleString()})`
        );
      }
      return `• \`${t.combo}\` @ T-1 ${t.marketOdds.toFixed(1)}倍 (EV +${(t.ev * 100).toFixed(0)}%) → -¥${unit.toLocaleString()}`;
    })
    .join("\n");

  const resultLine = r.resultCombo ? `結果: *${r.resultCombo}*` : "結果: N/A";

  await send({
    text: `[boatrace] ${icon} ${r.stadiumName}${r.raceNumber}R ${plStr}`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text:
            `${icon} *${r.stadiumName} ${r.raceNumber}R* ${plStr}\n` +
            `${resultLine}\n` +
            `${ticketLines}\n` +
            `購入 ¥${r.betAmount.toLocaleString()} / 払戻 ¥${r.payout.toLocaleString()} / 残高 ¥${r.bankroll.toLocaleString()}`,
        },
      },
    ],
  });
}

export interface DailySummaryInfo {
  date: string;
  totalRaces: number;
  totalBets: number;
  totalTickets: number;
  wins: number;
  totalWagered: number;
  totalPayout: number;
  bankroll: number; // current
  allTimeInitial: number; // bankroll when runner was first started
  startedAt: string; // ISO of first start
  skipCounts: {
    not_b1_top: number;
    gap12_low: number;
    top3_conc_low: number;
    gap23_low: number;
    no_ev_tickets: number;
    drift_drop: number;
  };
  t1DroppedTickets: number;
}

export async function notifyDailySummary(s: DailySummaryInfo): Promise<void> {
  const pl = s.totalPayout - s.totalWagered;
  const plStr =
    pl >= 0 ? `+¥${pl.toLocaleString()}` : `-¥${Math.abs(pl).toLocaleString()}`;
  const roi =
    s.totalWagered > 0
      ? `${((s.totalPayout / s.totalWagered) * 100).toFixed(1)}%`
      : "N/A";
  const hitRate =
    s.totalBets > 0 ? `${((s.wins / s.totalBets) * 100).toFixed(1)}%` : "N/A";
  const allTimeDelta = s.bankroll - s.allTimeInitial;
  const allTimeDeltaStr =
    allTimeDelta >= 0
      ? `+¥${allTimeDelta.toLocaleString()}`
      : `-¥${Math.abs(allTimeDelta).toLocaleString()}`;
  const allTimeRoiPct =
    s.allTimeInitial > 0 ? (s.bankroll / s.allTimeInitial) * 100 - 100 : null;
  const allTimeRoi =
    allTimeRoiPct == null
      ? "N/A"
      : `${allTimeRoiPct >= 0 ? "+" : ""}${allTimeRoiPct.toFixed(1)}%`;

  // SKIP 内訳
  const sc = s.skipCounts;
  const totalSkip =
    sc.not_b1_top +
    sc.gap12_low +
    sc.top3_conc_low +
    sc.gap23_low +
    sc.no_ev_tickets +
    sc.drift_drop;
  const skipDetail = [
    `not_B1=${sc.not_b1_top}`,
    `gap12=${sc.gap12_low}`,
    `conc=${sc.top3_conc_low}`,
    `gap23=${sc.gap23_low}`,
    `no_EV=${sc.no_ev_tickets}`,
    `drift=${sc.drift_drop}`,
  ].join(" / ");

  const tpr =
    s.totalBets > 0 ? (s.totalTickets / s.totalBets).toFixed(2) : "N/A";

  await send({
    text: `[boatrace] Daily: ${plStr} ROI ${roi} ${s.totalBets}R/${s.totalRaces}R hit ${hitRate}`,
    blocks: [
      {
        type: "header",
        text: { type: "plain_text", text: `📊 Daily Summary ${s.date}` },
      },
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text:
            `*購入:* ${s.totalBets}R / ${s.totalRaces}R (${s.totalTickets}点, ${tpr} T/R) 合計 ¥${s.totalWagered.toLocaleString()}\n` +
            `*的中:* ${s.wins}/${s.totalBets} = *${hitRate}*\n` +
            `*本日 ROI:* ${roi} | *P/L:* ${plStr}\n` +
            `*Bankroll:* ¥${s.bankroll.toLocaleString()} (${allTimeRoi}) / 初期 ¥${s.allTimeInitial.toLocaleString()} (since ${s.startedAt.slice(0, 10)})\n` +
            `*累計:* ${allTimeDeltaStr}\n` +
            `*SKIP Race:* ${totalSkip}R (${skipDetail})\n` +
            `*SKIP Ticket:* T-1 drop=${s.t1DroppedTickets}`,
        },
      },
    ],
  });
}

export async function notifyError(
  context: string,
  error: unknown,
): Promise<void> {
  const raw = error instanceof Error ? error.message : String(error);
  // Truncate and sanitize for Block Kit (backticks/special chars can break blocks)
  const msg = raw.slice(0, 500).replace(/[`*~]/g, " ");
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
