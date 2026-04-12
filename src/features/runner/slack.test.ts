import { afterEach, describe, expect, mock, test } from "bun:test";
import {
  type DailySummaryInfo,
  type PredictionInfo,
  type RaceResultInfo,
  type StartupInfo,
  notifyDailySummary,
  notifyError,
  notifyPrediction,
  notifyResult,
  notifyShutdown,
  notifyStartup,
  setSlackWebhook,
} from "./slack";

// Capture payloads sent to fetch
const fetchCalls: { url: string; body: unknown }[] = [];

const originalFetch = globalThis.fetch;

function installFetchMock() {
  globalThis.fetch = mock(async (url: string | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: typeof url === "string" ? url : url.url,
      body: JSON.parse(init?.body as string),
    });
    return new Response("ok", { status: 200 });
  }) as typeof fetch;
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  fetchCalls.length = 0;
  setSlackWebhook(undefined);
});

describe("notifyStartup", () => {
  const info: StartupInfo = {
    version: "0.14.1",
    date: "2026-04-07",
    venues: 5,
    races: 60,
    dryRun: true,
    evThreshold: 0.36,
  };

  test("sends payload with correct fields when webhook is set", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyStartup(info);

    expect(fetchCalls).toHaveLength(1);
    const body = fetchCalls[0].body as { text: string; blocks: unknown[] };
    expect(body.text).toContain("DRY RUN");
    expect(body.text).toContain("2026-04-07");
    expect(body.blocks).toHaveLength(2);
  });

  test("LIVE mode label when dryRun is false", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyStartup({ ...info, dryRun: false });

    const body = fetchCalls[0].body as { text: string };
    expect(body.text).toContain("LIVE");
    expect(body.text).not.toContain("DRY RUN");
  });

  test("does not call fetch when webhook is not set", async () => {
    installFetchMock();

    await notifyStartup(info);

    expect(fetchCalls).toHaveLength(0);
  });

  test("EV threshold formatted as percentage", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyStartup(info);

    const body = fetchCalls[0].body as {
      blocks: { fields?: { text: string }[] }[];
    };
    const evField = body.blocks[1].fields?.find((f) => f.text.includes("EV"));
    expect(evField?.text).toContain("+36%");
  });
});

describe("notifyPrediction", () => {
  const pred: PredictionInfo = {
    stadiumName: "平和島",
    raceNumber: 5,
    deadline: "14:30",
    top3Conc: 0.72,
    gap23: 0.18,
    tickets: [
      { combo: "1-3-2", modelProb: 0.117, marketOdds: 14.6, ev: 0.28 },
      { combo: "1-2-3", modelProb: 0.061, marketOdds: 25.5, ev: 0.17 },
    ],
    unit: 500,
    betAmount: 1000,
  };

  test("formats prediction payload", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyPrediction(pred);

    expect(fetchCalls).toHaveLength(1);
    const body = fetchCalls[0].body as {
      text: string;
      blocks: { text?: { text: string } }[];
    };
    expect(body.text).toContain("平和島");
    expect(body.text).toContain("5R");
    expect(body.text).toContain("¥1,000");
    const block = body.blocks[0].text?.text ?? "";
    expect(block).toContain("1-3-2");
    expect(block).toContain("14.6倍");
    expect(block).toContain("conc");
    expect(block).toContain("gap23");
    expect(block).toContain("¥500");
  });
});

describe("notifyResult", () => {
  test("win result shows positive P&L with ticket info", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    const result: RaceResultInfo = {
      stadiumName: "住之江",
      raceNumber: 8,
      won: true,
      betAmount: 500,
      payout: 15000,
      bankroll: 84500,
      tickets: [
        { combo: "1-3-2", marketOdds: 14.6, ev: 0.28 },
        { combo: "1-2-3", marketOdds: 25.5, ev: 0.17 },
      ],
      resultCombo: "1-3-2",
      officialPayoutPer100: 3000,
    };
    await notifyResult(result);

    const body = fetchCalls[0].body as {
      text: string;
      blocks: { text?: { text: string } }[];
    };
    expect(body.text).toContain("✅");
    expect(body.text).toContain("+¥14,500");
    const block = body.blocks[0].text?.text ?? "";
    expect(block).toContain("1-3-2");
    expect(block).toContain("⭕");
    expect(block).toContain("30.0倍");
    expect(block).toContain("+¥7,250");
  });

  test("loss result shows actual combo and no hit", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    const result: RaceResultInfo = {
      stadiumName: "大村",
      raceNumber: 3,
      won: false,
      betAmount: 500,
      payout: 0,
      bankroll: 69500,
      tickets: [{ combo: "1-2-3", marketOdds: 10.0, ev: 0.1 }],
      resultCombo: "2-1-6",
      officialPayoutPer100: 0,
    };
    await notifyResult(result);

    const body = fetchCalls[0].body as {
      text: string;
      blocks: { text?: { text: string } }[];
    };
    expect(body.text).toContain("❌");
    expect(body.text).toContain("-¥500");
    const block = body.blocks[0].text?.text ?? "";
    expect(block).toContain("1-2-3");
    expect(block).toContain("2-1-6");
    expect(block).not.toContain("⭕");
  });
});

describe("notifyDailySummary", () => {
  const baseSkipCounts = {
    not_b1_top: 0,
    top3_conc_low: 0,
    gap23_low: 0,
    no_ev_tickets: 0,
    drift_drop: 0,
  };

  test("computes ROI and shows skip/drift breakdown", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    const summary: DailySummaryInfo = {
      date: "2026-04-07",
      totalRaces: 108,
      totalBets: 5,
      totalTickets: 7,
      wins: 2,
      totalWagered: 2500,
      totalPayout: 6250,
      bankroll: 73750,
      allTimeInitial: 70000,
      startedAt: "2026-04-07T07:00:00+09:00",
      skipCounts: {
        not_b1_top: 12,
        top3_conc_low: 30,
        gap23_low: 45,
        no_ev_tickets: 16,
        drift_drop: 0,
      },
      t1DroppedTickets: 3,
    };
    await notifyDailySummary(summary);

    const body = fetchCalls[0].body as {
      text: string;
      blocks: { text?: { text: string } }[];
    };
    // ROI = 6250/2500 * 100 = 250.0%
    expect(body.text).toContain("250.0%");
    expect(body.text).toContain("5R/108R");
    const block = body.blocks[1].text?.text ?? "";
    expect(block).toContain("+¥3,750");
    expect(block).toContain("40.0%"); // hit 2/5
    expect(block).toContain("not_B1=12");
    expect(block).toContain("conc=30");
    expect(block).toContain("gap23=45");
    expect(block).toContain("drift=0");
    expect(block).toContain("SKIP Ticket");
    expect(block).toContain("T-1 drop=3");
    expect(block).toContain("1.40 T/R"); // 7/5
    expect(block).toContain("7点");
    expect(block).toContain("合計 ¥2,500");
  });

  test("handles zero wagers (no bets)", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    const summary: DailySummaryInfo = {
      date: "2026-04-07",
      totalRaces: 108,
      totalBets: 0,
      totalTickets: 0,
      wins: 0,
      totalWagered: 0,
      totalPayout: 0,
      bankroll: 70000,
      allTimeInitial: 70000,
      startedAt: "2026-04-07T07:00:00+09:00",
      skipCounts: baseSkipCounts,
      t1DroppedTickets: 0,
    };
    await notifyDailySummary(summary);

    const body = fetchCalls[0].body as { text: string };
    expect(body.text).toContain("N/A");
  });
});

describe("notifyError", () => {
  test("formats Error instance", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyError("scrape failed", new Error("connection timeout"));

    const body = fetchCalls[0].body as { text: string };
    expect(body.text).toContain("scrape failed");
    expect(body.text).toContain("connection timeout");
  });

  test("formats string error", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyError("unknown", "something broke");

    const body = fetchCalls[0].body as { text: string };
    expect(body.text).toContain("something broke");
  });
});

describe("notifyShutdown", () => {
  test("sends shutdown message", async () => {
    setSlackWebhook("https://hooks.slack.com/test");
    installFetchMock();

    await notifyShutdown();

    expect(fetchCalls).toHaveLength(1);
    const body = fetchCalls[0].body as { text: string };
    expect(body.text).toContain("Shutdown");
  });
});
