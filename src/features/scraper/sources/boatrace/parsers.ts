/** HTML parsers for boatrace.jp pages */

import type {
  BeforeInfoData,
  BeforeInfoEntry,
  PayoutData,
  RaceData,
  RaceEntryData,
  RaceResultData,
  RaceResultEntry,
} from "@/features/database";
import { logger } from "@/shared/logger";
import * as cheerio from "cheerio";
import type { Cheerio, CheerioAPI } from "cheerio";
import type { AnyNode } from "domhandler";
import { type RaceParams, STADIUMS, STADIUM_PREFECTURES } from "./constants";
import {
  beforeInfoDataSchema,
  raceDataSchema,
  raceResultDataSchema,
} from "./schemas";

export interface RaceContext {
  params: RaceParams;
  raceDate: string; // YYYY-MM-DD
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/** Convert full-width digits to half-width: "１２３" → "123" */
function toHalfWidth(s: string): string {
  return s.replace(/[０-９]/g, (c) =>
    String.fromCharCode(c.charCodeAt(0) - 0xfee0),
  );
}

function parseFloat_(s: string): number | undefined {
  const v = Number.parseFloat(toHalfWidth(s));
  return Number.isNaN(v) ? undefined : v;
}

function parseInt_(s: string): number | undefined {
  const v = Number.parseInt(toHalfWidth(s), 10);
  return Number.isNaN(v) ? undefined : v;
}

function normalizeText(s: string): string {
  return s.replace(/[\s\u3000]+/g, " ").trim();
}

/** Extract numeric values from a multi-line cell (split by <br>). */
function parseMultilineCell(
  html: string,
  parsers: Array<(line: string) => number | undefined>,
): Array<number | undefined> {
  const lines = html.split("<br");
  return parsers.map((parse, i) =>
    parse((lines[i] ?? "").replace(/<[^>]*>/g, "")),
  );
}

function extractInt(s: string): number | undefined {
  return parseInt_(s.replace(/[^0-9]/g, ""));
}

function extractFloat(s: string): number | undefined {
  return parseFloat_(s.replace(/[^0-9.]/g, ""));
}

/**
 * Extract sections of HTML that contain any of the given markers.
 * For each marker, finds the nearest enclosing <table> or <div> and extracts it.
 * Each marker extracts only the first matching section (no duplicates).
 * Much smaller than full HTML → faster cheerio.load.
 */
function extractSections(html: string, markers: string[]): string {
  let result = "";
  const used = new Set<number>();

  for (const marker of markers) {
    const idx = html.indexOf(marker);
    if (idx === -1) continue;

    const tableStart = html.lastIndexOf("<table", idx);
    const divStart = html.lastIndexOf("<div", idx);
    const start = Math.max(tableStart, divStart);
    if (start === -1 || used.has(start)) continue;

    const tagName = start === tableStart ? "table" : "div";
    const openTag = `<${tagName}`;
    const closeTag = `</${tagName}`;
    let depth = 0;
    let pos = start;

    while (pos < html.length) {
      const openIdx = html.indexOf(openTag, pos + 1);
      const closeIdx = html.indexOf(closeTag, pos + 1);
      if (closeIdx === -1) break;
      if (openIdx !== -1 && openIdx < closeIdx) {
        depth++;
        pos = openIdx;
      } else {
        if (depth === 0) {
          const end = html.indexOf(">", closeIdx) + 1;
          result += html.slice(start, end);
          used.add(start);
          break;
        }
        depth--;
        pos = closeIdx;
      }
    }
  }

  return result;
}

/** Create a CheerioAPI from partial HTML extracted by markers. */
function loadPartial(html: string, markers: string[]): CheerioAPI {
  return cheerio.load(extractSections(html, markers));
}

// ---------------------------------------------------------------------------
// parseRaceList — 出走表ページ
// ---------------------------------------------------------------------------

const RACELIST_MARKERS = [
  "heading2_title",
  "is-tableFixed__3rdadd",
  "is-thColor8",
];

function parseRacerIdentity($: Cheerio<AnyNode>): {
  boatNumber: number;
  racerId: number;
  racerName: string;
  racerClass?: string;
  branch?: string;
  birthplace?: string;
  racerWeight?: number;
} | null {
  const firstRow = $.find("tr").first();

  const boatCell = firstRow.find("td.is-fs14[class*='is-boatColor']");
  const boatNumber = parseInt_(boatCell.text().trim());
  if (!boatNumber) return null;

  const profileLink = $.find("a[href*='toban=']").first();
  const tobanMatch = (profileLink.attr("href") ?? "").match(/toban=(\d+)/);
  const racerId = tobanMatch ? Number.parseInt(tobanMatch[1], 10) : 0;
  if (!racerId) return null;

  const racerName = normalizeText($.find(".is-fs18.is-fBold a").first().text());
  if (!racerName) return null;

  const regText = normalizeText($.find(".is-fs11").first().text());
  const classMatch = regText.match(/\/ ?(A1|A2|B1|B2)/);

  const infoText = normalizeText($.find(".is-fs11").eq(1).text());
  const branchMatch = infoText.match(/^(.+?)\/(.+?)\s/);
  const weightMatch = infoText.match(/([\d.]+)kg/);

  return {
    boatNumber,
    racerId,
    racerName,
    racerClass: classMatch?.[1],
    branch: branchMatch?.[1],
    birthplace: branchMatch?.[2],
    racerWeight: weightMatch ? parseFloat_(weightMatch[1]) : undefined,
  };
}

function parseRacerStats(
  firstRow: Cheerio<AnyNode>,
  $: CheerioAPI,
): Pick<
  RaceEntryData,
  | "flyingCount"
  | "lateCount"
  | "averageSt"
  | "nationalWinRate"
  | "nationalTop2Rate"
  | "nationalTop3Rate"
  | "localWinRate"
  | "localTop2Rate"
  | "localTop3Rate"
  | "motorNumber"
  | "motorTop2Rate"
  | "motorTop3Rate"
  | "boatNumberAssigned"
  | "boatTop2Rate"
  | "boatTop3Rate"
> {
  const cells = firstRow.find("td.is-lineH2");

  const [flyingCount, lateCount, averageSt] = parseMultilineCell(
    cells.eq(0).html() ?? "",
    [extractInt, extractInt, extractFloat],
  );

  const [nationalWinRate, nationalTop2Rate, nationalTop3Rate] =
    parseMultilineCell(cells.eq(1).html() ?? "", [
      extractFloat,
      extractFloat,
      extractFloat,
    ]);

  const [localWinRate, localTop2Rate, localTop3Rate] = parseMultilineCell(
    cells.eq(2).html() ?? "",
    [extractFloat, extractFloat, extractFloat],
  );

  const [motorNumber, motorTop2Rate, motorTop3Rate] = parseMultilineCell(
    cells.eq(3).html() ?? "",
    [extractInt, extractFloat, extractFloat],
  );

  const [boatNumberAssigned, boatTop2Rate, boatTop3Rate] = parseMultilineCell(
    cells.eq(4).html() ?? "",
    [extractInt, extractFloat, extractFloat],
  );

  return {
    flyingCount,
    lateCount,
    averageSt,
    nationalWinRate,
    nationalTop2Rate,
    nationalTop3Rate,
    localWinRate,
    localTop2Rate,
    localTop3Rate,
    motorNumber,
    motorTop2Rate,
    motorTop3Rate,
    boatNumberAssigned,
    boatTop2Rate,
    boatTop3Rate,
  };
}

export function parseRaceList(
  html: string,
  context: RaceContext,
): RaceData | null {
  const { params, raceDate } = context;
  const stadiumId = Number.parseInt(params.stadiumCode, 10);

  const $ = loadPartial(html, RACELIST_MARKERS);

  const raceTitle = normalizeText($(".heading2_titleName").text()) || undefined;
  const gradeClass = $(".heading2_title").attr("class") ?? "";
  const raceGrade = extractGrade(gradeClass);

  const entryTable = $("div.table1.is-tableFixed__3rdadd table");
  if (entryTable.length === 0) {
    logger.warn(`No entry table found for R${params.raceNumber}`);
    return null;
  }

  const entries: RaceEntryData[] = [];

  entryTable.find("tbody.is-fs12").each((_i, tbody) => {
    const $tbody = $(tbody);
    const identity = parseRacerIdentity($tbody);
    if (!identity) return;

    const firstRow = $tbody.find("tr").first();
    const stats = parseRacerStats(firstRow, $);

    entries.push({ ...identity, ...stats });
  });

  if (entries.length === 0) {
    logger.warn(`No entries parsed for R${params.raceNumber}`);
    return null;
  }

  // Parse deadline from 締切予定時刻 row (12 cells for races 1-12)
  let deadline: string | undefined;
  const deadlineRow = $("td.is-thColor8")
    .filter((_i, el) => $(el).text().includes("締切予定時刻"))
    .parent();
  if (deadlineRow.length > 0) {
    const cells = deadlineRow.find("td:not(.is-thColor8)");
    const idx = params.raceNumber - 1;
    if (idx < cells.length) {
      const text = $(cells[idx]).text().trim();
      if (/^\d{1,2}:\d{2}$/.test(text)) {
        deadline = text;
      }
    }
  }

  const data: RaceData = {
    stadiumId,
    stadiumName: STADIUMS[params.stadiumCode] ?? `場${params.stadiumCode}`,
    stadiumPrefecture: STADIUM_PREFECTURES[params.stadiumCode],
    raceDate,
    raceNumber: params.raceNumber,
    raceTitle,
    raceGrade,
    distance: 1800,
    deadline,
    entries,
  };

  const result = raceDataSchema.safeParse(data);
  if (!result.success) {
    logger.warn(
      `Validation failed for race list R${params.raceNumber}: ${result.error.message}`,
    );
    return null;
  }

  return data;
}

// ---------------------------------------------------------------------------
// parseBeforeInfo — 直前情報ページ
// ---------------------------------------------------------------------------

const BEFOREINFO_MARKERS = ["is-w748", "is-w238"];

function parseExhibitionSt($: CheerioAPI, entries: BeforeInfoEntry[]): void {
  $("table.is-w238 .table1_boatImage1").each((_i, div) => {
    const $div = $(div);
    const typeClass = $div.find(".table1_boatImage1Number").attr("class") ?? "";
    const typeMatch = typeClass.match(/is-type(\d)/);
    const boatNumber = typeMatch ? Number.parseInt(typeMatch[1], 10) : 0;

    const timeText = normalizeText($div.find(".table1_boatImage1Time").text());
    const stMatch = timeText.match(/F?\s*\.?(\d+\.?\d*)/);
    const exhibitionSt = stMatch ? parseFloat_(`.${stMatch[1]}`) : undefined;

    const entry = entries.find((e) => e.boatNumber === boatNumber);
    if (entry && exhibitionSt !== undefined) {
      entry.exhibitionSt = exhibitionSt;
    }
  });
}

function applyStabilizer(html: string, entries: BeforeInfoEntry[]): void {
  if (html.includes("安定板使用")) {
    for (const entry of entries) {
      entry.stabilizer = true;
    }
  }
}

export function parseBeforeInfo(
  html: string,
  context: RaceContext,
): BeforeInfoData | null {
  const { params, raceDate } = context;
  const stadiumId = Number.parseInt(params.stadiumCode, 10);

  const $ = loadPartial(html, BEFOREINFO_MARKERS);
  const entries: BeforeInfoEntry[] = [];

  $("table.is-w748 tbody.is-fs12").each((i, tbody) => {
    const $tbody = $(tbody);
    const firstRow = $tbody.find("tr").first();
    const tds = firstRow.find("td");

    const exhibitionTimeRaw = parseFloat_(normalizeText(tds.eq(4).text()));
    const exhibitionTime =
      exhibitionTimeRaw && exhibitionTimeRaw > 0
        ? exhibitionTimeRaw
        : undefined;
    const tilt = parseFloat_(normalizeText(tds.eq(5).text()));

    const partsReplaced: string[] = [];
    $tbody.find("ul.labelGroup1 span.label4").each((_j, span) => {
      const part = normalizeText($(span).text());
      if (part) partsReplaced.push(part);
    });

    entries.push({
      boatNumber: i + 1,
      exhibitionTime,
      tilt,
      partsReplaced: partsReplaced.length > 0 ? partsReplaced : undefined,
    });
  });

  parseExhibitionSt($, entries);
  applyStabilizer(html, entries);

  if (entries.length === 0) {
    logger.warn(`No before-info entries parsed for R${params.raceNumber}`);
    return null;
  }

  const data: BeforeInfoData = {
    stadiumId,
    raceDate,
    raceNumber: params.raceNumber,
    entries,
  };

  const result = beforeInfoDataSchema.safeParse(data);
  if (!result.success) {
    logger.warn(
      `Validation failed for before-info R${params.raceNumber}: ${result.error.message}`,
    );
    return null;
  }

  return data;
}

// ---------------------------------------------------------------------------
// parseRaceResult — レース結果ページ
// ---------------------------------------------------------------------------

// Each marker matches only the FIRST occurrence in the HTML (extractSections limitation).
// Use distinct markers for sections that appear multiple times (e.g., is-w495 has 4 tables).
const RACERESULT_MARKERS = [
  "is-w495",
  "is-h292__3rdadd",
  "決まり手",
  "weather1_body",
  "勝式",
];

function parseResultEntries($: CheerioAPI): RaceResultEntry[] {
  const resultTable = $("table.is-w495 th:contains('着')")
    .closest("table")
    .first();

  if (resultTable.length === 0) return [];

  const entries: RaceResultEntry[] = [];

  resultTable.find("tbody").each((_i, tbody) => {
    const row = $(tbody).find("tr").first();
    const tds = row.find("td");

    const posText = normalizeText(tds.eq(0).text());
    const finishPosition = parseInt_(toHalfWidth(posText));

    const boatNumber = parseInt_(normalizeText(tds.eq(1).text()));
    if (!boatNumber) return;

    const timeText = normalizeText(tds.eq(3).text());

    entries.push({
      boatNumber,
      finishPosition,
      raceTime: timeText || undefined,
    });
  });

  return entries;
}

function parseStartInfo($: CheerioAPI, entries: RaceResultEntry[]): void {
  $(".table1_boatImage1.is-type1__3rdadd").each((_i, div) => {
    const $div = $(div);
    const typeClass = $div.find(".table1_boatImage1Number").attr("class") ?? "";
    const typeMatch = typeClass.match(/is-type(\d)/);
    const boatNumber = typeMatch ? Number.parseInt(typeMatch[1], 10) : 0;

    const courseNumber = _i + 1;

    const timeInnerText = normalizeText(
      $div.find(".table1_boatImage1TimeInner").text(),
    );
    const stMatch = timeInnerText.match(/\.(\d+)/);
    const startTiming = stMatch ? parseFloat_(`.${stMatch[1]}`) : undefined;

    const entry = entries.find((e) => e.boatNumber === boatNumber);
    if (entry) {
      entry.courseNumber = courseNumber;
      entry.startTiming = startTiming;
    }
  });
}

export function parseRaceResult(
  html: string,
  context: RaceContext,
): RaceResultData | null {
  const { params, raceDate } = context;
  const stadiumId = Number.parseInt(params.stadiumCode, 10);

  const $ = loadPartial(html, RACERESULT_MARKERS);

  const resultEntries = parseResultEntries($);
  if (resultEntries.length === 0) {
    logger.warn(`No result table found for R${params.raceNumber}`);
    return null;
  }

  parseStartInfo($, resultEntries);

  const techniqueTable = $("th:contains('決まり手')").closest("table");
  const technique =
    normalizeText(techniqueTable.find("tbody td").first().text()) || undefined;

  const weather = parseWeather($);
  const payouts = parsePayouts($);

  const data: RaceResultData = {
    stadiumId,
    raceDate,
    raceNumber: params.raceNumber,
    weather: weather.weatherText,
    windSpeed: weather.windSpeed,
    windDirection: weather.windDirection,
    waveHeight: weather.waveHeight,
    temperature: weather.temperature,
    waterTemperature: weather.waterTemperature,
    technique,
    entries: resultEntries,
    payouts: payouts.length > 0 ? payouts : undefined,
  };

  const result = raceResultDataSchema.safeParse(data);
  if (!result.success) {
    logger.warn(
      `Validation failed for race result R${params.raceNumber}: ${result.error.message}`,
    );
    return null;
  }

  return data;
}

// ---------------------------------------------------------------------------
// Weather parsing (shared between beforeinfo and result pages)
// ---------------------------------------------------------------------------

interface WeatherInfo {
  weatherText?: string;
  windSpeed?: number;
  windDirection?: number;
  waveHeight?: number;
  temperature?: number;
  waterTemperature?: number;
}

export function parseWeather($: CheerioAPI): WeatherInfo {
  const weather: WeatherInfo = {};

  const weatherUnit = $(".weather1_bodyUnit.is-weather");
  weather.weatherText =
    normalizeText(weatherUnit.find(".weather1_bodyUnitLabelTitle").text()) ||
    undefined;

  const tempText = $(
    ".weather1_bodyUnit.is-direction .weather1_bodyUnitLabelData",
  )
    .text()
    .trim();
  const tempMatch = tempText.match(/([\d.]+)/);
  weather.temperature = tempMatch ? parseFloat_(tempMatch[1]) : undefined;

  const windText = $(".weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData")
    .text()
    .trim();
  const windMatch = windText.match(/(\d+)/);
  weather.windSpeed = windMatch ? parseInt_(windMatch[1]) : undefined;

  const windDirEl = $(
    ".weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage",
  );
  const windDirClass = windDirEl.attr("class") ?? "";
  const windDirMatch = windDirClass.match(/is-wind(\d+)/);
  weather.windDirection = windDirMatch ? parseInt_(windDirMatch[1]) : undefined;

  const waterTempText = $(
    ".weather1_bodyUnit.is-waterTemperature .weather1_bodyUnitLabelData",
  )
    .text()
    .trim();
  const waterTempMatch = waterTempText.match(/([\d.]+)/);
  weather.waterTemperature = waterTempMatch
    ? parseFloat_(waterTempMatch[1])
    : undefined;

  const waveText = $(".weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData")
    .text()
    .trim();
  const waveMatch = waveText.match(/(\d+)/);
  weather.waveHeight = waveMatch ? parseInt_(waveMatch[1]) : undefined;

  return weather;
}

// ---------------------------------------------------------------------------
// Payouts parsing
// ---------------------------------------------------------------------------

function parsePayouts($: CheerioAPI): PayoutData[] {
  const payouts: PayoutData[] = [];

  const payoutTable = $("th:contains('勝式')").closest("table");
  if (payoutTable.length === 0) return payouts;

  payoutTable.find("tbody").each((_i, tbody) => {
    const $tbody = $(tbody);
    const rows = $tbody.find("tr.is-p3-0");

    const betTypeEl = rows.first().find("td[rowspan]").first();
    const betType = normalizeText(betTypeEl.text());
    if (!betType) return;

    rows.each((_j, row) => {
      const $row = $(row);

      const numbers: string[] = [];
      $row.find(".numberSet1_number").each((_k, num) => {
        numbers.push(normalizeText($(num).text()));
      });
      if (numbers.length === 0) return;

      const separators: string[] = [];
      $row.find(".numberSet1_text").each((_k, sep) => {
        separators.push(normalizeText($(sep).text()));
      });

      let combination = "";
      for (let k = 0; k < numbers.length; k++) {
        if (k > 0 && separators[k - 1]) {
          combination += separators[k - 1];
        }
        combination += numbers[k];
      }

      const payoutText = normalizeText($row.find(".is-payout1").text());
      const payoutMatch = payoutText.match(/[\d,]+/);
      if (!payoutMatch) return;
      const payout = Number.parseInt(payoutMatch[0].replace(/,/g, ""), 10);
      if (Number.isNaN(payout) || payout === 0) return;

      payouts.push({ betType, combination, payout });
    });
  });

  return payouts;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function extractGrade(classStr: string): string | undefined {
  const lower = classStr.toLowerCase();
  if (lower.includes("is-sg")) return "SG";
  if (lower.includes("is-g1")) return "G1";
  if (lower.includes("is-g2")) return "G2";
  if (lower.includes("is-g3")) return "G3";
  if (lower.includes("is-ippan")) return "一般";
  return undefined;
}
