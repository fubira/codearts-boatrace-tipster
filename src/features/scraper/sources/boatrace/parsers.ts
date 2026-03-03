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
import type { CheerioAPI } from "cheerio";
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

// ---------------------------------------------------------------------------
// parseRaceList — 出走表ページ
// ---------------------------------------------------------------------------

export function parseRaceList(
  $: CheerioAPI,
  context: RaceContext,
): RaceData | null {
  const { params, raceDate } = context;
  const stadiumId = Number.parseInt(params.stadiumCode, 10);

  // Race title from heading
  const raceTitle = normalizeText($(".heading2_titleName").text()) || undefined;

  // Race grade from heading2_title class
  const gradeClass = $(".heading2_title").attr("class") ?? "";
  const raceGrade = extractGrade(gradeClass);

  // Entry table: div.table1.is-tableFixed__3rdadd
  const entryTable = $("div.table1.is-tableFixed__3rdadd table");
  if (entryTable.length === 0) {
    logger.warn(`No entry table found for R${params.raceNumber}`);
    return null;
  }

  const entries: RaceEntryData[] = [];

  entryTable.find("tbody.is-fs12").each((_i, tbody) => {
    const $tbody = $(tbody);
    const firstRow = $tbody.find("tr").first();

    // Boat number from is-boatColor + is-fs14 cell (main boat number, not past results)
    const boatCell = firstRow.find("td.is-fs14[class*='is-boatColor']");
    const boatNumber = parseInt_(boatCell.text().trim());
    if (!boatNumber) return;

    // Racer ID from profile link
    const profileLink = $tbody.find("a[href*='toban=']").first();
    const tobanMatch = (profileLink.attr("href") ?? "").match(/toban=(\d+)/);
    const racerId = tobanMatch ? Number.parseInt(tobanMatch[1], 10) : 0;
    if (!racerId) return;

    // Racer name
    const racerName = normalizeText(
      $tbody.find(".is-fs18.is-fBold a").first().text(),
    );
    if (!racerName) return;

    // Racer class and registration info: "3470 / B1"
    const regDiv = $tbody.find(".is-fs11").first();
    const regText = normalizeText(regDiv.text());
    const classMatch = regText.match(/\/ ?(A1|A2|B1|B2)/);
    const racerClass = classMatch?.[1];

    // Branch / birthplace and age/weight
    const infoDiv = $tbody.find(".is-fs11").eq(1);
    const infoText = normalizeText(infoDiv.text());
    // "徳島/徳島 56歳/46.5kg"
    const branchMatch = infoText.match(/^(.+?)\/(.+?)\s/);
    const branch = branchMatch?.[1];
    const birthplace = branchMatch?.[2];
    const weightMatch = infoText.match(/([\d.]+)kg/);
    const racerWeight = weightMatch ? parseFloat_(weightMatch[1]) : undefined;

    // F count, L count, average ST: in the rowspan=4 cell
    const statsCell = firstRow.find("td.is-lineH2").eq(0);
    const statsLines = (statsCell.html() ?? "").split("<br");
    const flyingCount = parseInt_((statsLines[0] ?? "").replace(/[^0-9]/g, ""));
    const lateCount = parseInt_((statsLines[1] ?? "").replace(/[^0-9]/g, ""));
    const averageSt = parseFloat_(
      (statsLines[2] ?? "").replace(/[^0-9.]/g, ""),
    );

    // National stats: win rate, 2-rate, 3-rate
    const nationalCell = firstRow.find("td.is-lineH2").eq(1);
    const nationalLines = (nationalCell.html() ?? "").split("<br");
    const nationalWinRate = parseFloat_(
      (nationalLines[0] ?? "").replace(/[^0-9.]/g, ""),
    );
    const nationalTop2Rate = parseFloat_(
      (nationalLines[1] ?? "").replace(/[^0-9.]/g, ""),
    );
    const nationalTop3Rate = parseFloat_(
      (nationalLines[2] ?? "").replace(/[^0-9.]/g, ""),
    );

    // Local stats
    const localCell = firstRow.find("td.is-lineH2").eq(2);
    const localLines = (localCell.html() ?? "").split("<br");
    const localWinRate = parseFloat_(
      (localLines[0] ?? "").replace(/[^0-9.]/g, ""),
    );
    const localTop2Rate = parseFloat_(
      (localLines[1] ?? "").replace(/[^0-9.]/g, ""),
    );
    const localTop3Rate = parseFloat_(
      (localLines[2] ?? "").replace(/[^0-9.]/g, ""),
    );

    // Motor stats
    const motorCell = firstRow.find("td.is-lineH2").eq(3);
    const motorLines = (motorCell.html() ?? "").split("<br");
    const motorNumber = parseInt_((motorLines[0] ?? "").replace(/[^0-9]/g, ""));
    const motorTop2Rate = parseFloat_(
      (motorLines[1] ?? "").replace(/[^0-9.]/g, ""),
    );
    const motorTop3Rate = parseFloat_(
      (motorLines[2] ?? "").replace(/[^0-9.]/g, ""),
    );

    // Boat stats
    const boatCell2 = firstRow.find("td.is-lineH2").eq(4);
    const boatLines = (boatCell2.html() ?? "").split("<br");
    const boatNumberAssigned = parseInt_(
      (boatLines[0] ?? "").replace(/[^0-9]/g, ""),
    );
    const boatTop2Rate = parseFloat_(
      (boatLines[1] ?? "").replace(/[^0-9.]/g, ""),
    );
    const boatTop3Rate = parseFloat_(
      (boatLines[2] ?? "").replace(/[^0-9.]/g, ""),
    );

    entries.push({
      racerId,
      boatNumber,
      racerName,
      racerClass,
      racerWeight,
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
      branch,
      birthplace,
    });
  });

  if (entries.length === 0) {
    logger.warn(`No entries parsed for R${params.raceNumber}`);
    return null;
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

export function parseBeforeInfo(
  $: CheerioAPI,
  context: RaceContext,
): BeforeInfoData | null {
  const { params, raceDate } = context;
  const stadiumId = Number.parseInt(params.stadiumCode, 10);
  const entries: BeforeInfoEntry[] = [];

  // Main table: table.is-w748 > tbody.is-fs12
  $("table.is-w748 tbody.is-fs12").each((i, tbody) => {
    const $tbody = $(tbody);
    const boatNumber = i + 1;
    const firstRow = $tbody.find("tr").first();
    const tds = firstRow.find("td");

    // Column layout (from thead):
    // 0: 枠 (boat number, rowspan=4)
    // 1: 写真 (photo, rowspan=4)
    // 2: ボートレーサー (name, rowspan=4)
    // 3: 体重 (weight, rowspan=2)
    // 4: 展示タイム (rowspan=4)
    // 5: チルト (rowspan=4)
    // 6: プロペラ (rowspan=4)
    // 7: 部品交換 (rowspan=4)
    // 8: 前走成績R
    // 9: 前走成績値

    // Exhibition time (index 4 in first row)
    const exhibitionTimeText = normalizeText(tds.eq(4).text());
    const exhibitionTime = parseFloat_(exhibitionTimeText);

    // Tilt (index 5)
    const tiltText = normalizeText(tds.eq(5).text());
    const tilt = parseFloat_(tiltText);

    // Parts replaced (in labelGroup1 ul)
    const partsReplaced: string[] = [];
    $tbody.find("ul.labelGroup1 span.label4").each((_j, span) => {
      const part = normalizeText($(span).text());
      if (part) partsReplaced.push(part);
    });

    // Weight adjustment (index 3, second row)
    // The third row has the "調整重量" value
    const thirdRow = $tbody.find("tr").eq(2);
    const adjustWeightTd = thirdRow.find("td").first();
    // This contains the adjustment weight value (e.g. "0.5")

    entries.push({
      boatNumber,
      exhibitionTime,
      tilt,
      partsReplaced: partsReplaced.length > 0 ? partsReplaced : undefined,
    });
  });

  // Exhibition ST from table.is-w238
  $("table.is-w238 .table1_boatImage1").each((_i, div) => {
    const $div = $(div);
    const numberEl = $div.find(".table1_boatImage1Number");
    const typeClass = numberEl.attr("class") ?? "";
    const typeMatch = typeClass.match(/is-type(\d)/);
    const boatNumber = typeMatch ? Number.parseInt(typeMatch[1], 10) : 0;

    const timeText = normalizeText($div.find(".table1_boatImage1Time").text());
    // timeText: ".01" or "F.04" (flying)
    const stMatch = timeText.match(/F?\s*\.?(\d+\.?\d*)/);
    const exhibitionSt = stMatch ? parseFloat_(`.${stMatch[1]}`) : undefined;

    const entry = entries.find((e) => e.boatNumber === boatNumber);
    if (entry && exhibitionSt !== undefined) {
      entry.exhibitionSt = exhibitionSt;
    }
  });

  // Stabilizer: check for "安定板" text on the page
  const pageText = $("body").text();
  const hasStabilizer = pageText.includes("安定板使用");
  if (hasStabilizer) {
    for (const entry of entries) {
      entry.stabilizer = true;
    }
  }

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

export function parseRaceResult(
  $: CheerioAPI,
  context: RaceContext,
): RaceResultData | null {
  const { params, raceDate } = context;
  const stadiumId = Number.parseInt(params.stadiumCode, 10);

  // Result table: table.is-w495 with 着/枠/ボートレーサー/レースタイム columns
  const resultTable = $("table.is-w495 th:contains('着')")
    .closest("table")
    .first();

  if (resultTable.length === 0) {
    logger.warn(`No result table found for R${params.raceNumber}`);
    return null;
  }

  const resultEntries: RaceResultEntry[] = [];

  resultTable.find("tbody").each((_i, tbody) => {
    const row = $(tbody).find("tr").first();
    const tds = row.find("td");

    // Column 0: 着 (finish position) — full-width number like "１"
    const posText = normalizeText(tds.eq(0).text());
    const finishPosition = parseInt_(
      posText.replace(/[０-９]/g, (c) =>
        String.fromCharCode(c.charCodeAt(0) - 0xfee0),
      ),
    );

    // Column 1: 枠 (boat number) — has is-boatColor class
    const boatNumber = parseInt_(normalizeText(tds.eq(1).text()));
    if (!boatNumber) return;

    // Column 3: レースタイム — format: 1'48"7
    const timeText = normalizeText(tds.eq(3).text());
    const raceTime = timeText || undefined;

    resultEntries.push({
      boatNumber,
      finishPosition,
      raceTime,
    });
  });

  // Start info: course order and start timings from the second table
  $(".table1_boatImage1.is-type1__3rdadd").each((_i, div) => {
    const $div = $(div);
    const numberEl = $div.find(".table1_boatImage1Number");
    const typeClass = numberEl.attr("class") ?? "";
    const typeMatch = typeClass.match(/is-type(\d)/);
    const boatNumber = typeMatch ? Number.parseInt(typeMatch[1], 10) : 0;

    // Course number is the display order (i + 1)
    const courseNumber = _i + 1;

    // Start timing from TimeInner
    const timeInnerText = normalizeText(
      $div.find(".table1_boatImage1TimeInner").text(),
    );
    // ".18 まくり" or ".28"
    const stMatch = timeInnerText.match(/\.(\d+)/);
    const startTiming = stMatch ? parseFloat_(`.${stMatch[1]}`) : undefined;

    const entry = resultEntries.find((e) => e.boatNumber === boatNumber);
    if (entry) {
      entry.courseNumber = courseNumber;
      entry.startTiming = startTiming;
    }
  });

  // Decision technique (決まり手)
  const techniqueTable = $("th:contains('決まり手')").closest("table");
  const technique =
    normalizeText(techniqueTable.find("tbody td").first().text()) || undefined;

  // Weather conditions
  const weather = parseWeather($);

  // Payouts
  const payouts = parsePayouts($);

  if (resultEntries.length === 0) {
    logger.warn(`No result entries parsed for R${params.raceNumber}`);
    return null;
  }

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

  // Weather text
  const weatherUnit = $(".weather1_bodyUnit.is-weather");
  weather.weatherText =
    normalizeText(weatherUnit.find(".weather1_bodyUnitLabelTitle").text()) ||
    undefined;

  // Temperature
  const tempText = $(
    ".weather1_bodyUnit.is-direction .weather1_bodyUnitLabelData",
  )
    .text()
    .trim();
  const tempMatch = tempText.match(/([\d.]+)/);
  weather.temperature = tempMatch ? parseFloat_(tempMatch[1]) : undefined;

  // Wind speed
  const windText = $(".weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData")
    .text()
    .trim();
  const windMatch = windText.match(/(\d+)/);
  weather.windSpeed = windMatch ? parseInt_(windMatch[1]) : undefined;

  // Wind direction: is-wind1 through is-wind16
  const windDirEl = $(
    ".weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage",
  );
  const windDirClass = windDirEl.attr("class") ?? "";
  const windDirMatch = windDirClass.match(/is-wind(\d+)/);
  weather.windDirection = windDirMatch ? parseInt_(windDirMatch[1]) : undefined;

  // Water temperature
  const waterTempText = $(
    ".weather1_bodyUnit.is-waterTemperature .weather1_bodyUnitLabelData",
  )
    .text()
    .trim();
  const waterTempMatch = waterTempText.match(/([\d.]+)/);
  weather.waterTemperature = waterTempMatch
    ? parseFloat_(waterTempMatch[1])
    : undefined;

  // Wave height
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

  // Payout table has 勝式/組番/払戻金/人気 columns
  const payoutTable = $("th:contains('勝式')").closest("table");
  if (payoutTable.length === 0) return payouts;

  payoutTable.find("tbody").each((_i, tbody) => {
    const $tbody = $(tbody);
    const rows = $tbody.find("tr.is-p3-0");

    // First row has the bet type in a rowspan td
    const betTypeEl = rows.first().find("td[rowspan]").first();
    const betType = normalizeText(betTypeEl.text());
    if (!betType) return;

    rows.each((_j, row) => {
      const $row = $(row);

      // Combination from numberSet1_number elements
      const numbers: string[] = [];
      $row.find(".numberSet1_number").each((_k, num) => {
        numbers.push(normalizeText($(num).text()));
      });
      if (numbers.length === 0) return;

      // Separator: - or =
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

      // Payout amount
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

function extractGrade(classStr: string): string | undefined {
  if (classStr.includes("is-sg")) return "SG";
  if (classStr.includes("is-g1")) return "G1";
  if (classStr.includes("is-g2")) return "G2";
  if (classStr.includes("is-g3")) return "G3";
  if (classStr.includes("is-ippan")) return "一般";
  return undefined;
}
