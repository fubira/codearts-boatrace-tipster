/** Odds page parsers for boatrace.jp */

import * as cheerio from "cheerio";

export interface OddsEntry {
  betType: string;
  combination: string;
  odds: number;
}

function parseFloat_(s: string): number | undefined {
  const v = Number.parseFloat(s.replace(/,/g, ""));
  return Number.isNaN(v) || v <= 0 ? undefined : v;
}

/** Find table following a title7_mainLabel with given text */
function findTableByLabel(
  $: cheerio.CheerioAPI,
  label: string,
): ReturnType<cheerio.CheerioAPI> {
  const header = $(`.title7_mainLabel:contains("${label}")`);
  // Navigate: title7_mainLabel → .title7 → next sibling .table1 → inner table
  return header.closest(".title7").next(".table1").find("table").first();
}

/**
 * Parse oddstf page — 単勝・複勝オッズ
 */
export function parseOddsTf(html: string): OddsEntry[] {
  const $ = cheerio.load(html);
  const entries: OddsEntry[] = [];

  const tanshoTable = findTableByLabel($, "単勝オッズ");
  tanshoTable.find("tbody").each((_i, tbody) => {
    const tr = $(tbody).find("tr").first();
    const tds = tr.find("td");
    const boatNum = tds.first().text().trim();
    const odds = parseFloat_(tr.find(".oddsPoint").text().trim());
    if (boatNum && odds) {
      entries.push({ betType: "単勝", combination: boatNum, odds });
    }
  });

  const fukushoTable = findTableByLabel($, "複勝オッズ");
  fukushoTable.find("tbody").each((_i, tbody) => {
    const tr = $(tbody).find("tr").first();
    const tds = tr.find("td");
    const boatNum = tds.first().text().trim();
    const oddsText = tr.find(".oddsPoint").text().trim();
    const lower = parseFloat_(oddsText.split("-")[0]);
    if (boatNum && lower) {
      entries.push({ betType: "複勝", combination: boatNum, odds: lower });
    }
  });

  return entries;
}

/**
 * Parse odds2tf page — 2連単・2連複オッズ
 * Matrix table: header row = 1st place boats, body rows = 2nd place boat + odds
 */
export function parseOdds2Tf(html: string): OddsEntry[] {
  const $ = cheerio.load(html);
  const entries: OddsEntry[] = [];

  for (const [label, betType, separator] of [
    ["2連単オッズ", "2連単", "-"],
    ["2連複オッズ", "2連複", "="],
  ] as const) {
    const table = findTableByLabel($, label);
    if (table.length === 0) continue;

    // Header: first place boat numbers
    const headerBoats: string[] = [];
    table.find("thead th").each((_i, th) => {
      const text = $(th).text().trim();
      if (text.match(/^\d$/)) headerBoats.push(text);
    });

    table.find("tbody.is-p3-0 tr").each((_i, tr) => {
      const tds = $(tr).find("td");
      let colIdx = 0;

      for (let j = 0; j < tds.length; j++) {
        const td = $(tds[j]);
        if (td.hasClass("oddsPoint")) {
          const odds = parseFloat_(td.text().trim());
          const secondBoatTd = $(tds[j - 1]);
          const secondBoat = secondBoatTd.text().trim();

          if (odds && secondBoat.match(/^\d$/) && colIdx < headerBoats.length) {
            const firstBoat = headerBoats[colIdx];
            const combo =
              separator === "="
                ? [firstBoat, secondBoat].sort().join("=")
                : `${firstBoat}${separator}${secondBoat}`;
            entries.push({ betType, combination: combo, odds });
          }
          colIdx++;
        }
      }
    });
  }

  return entries;
}

/**
 * Parse odds3t page — 3連単オッズ
 */
export function parseOdds3T(html: string): OddsEntry[] {
  const $ = cheerio.load(html);
  const entries: OddsEntry[] = [];

  const table = findTableByLabel($, "3連単オッズ");
  if (table.length === 0) return entries;

  const headerBoats: string[] = [];
  table.find("thead th").each((_i, th) => {
    const text = $(th).text().trim();
    if (text.match(/^\d$/)) headerBoats.push(text);
  });

  table.find("tbody.is-p3-0").each((_i, tbody) => {
    const rows = $(tbody).find("tr");
    let secondBoat = "";

    rows.each((_j, tr) => {
      const tds = $(tr).find("td");
      let colIdx = 0;

      for (let k = 0; k < tds.length; k++) {
        const td = $(tds[k]);

        if (td.attr("rowspan") && td.hasClass("is-borderLeftNone")) {
          secondBoat = td.text().trim();
          continue;
        }

        if (td.hasClass("oddsPoint")) {
          const odds = parseFloat_(td.text().trim());
          const thirdBoatTd = $(tds[k - 1]);
          const thirdBoat = thirdBoatTd.text().trim();

          if (odds && thirdBoat.match(/^\d$/) && colIdx < headerBoats.length) {
            const firstBoat = headerBoats[colIdx];
            entries.push({
              betType: "3連単",
              combination: `${firstBoat}-${secondBoat}-${thirdBoat}`,
              odds,
            });
          }
          colIdx++;
        }
      }
    });
  });

  return entries;
}

/**
 * Parse odds3f page — 3連複オッズ
 */
export function parseOdds3F(html: string): OddsEntry[] {
  const $ = cheerio.load(html);
  const entries: OddsEntry[] = [];

  const table = findTableByLabel($, "3連複オッズ");
  if (table.length === 0) return entries;

  const headerBoats: string[] = [];
  table.find("thead th").each((_i, th) => {
    const text = $(th).text().trim();
    if (text.match(/^\d$/)) headerBoats.push(text);
  });

  table.find("tbody.is-p3-0").each((_i, tbody) => {
    const rows = $(tbody).find("tr");
    let secondBoat = "";

    rows.each((_j, tr) => {
      const tds = $(tr).find("td");
      let colIdx = 0;

      for (let k = 0; k < tds.length; k++) {
        const td = $(tds[k]);

        if (td.attr("rowspan") && !td.hasClass("is-disabled")) {
          const text = td.text().trim();
          if (text.match(/^\d$/)) secondBoat = text;
          continue;
        }

        if (td.hasClass("oddsPoint")) {
          const odds = parseFloat_(td.text().trim());
          const thirdBoatTd = $(tds[k - 1]);
          const thirdBoat = thirdBoatTd.text().trim();

          if (odds && thirdBoat.match(/^\d$/) && colIdx < headerBoats.length) {
            const firstBoat = headerBoats[colIdx];
            const sorted = [firstBoat, secondBoat, thirdBoat].sort();
            entries.push({
              betType: "3連複",
              combination: `${sorted[0]}=${sorted[1]}=${sorted[2]}`,
              odds,
            });
          }
          colIdx++;
        }
      }
    });
  });

  return entries;
}
