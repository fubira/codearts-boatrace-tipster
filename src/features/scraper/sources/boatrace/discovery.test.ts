import { describe, expect, test } from "bun:test";
import * as cheerio from "cheerio";
import { parseSchedulePage } from "./discovery";

function makeScheduleHtml(links: string[]): string {
  const anchors = links.map((href) => `<a href="${href}">venue</a>`).join("\n");
  return `<html><body>${anchors}</body></html>`;
}

describe("parseSchedulePage", () => {
  test("extracts unique stadium codes from jcd= links", () => {
    const html = makeScheduleHtml([
      "/race/racelist?jcd=04&hd=20260301",
      "/race/racelist?jcd=12&hd=20260301",
      "/race/racelist?jcd=24&hd=20260301",
    ]);
    const $ = cheerio.load(html);
    const venues = parseSchedulePage($, "20260301");

    expect(venues).toHaveLength(3);
    expect(venues.map((v) => v.stadiumCode)).toEqual(["04", "12", "24"]);
    expect(venues.every((v) => v.date === "20260301")).toBe(true);
  });

  test("deduplicates same stadium code", () => {
    const html = makeScheduleHtml([
      "/race/racelist?rno=1&jcd=04&hd=20260301",
      "/race/racelist?rno=2&jcd=04&hd=20260301",
      "/race/racelist?rno=3&jcd=04&hd=20260301",
    ]);
    const $ = cheerio.load(html);
    const venues = parseSchedulePage($, "20260301");

    expect(venues).toHaveLength(1);
    expect(venues[0].stadiumCode).toBe("04");
  });

  test("sorts by stadium code ascending", () => {
    const html = makeScheduleHtml([
      "/race/racelist?jcd=24&hd=20260301",
      "/race/racelist?jcd=01&hd=20260301",
      "/race/racelist?jcd=13&hd=20260301",
    ]);
    const $ = cheerio.load(html);
    const venues = parseSchedulePage($, "20260301");

    expect(venues.map((v) => v.stadiumCode)).toEqual(["01", "13", "24"]);
  });

  test("returns empty array when no jcd= links", () => {
    const html = makeScheduleHtml(["/race/other?page=1", "/about"]);
    const $ = cheerio.load(html);
    const venues = parseSchedulePage($, "20260301");

    expect(venues).toEqual([]);
  });

  test("returns empty array for empty HTML", () => {
    const $ = cheerio.load("<html><body></body></html>");
    const venues = parseSchedulePage($, "20260301");

    expect(venues).toEqual([]);
  });

  test("single-digit jcd is ignored, 3-digit jcd matches first 2 digits", () => {
    const html = makeScheduleHtml([
      "/race/racelist?jcd=4&hd=20260301",
      "/race/racelist?jcd=123&hd=20260301",
      "/race/racelist?jcd=04&hd=20260301",
    ]);
    const $ = cheerio.load(html);
    const venues = parseSchedulePage($, "20260301");

    // jcd=4 doesn't match \d{2}, jcd=123 matches "12", jcd=04 matches "04"
    expect(venues).toHaveLength(2);
    expect(venues.map((v) => v.stadiumCode)).toEqual(["04", "12"]);
  });

  test("handles mixed links with and without jcd", () => {
    const html = makeScheduleHtml([
      "/race/racelist?jcd=04&hd=20260301",
      "/news/detail?id=123",
      "/race/racelist?jcd=12&hd=20260301",
      "/owpc/pc/race/index?hd=20260301",
    ]);
    const $ = cheerio.load(html);
    const venues = parseSchedulePage($, "20260301");

    expect(venues).toHaveLength(2);
    expect(venues.map((v) => v.stadiumCode)).toEqual(["04", "12"]);
  });
});
