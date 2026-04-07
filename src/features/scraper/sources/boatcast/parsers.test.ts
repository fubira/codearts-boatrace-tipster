import { describe, expect, test } from "bun:test";
import { parseOriten, parseStt } from "./parsers";

describe("parseOriten", () => {
  test("parses full-lap (一周) format", () => {
    const content = [
      "data=",
      "1\t3",
      "一　周\tまわり足\t直　線",
      "1\t大西　　　賢\t37.44\t5.68\t7.14",
      "2\t村田　　修次\t37.81\t5.74\t7.37",
      "3\t西村　　　勝\t37.56\t5.85\t7.24",
      "4\t新出　　浩司\t38.07\t5.62\t7.35",
      "5\t伊藤　　紘章\t37.94\t5.75\t7.27",
      "6\t前田　　光昭\t37.86\t6.09\t7.24",
    ].join("\n");

    const entries = parseOriten(content);
    expect(entries).toHaveLength(6);
    expect(entries[0]).toEqual({
      boatNumber: 1,
      lapTime: 37.44,
      turnTime: 5.68,
      straightTime: 7.14,
    });
    expect(entries[5]).toEqual({
      boatNumber: 6,
      lapTime: 37.86,
      turnTime: 6.09,
      straightTime: 7.24,
    });
  });

  test("parses half-lap (半周ラップ) format — Kiryu", () => {
    const content = [
      "data=",
      "1\t3",
      "半周ラップ\tまわり足\t直　線",
      "1\t富澤　　祐作\t18.16\t4.69\t7.70",
      "2\t塩崎　　優司\t18.72\t4.43\t7.88",
    ].join("\n");

    const entries = parseOriten(content);
    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      boatNumber: 1,
      lapTime: 18.16,
      turnTime: 4.69,
      straightTime: 7.7,
    });
  });

  test("returns empty array for empty/short content", () => {
    expect(parseOriten("")).toHaveLength(0);
    expect(parseOriten("data=")).toHaveLength(0);
    expect(parseOriten("short")).toHaveLength(0);
  });

  test("skips lines with insufficient columns", () => {
    const content = [
      "data=",
      "1\t3",
      "一　周\tまわり足\t直　線",
      "1\t大西\t37.44\t5.68\t7.14",
      "bad line",
      "3\t西村\t37.56\t5.85\t7.24",
    ].join("\n");

    const entries = parseOriten(content);
    expect(entries).toHaveLength(2);
    expect(entries[0].boatNumber).toBe(1);
    expect(entries[1].boatNumber).toBe(3);
  });
});

describe("parseStt", () => {
  test("parses normal stt with all data", () => {
    const content = [
      "data=",
      "1",
      "1\t1\t大西　　　賢\t.19\t.03\tF\t2.0",
      "2\t2\t村田　　修次\t.15\t.07\t\t2.5",
      "3\t3\t西村　　　勝\t.19\t.09\t\t1.0",
      "4\t4\t新出　　浩司\t.13\t.25\t\t3.0",
      "5\t5\t伊藤　　紘章\t.12\t.01\tF\t3.0",
      "6\t6\t前田　　光昭\t.01\t.02\t\t1.0",
    ].join("\n");

    const entries = parseStt(content);
    expect(entries).toHaveLength(6);
    expect(entries[0]).toEqual({
      boatNumber: 1,
      course: 1,
      st1: 0.19,
      st2: 0.03,
      isFlying: true,
      slitDiff: 2.0,
    });
    expect(entries[1]).toEqual({
      boatNumber: 2,
      course: 2,
      st1: 0.15,
      st2: 0.07,
      isFlying: false,
      slitDiff: 2.5,
    });
  });

  test("handles ------ values as null", () => {
    const content = [
      "data=",
      "1",
      "1\t1\t富澤　　祐作\t------\t.14\t\t------",
      "2\t2\t塩崎　　優司\t------\t.04\t\t------",
    ].join("\n");

    const entries = parseStt(content);
    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      boatNumber: 1,
      course: 1,
      st1: null,
      st2: 0.14,
      isFlying: false,
      slitDiff: null,
    });
  });

  test("detects maezuke (course != boat_number)", () => {
    const content = [
      "data=",
      "1",
      "1\t2\t選手A\t.10\t.05\t\t1.0",
      "4\t1\t選手B\t.08\t.03\t\t2.0",
    ].join("\n");

    const entries = parseStt(content);
    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({
      boatNumber: 1,
      course: 2,
      st1: 0.1,
      st2: 0.05,
      isFlying: false,
      slitDiff: 1.0,
    });
    expect(entries[1]).toEqual({
      boatNumber: 4,
      course: 1,
      st1: 0.08,
      st2: 0.03,
      isFlying: false,
      slitDiff: 2.0,
    });
  });

  test("returns empty array for empty/short content", () => {
    expect(parseStt("")).toHaveLength(0);
    expect(parseStt("data=")).toHaveLength(0);
  });
});
