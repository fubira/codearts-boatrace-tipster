import { describe, expect, test } from "bun:test";
import {
  type RaceSlot,
  allDone,
  buildSchedule,
  getActionableRaces,
} from "./race-scheduler";

function makeSlot(overrides: Partial<RaceSlot> = {}): RaceSlot {
  return {
    raceId: 1,
    stadiumId: 1,
    stadiumName: "桐生",
    raceNumber: 1,
    deadline: "15:00",
    deadlineMs: new Date("2026-04-03T15:00:00+09:00").getTime(),
    status: "waiting",
    ...overrides,
  };
}

describe("buildSchedule", () => {
  const stadiumNames = new Map([
    [1, "桐生"],
    [4, "平和島"],
  ]);

  test("builds schedule from DB rows", () => {
    const races = [
      { id: 100, stadium_id: 1, race_number: 1, deadline: "10:30" },
      { id: 101, stadium_id: 4, race_number: 1, deadline: "10:45" },
      { id: 102, stadium_id: 1, race_number: 2, deadline: "11:00" },
    ];
    const schedule = buildSchedule(races, stadiumNames, "2026-04-03");

    expect(schedule).toHaveLength(3);
    expect(schedule[0].stadiumName).toBe("桐生");
    expect(schedule[0].deadline).toBe("10:30");
    expect(schedule[1].stadiumName).toBe("平和島");
    expect(schedule[2].raceNumber).toBe(2);
  });

  test("sorts by deadline ascending", () => {
    const races = [
      { id: 100, stadium_id: 1, race_number: 2, deadline: "11:00" },
      { id: 101, stadium_id: 4, race_number: 1, deadline: "10:30" },
    ];
    const schedule = buildSchedule(races, stadiumNames, "2026-04-03");

    expect(schedule[0].deadline).toBe("10:30");
    expect(schedule[1].deadline).toBe("11:00");
  });

  test("skips races without deadline", () => {
    const races = [
      { id: 100, stadium_id: 1, race_number: 1, deadline: "10:30" },
      { id: 101, stadium_id: 4, race_number: 1, deadline: null },
    ];
    const schedule = buildSchedule(races, stadiumNames, "2026-04-03");

    expect(schedule).toHaveLength(1);
  });

  test("converts deadline to epoch ms in JST", () => {
    const races = [
      { id: 100, stadium_id: 1, race_number: 1, deadline: "15:00" },
    ];
    const schedule = buildSchedule(races, stadiumNames, "2026-04-03");

    const expected = new Date("2026-04-03T15:00:00+09:00").getTime();
    expect(schedule[0].deadlineMs).toBe(expected);
  });
});

describe("getActionableRaces", () => {
  const deadlineMs = new Date("2026-04-03T15:00:00+09:00").getTime();

  test("returns beforeInfo when within 5 min of deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "waiting" });
    const now = deadlineMs - 4 * 60_000; // 4 min before

    const { beforeInfo, predict, results } = getActionableRaces([slot], now);
    expect(beforeInfo).toHaveLength(1);
    expect(predict).toHaveLength(0);
    expect(results).toHaveLength(0);
  });

  test("does not return beforeInfo when more than 7 min before deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "waiting" });
    const now = deadlineMs - 8 * 60_000; // 8 min before

    const { beforeInfo } = getActionableRaces([slot], now);
    expect(beforeInfo).toHaveLength(0);
  });

  test("returns predict when within 5 min of deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "before_info" });
    const now = deadlineMs - 4 * 60_000; // 4 min before

    const { beforeInfo, predict } = getActionableRaces([slot], now);
    expect(beforeInfo).toHaveLength(0);
    expect(predict).toHaveLength(1);
  });

  test("does not return predict when more than 5 min before deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "before_info" });
    const now = deadlineMs - 6 * 60_000; // 6 min before

    const { predict } = getActionableRaces([slot], now);
    expect(predict).toHaveLength(0);
  });

  test("returns oddsT3 when within 3 min but more than 1 min before deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "predicted" });
    const now = deadlineMs - 2 * 60_000; // 2 min before

    const { oddsT3, odds } = getActionableRaces([slot], now);
    expect(oddsT3).toHaveLength(1);
    expect(odds).toHaveLength(0);
  });

  test("does not return oddsT3 when more than 3 min before deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "predicted" });
    const now = deadlineMs - 4 * 60_000; // 4 min before

    const { oddsT3 } = getActionableRaces([slot], now);
    expect(oddsT3).toHaveLength(0);
  });

  test("returns odds when within 1 min of deadline for predicted races", () => {
    const slot = makeSlot({ deadlineMs, status: "predicted" });
    const now = deadlineMs - 0.5 * 60_000; // 30 sec before

    const { oddsT3, odds } = getActionableRaces([slot], now);
    expect(odds).toHaveLength(1);
    expect(oddsT3).toHaveLength(0);
  });

  test("does not return odds when more than 1 min before deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "predicted" });
    const now = deadlineMs - 2 * 60_000; // 2 min before

    const { odds } = getActionableRaces([slot], now);
    expect(odds).toHaveLength(0);
  });

  test("returns results 12 min after deadline for decided races", () => {
    const slot = makeSlot({ deadlineMs, status: "decided" });
    const now = deadlineMs + 13 * 60_000; // 13 min after

    const { results } = getActionableRaces([slot], now);
    expect(results).toHaveLength(1);
  });

  test("does not return results before 12 min after deadline", () => {
    const slot = makeSlot({ deadlineMs, status: "decided" });
    const now = deadlineMs + 10 * 60_000; // 10 min after

    const { results } = getActionableRaces([slot], now);
    expect(results).toHaveLength(0);
  });

  test("does not return done races", () => {
    const slot = makeSlot({ deadlineMs, status: "done" });
    const now = deadlineMs + 30 * 60_000;

    const { beforeInfo, predict, results } = getActionableRaces([slot], now);
    expect(beforeInfo).toHaveLength(0);
    expect(predict).toHaveLength(0);
    expect(results).toHaveLength(0);
  });

  test("skips waiting race when deadline passed by 5+ minutes (late start)", () => {
    const slot = makeSlot({ deadlineMs, status: "waiting" });
    const now = deadlineMs + 6 * 60_000; // 6 min after deadline

    const { beforeInfo, predict, results } = getActionableRaces([slot], now);
    expect(beforeInfo).toHaveLength(0);
    expect(predict).toHaveLength(0);
    expect(results).toHaveLength(0);
    expect(slot.status).toBe("done");
  });

  test("skips before_info race when deadline passed by 5+ minutes", () => {
    const slot = makeSlot({ deadlineMs, status: "before_info" });
    const now = deadlineMs + 6 * 60_000;

    const { predict } = getActionableRaces([slot], now);
    expect(predict).toHaveLength(0);
    expect(slot.status).toBe("done");
  });

  test("does not skip waiting race just after deadline (within 5 min)", () => {
    const slot = makeSlot({ deadlineMs, status: "waiting" });
    const now = deadlineMs + 3 * 60_000; // 3 min after deadline

    const { beforeInfo } = getActionableRaces([slot], now);
    expect(beforeInfo).toHaveLength(1);
    expect(slot.status).toBe("waiting");
  });

  test("handles result_pending same as decided", () => {
    const slot = makeSlot({ deadlineMs, status: "result_pending" });
    const now = deadlineMs + 12 * 60_000;

    const { results } = getActionableRaces([slot], now);
    expect(results).toHaveLength(1);
  });

  test("skips predicted race when deadline passed by 5+ minutes", () => {
    const slot = makeSlot({ deadlineMs, status: "predicted" });
    const now = deadlineMs + 6 * 60_000;

    const { odds } = getActionableRaces([slot], now);
    expect(odds).toHaveLength(0);
    expect(slot.status).toBe("done");
  });

  test("handles mixed statuses across multiple races", () => {
    const slots = [
      makeSlot({
        raceId: 1,
        deadlineMs: deadlineMs - 40 * 60_000,
        status: "decided",
      }),
      makeSlot({ raceId: 2, deadlineMs, status: "waiting" }),
      makeSlot({
        raceId: 3,
        deadlineMs: deadlineMs + 30 * 60_000,
        status: "waiting",
      }),
    ];
    const now = deadlineMs - 4 * 60_000; // 4 min before race 2

    const { beforeInfo, predict, results } = getActionableRaces(slots, now);
    // Race 1: decided, deadline was 40 min ago → 36 min after → results
    expect(results).toHaveLength(1);
    expect(results[0].raceId).toBe(1);
    // Race 2: waiting, 4 min before → beforeInfo
    expect(beforeInfo).toHaveLength(1);
    expect(beforeInfo[0].raceId).toBe(2);
    // Race 3: waiting, 34 min before → not yet
    expect(predict).toHaveLength(0);
  });
});

describe("allDone", () => {
  test("returns true when all races are done", () => {
    const slots = [makeSlot({ status: "done" }), makeSlot({ status: "done" })];
    expect(allDone(slots)).toBe(true);
  });

  test("returns false when some races are not done", () => {
    const slots = [
      makeSlot({ status: "done" }),
      makeSlot({ status: "predicted" }),
    ];
    expect(allDone(slots)).toBe(false);
  });

  test("returns false for empty schedule", () => {
    expect(allDone([])).toBe(false);
  });
});
