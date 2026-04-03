import { describe, expect, test } from "bun:test";
import { calcMaxBetForPool, parseTanshoPool } from "./pool-size";

describe("parseTanshoPool", () => {
  const sampleData = `data=
1
43\t236\t76\t85\t31\t11\t23\t534\t383\t101\t5\t14\t8\t48\t100\t24\t12\t9\t15\t516
2274\t0\t2274
116\t105\t80\t12\t27\t4\t13\t11\t10\t2\t4\t1\t24\t56\t17
482\t0\t482
45\t0\t1\t6\t2\t3\t0\t0\t0\t0\t0\t0
57\t0\t57
15\t5\t9\t20\t2\t0\t0\t0\t0\t0\t0\t0
51\t0\t51
渡邉　　英児\t大井　　清貴\t根岸　　真優\t中山　　　将\t乙藤　　智史\t前田　　将太`;

  test("parses tansho total votes", () => {
    const result = parseTanshoPool(sampleData);
    expect(result).not.toBeNull();
    expect(result?.totalVotes).toBe(57);
    expect(result?.poolSize).toBe(5700);
  });

  test("parses per-boat votes", () => {
    const result = parseTanshoPool(sampleData);
    expect(result?.votesByBoat).toEqual([45, 0, 1, 6, 2, 3]);
  });

  test("returns null for empty data", () => {
    expect(parseTanshoPool("data=\n2")).toBeNull();
  });

  test("returns null for invalid data", () => {
    expect(parseTanshoPool("")).toBeNull();
  });
});

describe("calcMaxBetForPool", () => {
  test("limits bet for small pool", () => {
    const pool = {
      totalVotes: 57,
      poolSize: 5700,
      votesByBoat: [45, 0, 1, 6, 2, 3],
    };
    const { maxBet } = calcMaxBetForPool(0.65, 2.0, pool, 4000);
    expect(maxBet).toBeLessThan(4000);
    expect(maxBet).toBeGreaterThanOrEqual(0);
  });

  test("allows full bet for large pool", () => {
    const pool = {
      totalVotes: 5000,
      poolSize: 500000,
      votesByBoat: [2500, 500, 500, 500, 500, 500],
    };
    const { maxBet } = calcMaxBetForPool(0.65, 2.0, pool, 4000);
    expect(maxBet).toBe(4000);
  });

  test("returns 0 when EV would go negative at any bet", () => {
    // odds = 1.0, prob = 0.5 → EV already negative, no bet makes sense
    const pool = {
      totalVotes: 100,
      poolSize: 10000,
      votesByBoat: [90, 2, 2, 2, 2, 2],
    };
    const { maxBet } = calcMaxBetForPool(0.5, 1.0, pool, 4000);
    expect(maxBet).toBe(0);
  });

  test("caps at betCap even if pool allows more", () => {
    const pool = {
      totalVotes: 50000,
      poolSize: 5000000,
      votesByBoat: [10000, 10000, 10000, 10000, 5000, 5000],
    };
    const { maxBet } = calcMaxBetForPool(0.7, 2.0, pool, 4000);
    expect(maxBet).toBe(4000);
  });

  test("rounds down to 100 yen units", () => {
    const pool = {
      totalVotes: 200,
      poolSize: 20000,
      votesByBoat: [100, 20, 20, 20, 20, 20],
    };
    const { maxBet } = calcMaxBetForPool(0.65, 2.0, pool, 4000);
    expect(maxBet % 100).toBe(0);
  });
});
