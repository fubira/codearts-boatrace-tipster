import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  test,
} from "bun:test";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import {
  cachePathFor,
  enableCache,
  hasCacheEntry,
  readCache,
  writeCache,
} from "./cache-manager";

describe("cachePathFor", () => {
  test("path with hd= creates YYYYMM subdirectory", () => {
    const result = cachePathFor(
      "/owpc/pc/race/racelist?rno=1&jcd=04&hd=20250115",
    );
    expect(result).toContain("owpc/pc/race/racelist");
    expect(result).toContain("/202501/");
    expect(result).toContain("rno=1&jcd=04&hd=20250115.html.gz");
  });

  test("strips leading slash", () => {
    const withSlash = cachePathFor("/race/racelist?jcd=04&hd=20260301");
    const withoutSlash = cachePathFor("race/racelist?jcd=04&hd=20260301");
    expect(withSlash).toBe(withoutSlash);
  });

  test("path without hd= has no YYYYMM subdirectory", () => {
    const result = cachePathFor("/owpc/pc/race/index?page=1");
    expect(result).toContain("owpc/pc/race/index");
    expect(result).toContain("page=1.html.gz");
    // No 6-digit subdir
    expect(result).not.toMatch(/\/\d{6}\//);
  });

  test("different race numbers produce different paths", () => {
    const r1 = cachePathFor("/race/racelist?rno=1&jcd=04&hd=20250115");
    const r2 = cachePathFor("/race/racelist?rno=2&jcd=04&hd=20250115");
    expect(r1).not.toBe(r2);
  });

  test("different dates produce different YYYYMM dirs", () => {
    const jan = cachePathFor("/race/racelist?rno=1&jcd=04&hd=20250115");
    const feb = cachePathFor("/race/racelist?rno=1&jcd=04&hd=20250215");
    expect(jan).toContain("/202501/");
    expect(feb).toContain("/202502/");
  });

  test("always ends with .html.gz", () => {
    expect(cachePathFor("/race/racelist?rno=1&jcd=04&hd=20250115")).toEndWith(
      ".html.gz",
    );
    expect(cachePathFor("/simple/path?key=val")).toEndWith(".html.gz");
  });
});

describe("writeCache / readCache roundtrip", () => {
  const testCacheDir = resolve(import.meta.dirname, "__test_cache__");

  beforeAll(() => {
    // Enable cache for these tests
    enableCache();
  });

  afterEach(() => {
    // Clean up test cache files
    if (existsSync(testCacheDir)) {
      rmSync(testCacheDir, { recursive: true });
    }
  });

  test("write then read returns original content", () => {
    const path = "/owpc/pc/race/racelist?rno=1&jcd=04&hd=20260301";
    const html = "<html><body>テスト</body></html>";

    writeCache(path, html);
    const result = readCache(path);

    expect(result).toBe(html);
  });

  test("hasCacheEntry returns true after write", () => {
    // Use a unique path unlikely to collide with real cache
    const path = "/owpc/pc/race/racelist?rno=99&jcd=99&hd=29990101";
    // Clean up if leftover from a previous run
    const { existsSync: ex, unlinkSync } = require("node:fs");
    const fullPath = cachePathFor(path);
    if (ex(fullPath)) unlinkSync(fullPath);

    expect(hasCacheEntry(path)).toBe(false);

    writeCache(path, "<html>test</html>");
    expect(hasCacheEntry(path)).toBe(true);

    // Cleanup
    if (ex(fullPath)) unlinkSync(fullPath);
  });

  test("readCache returns undefined for non-existent path", () => {
    const result = readCache(
      "/owpc/pc/race/racelist?rno=99&jcd=99&hd=99991231",
    );
    expect(result).toBeUndefined();
  });

  test("preserves multi-byte content through gzip roundtrip", () => {
    const path = "/owpc/pc/race/beforeinfo?rno=1&jcd=01&hd=20260301";
    const html = "<html><body>桐生 レース1 展示情報 風速3m</body></html>";

    writeCache(path, html);
    const result = readCache(path);

    expect(result).toBe(html);
  });
});
