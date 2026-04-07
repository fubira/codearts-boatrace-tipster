/** Parsers for BOATCAST cache files (oriten / stt) */

export interface OritenEntry {
  boatNumber: number;
  lapTime: number | null;
  turnTime: number | null;
  straightTime: number | null;
}

export interface SttEntry {
  boatNumber: number;
  course: number | null;
  st1: number | null;
  st2: number | null;
  isFlying: boolean;
  slitDiff: number | null;
}

function parseNumOrNull(value: string): number | null {
  if (!value || value.includes("-")) return null;
  const n = Number.parseFloat(value);
  return Number.isNaN(n) ? null : n;
}

/**
 * Parse oriten (展示タイム詳細) cache file.
 *
 * Format:
 *   data=
 *   {n}\t{n}
 *   一　周|半周ラップ\tまわり足\t直　線
 *   {boatNumber}\t{name}\t{lap}\t{turn}\t{straight}
 *   ... (6 rows)
 */
export function parseOriten(content: string): OritenEntry[] {
  if (!content || content.length < 10) return [];

  const lines = content.split("\n").filter((l) => l.trim() !== "");
  // Skip: "data=", metadata line, header line → data starts at index 3
  if (lines.length < 4) return [];

  const entries: OritenEntry[] = [];
  for (let i = 3; i < lines.length; i++) {
    const cols = lines[i].split("\t");
    if (cols.length < 5) continue;

    const boatNumber = Number.parseInt(cols[0], 10);
    if (Number.isNaN(boatNumber) || boatNumber < 1 || boatNumber > 6) continue;

    entries.push({
      boatNumber,
      lapTime: parseNumOrNull(cols[2]),
      turnTime: parseNumOrNull(cols[3]),
      straightTime: parseNumOrNull(cols[4]),
    });
  }

  return entries;
}

/**
 * Parse stt (スタート展示) cache file.
 *
 * Format:
 *   data=
 *   {n}
 *   {boatNumber}\t{course}\t{name}\t{st1}\t{st2}\t{fFlag}\t{slitDiff}
 *   ... (6 rows)
 */
export function parseStt(content: string): SttEntry[] {
  if (!content || content.length < 10) return [];

  const lines = content.split("\n").filter((l) => l.trim() !== "");
  // Skip: "data=", metadata line → data starts at index 2
  if (lines.length < 3) return [];

  const entries: SttEntry[] = [];
  for (let i = 2; i < lines.length; i++) {
    const cols = lines[i].split("\t");
    if (cols.length < 6) continue;

    const boatNumber = Number.parseInt(cols[0], 10);
    if (Number.isNaN(boatNumber) || boatNumber < 1 || boatNumber > 6) continue;

    entries.push({
      boatNumber,
      course: parseNumOrNull(cols[1]),
      st1: parseNumOrNull(cols[3]),
      st2: parseNumOrNull(cols[4]),
      isFlying: cols[5]?.trim() === "F",
      slitDiff: parseNumOrNull(cols[6] ?? ""),
    });
  }

  return entries;
}
