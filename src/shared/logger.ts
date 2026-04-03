import { appendFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

let currentLevel: LogLevel = "info";
let logDir: string | null = null;

export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
}

/** Enable file logging to the specified directory (daily rotation). */
export function enableFileLog(dir: string): void {
  mkdirSync(dir, { recursive: true });
  logDir = dir;
}

function shouldLog(level: LogLevel): boolean {
  return LOG_LEVELS[level] >= LOG_LEVELS[currentLevel];
}

function formatMessage(level: LogLevel, message: string): string {
  const timestamp = new Date()
    .toLocaleString("sv-SE", { timeZone: "Asia/Tokyo", hour12: false })
    .replace(",", "");
  return `[${timestamp}] [${level.toUpperCase()}] ${message}`;
}

function writeToFile(formatted: string): void {
  if (!logDir) return;
  const date = new Date()
    .toLocaleDateString("sv-SE", { timeZone: "Asia/Tokyo" })
    .replace(/\//g, "-");
  const filePath = resolve(logDir, `runner-${date}.log`);
  appendFileSync(filePath, `${formatted}\n`);
}

export const logger = {
  debug(message: string): void {
    if (shouldLog("debug")) {
      const formatted = formatMessage("debug", message);
      console.debug(formatted);
      writeToFile(formatted);
    }
  },
  info(message: string): void {
    if (shouldLog("info")) {
      const formatted = formatMessage("info", message);
      console.info(formatted);
      writeToFile(formatted);
    }
  },
  warn(message: string): void {
    if (shouldLog("warn")) {
      const formatted = formatMessage("warn", message);
      console.warn(formatted);
      writeToFile(formatted);
    }
  },
  error(message: string): void {
    if (shouldLog("error")) {
      const formatted = formatMessage("error", message);
      console.error(formatted);
      writeToFile(formatted);
    }
  },
};
