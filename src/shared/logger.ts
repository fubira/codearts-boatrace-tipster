type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

let currentLevel: LogLevel = "info";

export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
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

export const logger = {
  debug(message: string): void {
    if (shouldLog("debug")) {
      console.debug(formatMessage("debug", message));
    }
  },
  info(message: string): void {
    if (shouldLog("info")) {
      console.info(formatMessage("info", message));
    }
  },
  warn(message: string): void {
    if (shouldLog("warn")) {
      console.warn(formatMessage("warn", message));
    }
  },
  error(message: string): void {
    if (shouldLog("error")) {
      console.error(formatMessage("error", message));
    }
  },
};
