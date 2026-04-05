import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const PROJECT_ROOT = resolve(import.meta.dirname, "../..");

const pkg = JSON.parse(
  readFileSync(resolve(PROJECT_ROOT, "package.json"), "utf-8"),
);

export const config = {
  projectRoot: PROJECT_ROOT,
  version: pkg.version as string,
  dataDir: resolve(PROJECT_ROOT, "data"),
  cacheDir: resolve(PROJECT_ROOT, "data/cache"),
  dbPath: resolve(PROJECT_ROOT, "data/boatrace-tipster.db"),
} as const;

export function loadTelebotCredentials(): {
  subscriberNumber: string;
  pin: string;
  password: string;
  betPassword: string;
} | null {
  const subscriberNumber = process.env.TELEBOAT_SUBSCRIBER_NUMBER;
  const pin = process.env.TELEBOAT_PIN;
  const password = process.env.TELEBOAT_PASSWORD;
  const betPassword = process.env.TELEBOAT_BET_PASSWORD;
  if (!subscriberNumber || !pin || !password || !betPassword) return null;
  return { subscriberNumber, pin, password, betPassword };
}
