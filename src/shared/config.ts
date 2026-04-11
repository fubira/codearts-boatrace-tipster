import { existsSync, readFileSync } from "node:fs";
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

export interface ModelStrategy {
  evThreshold: number;
  gap23Threshold: number;
  top3ConcThreshold: number;
  unitDivisor: number;
  betCap: number;
}

/** Load P2 strategy parameters from model_meta.json */
export function loadModelStrategy(): ModelStrategy {
  const metaPath = resolve(
    PROJECT_ROOT,
    "ml/models/trifecta_v1/ranking/model_meta.json",
  );
  const defaults: ModelStrategy = {
    evThreshold: 0.0,
    gap23Threshold: 0.13,
    top3ConcThreshold: 0.0,
    unitDivisor: 150,
    betCap: 30000,
  };
  if (!existsSync(metaPath)) {
    return defaults;
  }
  const meta = JSON.parse(readFileSync(metaPath, "utf-8"));
  const strategy = meta.strategy ?? {};
  return {
    evThreshold: strategy.ev_threshold ?? defaults.evThreshold,
    gap23Threshold: strategy.gap23_threshold ?? defaults.gap23Threshold,
    top3ConcThreshold:
      strategy.top3_conc_threshold ?? defaults.top3ConcThreshold,
    unitDivisor: strategy.unit_divisor ?? defaults.unitDivisor,
    betCap: strategy.bet_cap ?? defaults.betCap,
  };
}

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
