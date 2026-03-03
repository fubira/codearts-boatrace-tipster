import { resolve } from "node:path";

const PROJECT_ROOT = resolve(import.meta.dirname, "../..");

export const config = {
  projectRoot: PROJECT_ROOT,
  dataDir: resolve(PROJECT_ROOT, "data"),
  cacheDir: resolve(PROJECT_ROOT, "data/cache"),
  dbPath: resolve(PROJECT_ROOT, "data/boatrace-tipster.db"),
} as const;
