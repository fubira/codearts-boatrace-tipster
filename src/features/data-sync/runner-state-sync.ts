/** Pull-only sync for runner-state.json.
 *
 * Pull だけサポートする (push しない): runner-state.json はサーバ runner
 * が管理する運用状態 (bankroll、today bets/results、skip counts)。ローカル
 * から上書きすると本番 runner の判断を壊すため、方向を固定する。
 */

import { config } from "@/shared/config";
import { logger } from "@/shared/logger";
import { rsync } from "./ssh";
import type { SyncConfig } from "./sync-config";

export interface RunnerStateSyncResult {
  pulled: number;
}

export function syncRunnerState(
  conf: SyncConfig,
  options?: { dryRun?: boolean },
): RunnerStateSyncResult {
  const localPath = `${config.dataDir}/runner-state.json`;
  const remotePath = `${conf.server}:${conf.prodDir}/data/runner-state.json`;

  logger.info("Pulling runner-state.json from server...");
  const result = rsync(remotePath, localPath, {
    dryRun: options?.dryRun,
  });

  return { pulled: result.transferred };
}
