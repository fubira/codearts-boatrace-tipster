/** SSH helper for executing remote commands. */

import { execSync } from "node:child_process";
import type { SyncConfig } from "./sync-config";

export function sshExec(
  conf: SyncConfig,
  command: string,
  options?: { timeout?: number },
): string {
  const timeout = options?.timeout ?? 30_000;
  const escaped = command.replace(/'/g, "'\\''");
  const fullCommand = `ssh ${conf.server} '${escaped}'`;
  return execSync(fullCommand, {
    encoding: "utf-8",
    timeout,
    stdio: ["pipe", "pipe", "pipe"],
  }).trim();
}

export function rsync(
  source: string,
  dest: string,
  options?: { ignoreExisting?: boolean; dryRun?: boolean },
): { stdout: string; transferred: number } {
  const args = ["-az", "--stats"];
  if (options?.ignoreExisting) args.push("--ignore-existing");
  if (options?.dryRun) args.push("--dry-run");

  const cmd = `rsync ${args.join(" ")} ${source} ${dest}`;
  const stdout = execSync(cmd, {
    encoding: "utf-8",
    timeout: 600_000,
    maxBuffer: 50 * 1024 * 1024,
    stdio: ["pipe", "pipe", "pipe"],
  });

  const match = stdout.match(
    /Number of regular files transferred:\s+(\d[\d,]*)/,
  );
  const transferred = match
    ? Number.parseInt(match[1].replace(/,/g, ""), 10)
    : 0;

  return { stdout, transferred };
}
