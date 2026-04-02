/** Python command builder via uv. Works in both Docker and development. */

import { resolve } from "node:path";
import { config } from "./config";

export function pythonCommand(
  scriptModule: string,
  args: string[],
): { cmd: string[]; cwd: string } {
  const mlDir = resolve(config.projectRoot, "ml");
  return {
    cmd: [
      "uv",
      "run",
      "--directory",
      mlDir,
      "python",
      "-m",
      scriptModule,
      ...args,
    ],
    cwd: mlDir,
  };
}
