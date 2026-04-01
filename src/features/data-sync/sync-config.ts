export interface SyncConfig {
  server: string;
  prodDir: string;
}

export function loadSyncConfig(): SyncConfig {
  const server = process.env.PRODUCT_SERVER;
  const prodDir = process.env.PRODUCT_DIR;

  if (!server || !prodDir) {
    throw new Error(
      "Missing environment variables.\n" +
        "Set PRODUCT_SERVER and PRODUCT_DIR (absolute path):\n" +
        "  export PRODUCT_SERVER=<hostname>\n" +
        "  export PRODUCT_DIR=<absolute path to project>",
    );
  }

  return { server, prodDir: prodDir.replace(/\/+$/, "") };
}
