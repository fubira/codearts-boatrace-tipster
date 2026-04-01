import { closeDatabase, initializeDatabase } from "@/features/database";
import { checkIntegrity } from "@/features/database/integrity";
import { Command } from "commander";

export const verifyCommand = new Command("verify")
  .description("Verify database integrity")
  .action(() => {
    initializeDatabase();

    try {
      const results = checkIntegrity();
      let hasError = false;
      let hasWarn = false;

      for (const r of results) {
        const icon = r.status === "ok" ? "✓" : r.status === "warn" ? "!" : "✗";
        const color =
          r.status === "ok"
            ? "\x1b[32m"
            : r.status === "warn"
              ? "\x1b[33m"
              : "\x1b[31m";
        console.log(`${color}  ${icon} ${r.name}\x1b[0m: ${r.detail}`);

        if (r.status === "error") hasError = true;
        if (r.status === "warn") hasWarn = true;
      }

      console.log();
      if (hasError) {
        console.log(
          "\x1b[31mErrors found. Database may have integrity issues.\x1b[0m",
        );
        process.exit(1);
      } else if (hasWarn) {
        console.log("\x1b[33mWarnings found. Review above for details.\x1b[0m");
      } else {
        console.log("\x1b[32mAll checks passed.\x1b[0m");
      }
    } finally {
      closeDatabase();
    }
  });
