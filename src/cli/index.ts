#!/usr/bin/env bun
import { analyzeCommand } from "@/cli/commands/analyze";
import { backupCommand } from "@/cli/commands/backup";
import { dataCommand } from "@/cli/commands/data";
import { predictCommand } from "@/cli/commands/predict";
import { scrapeCommand } from "@/cli/commands/scrape";
import { scrapeOddsCommand } from "@/cli/commands/scrape-odds";
import { setLogLevel } from "@/shared/logger";
import { Command } from "commander";

const program = new Command();

program
  .name("boatrace-tipster")
  .description("Boat race prediction AI powered by machine learning")
  .version("0.1.0")
  .option("-v, --verbose", "enable verbose logging")
  .hook("preAction", (thisCommand) => {
    if (thisCommand.opts().verbose) {
      setLogLevel("debug");
    }
  });

program.addCommand(scrapeCommand);
program.addCommand(scrapeOddsCommand);
program.addCommand(predictCommand);
program.addCommand(analyzeCommand);
program.addCommand(dataCommand);
program.addCommand(backupCommand);

program.parse();
