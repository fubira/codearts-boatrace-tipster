import { runScrapeDaemon } from "@/features/scraper/scrape-daemon";
import { Command } from "commander";

export const scrapeDaemonCommand = new Command("scrape-daemon")
  .description("Run scraping daemon (data collection only, no prediction)")
  .action(async () => {
    await runScrapeDaemon();
  });
