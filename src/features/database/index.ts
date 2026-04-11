export { getDatabase, initializeDatabase, closeDatabase } from "./client";
export { SCHEMA_VERSION, CREATE_TABLES_SQL, MIGRATIONS } from "./schema";
export {
  saveRaces,
  saveRaceResults,
  saveBeforeInfo,
  saveOdds,
  saveOddsSnapshot,
  loadSnapshotWinProbs,
  loadSnapshotTrifectaOdds,
  savePurchaseRecord,
  isRaceScraped,
  updateDeadline,
} from "./storage";
export type {
  RaceData,
  RaceEntryData,
  RaceResultData,
  RaceResultEntry,
  PayoutData,
  BeforeInfoData,
  BeforeInfoEntry,
  SaveResult,
  OddsData,
  PurchaseRecordData,
  OddsTiming,
} from "./storage";
