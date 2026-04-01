import type { Database } from "bun:sqlite";
import { logger } from "@/shared/logger";
import { getDatabase } from "./client";

// --- Input types ---

export interface RaceEntryData {
  racerId: number;
  boatNumber: number;
  racerName: string;
  racerClass?: string;
  racerWeight?: number;
  flyingCount?: number;
  lateCount?: number;
  averageSt?: number;
  nationalWinRate?: number;
  nationalTop2Rate?: number;
  nationalTop3Rate?: number;
  localWinRate?: number;
  localTop2Rate?: number;
  localTop3Rate?: number;
  motorNumber?: number;
  motorTop2Rate?: number;
  motorTop3Rate?: number;
  boatNumberAssigned?: number;
  boatTop2Rate?: number;
  boatTop3Rate?: number;
  branch?: string;
  birthplace?: string;
  birthDate?: string;
}

export interface RaceData {
  stadiumId: number;
  stadiumName: string;
  stadiumPrefecture?: string;
  raceDate: string;
  raceNumber: number;
  raceTitle?: string;
  raceGrade?: string;
  distance?: number;
  deadline?: string;
  entries: RaceEntryData[];
}

export interface RaceResultEntry {
  boatNumber: number;
  courseNumber?: number;
  startTiming?: number;
  finishPosition?: number;
  raceTime?: string;
}

export interface RaceResultData {
  stadiumId: number;
  raceDate: string;
  raceNumber: number;
  weather?: string;
  windSpeed?: number;
  windDirection?: number;
  waveHeight?: number;
  temperature?: number;
  waterTemperature?: number;
  technique?: string;
  entries: RaceResultEntry[];
  payouts?: PayoutData[];
}

export interface PayoutData {
  betType: string;
  combination: string;
  payout: number;
}

export interface BeforeInfoEntry {
  boatNumber: number;
  exhibitionTime?: number;
  tilt?: number;
  exhibitionSt?: number;
  stabilizer?: boolean;
  partsReplaced?: string[];
}

export interface BeforeInfoData {
  stadiumId: number;
  raceDate: string;
  raceNumber: number;
  entries: BeforeInfoEntry[];
}

export interface RacerCourseStatsData {
  racerId: number;
  courseNumber: number;
  winRate?: number;
  top2Rate?: number;
  top3Rate?: number;
}

// --- Save result ---

export interface SaveResult {
  racesUpserted: number;
  entriesUpserted: number;
}

// --- Internal helpers ---

function upsertStadium(
  db: Database,
  id: number,
  name: string,
  prefecture?: string,
): void {
  db.query(
    `INSERT INTO stadiums (id, name, prefecture) VALUES ($id, $name, $prefecture)
     ON CONFLICT (id) DO UPDATE SET name = excluded.name, prefecture = COALESCE(excluded.prefecture, stadiums.prefecture)`,
  ).run({ $id: id, $name: name, $prefecture: prefecture ?? null });
}

function upsertRacer(
  db: Database,
  racer: {
    id: number;
    name: string;
    branch?: string;
    birthplace?: string;
    birthDate?: string;
    class?: string;
  },
): void {
  db.query(
    `INSERT INTO racers (id, name, branch, birthplace, birth_date, class)
     VALUES ($id, $name, $branch, $birthplace, $birthDate, $class)
     ON CONFLICT (id) DO UPDATE SET
       name = excluded.name,
       branch = COALESCE(excluded.branch, racers.branch),
       birthplace = COALESCE(excluded.birthplace, racers.birthplace),
       birth_date = COALESCE(excluded.birth_date, racers.birth_date),
       class = COALESCE(excluded.class, racers.class)`,
  ).run({
    $id: racer.id,
    $name: racer.name,
    $branch: racer.branch ?? null,
    $birthplace: racer.birthplace ?? null,
    $birthDate: racer.birthDate ?? null,
    $class: racer.class ?? null,
  });
}

function upsertRace(
  db: Database,
  race: {
    stadiumId: number;
    raceDate: string;
    raceNumber: number;
    raceTitle?: string;
    raceGrade?: string;
    distance?: number;
    deadline?: string;
  },
): number {
  return (
    db
      .query(
        `INSERT INTO races (stadium_id, race_date, race_number, race_title, race_grade, distance, deadline)
         VALUES ($stadiumId, $raceDate, $raceNumber, $raceTitle, $raceGrade, $distance, $deadline)
         ON CONFLICT (stadium_id, race_date, race_number) DO UPDATE SET
           race_title = COALESCE(excluded.race_title, races.race_title),
           race_grade = COALESCE(excluded.race_grade, races.race_grade),
           distance = excluded.distance,
           deadline = COALESCE(excluded.deadline, races.deadline)
         RETURNING id`,
      )
      .get({
        $stadiumId: race.stadiumId,
        $raceDate: race.raceDate,
        $raceNumber: race.raceNumber,
        $raceTitle: race.raceTitle ?? null,
        $raceGrade: race.raceGrade ?? null,
        $distance: race.distance ?? 1800,
        $deadline: race.deadline ?? null,
      }) as { id: number }
  ).id;
}

function upsertRaceEntry(
  db: Database,
  raceId: number,
  entry: RaceEntryData,
): void {
  db.query(
    `INSERT INTO race_entries (race_id, racer_id, boat_number, racer_class, racer_weight,
       flying_count, late_count, average_st,
       national_win_rate, national_top2_rate, national_top3_rate,
       local_win_rate, local_top2_rate, local_top3_rate,
       motor_number, motor_top2_rate, motor_top3_rate,
       boat_number_assigned, boat_top2_rate, boat_top3_rate)
     VALUES ($raceId, $racerId, $boatNumber, $racerClass, $racerWeight,
       $flyingCount, $lateCount, $averageSt,
       $nationalWinRate, $nationalTop2Rate, $nationalTop3Rate,
       $localWinRate, $localTop2Rate, $localTop3Rate,
       $motorNumber, $motorTop2Rate, $motorTop3Rate,
       $boatNumberAssigned, $boatTop2Rate, $boatTop3Rate)
     ON CONFLICT (race_id, boat_number) DO UPDATE SET
       racer_id = excluded.racer_id,
       racer_class = COALESCE(excluded.racer_class, race_entries.racer_class),
       racer_weight = COALESCE(excluded.racer_weight, race_entries.racer_weight),
       flying_count = COALESCE(excluded.flying_count, race_entries.flying_count),
       late_count = COALESCE(excluded.late_count, race_entries.late_count),
       average_st = COALESCE(excluded.average_st, race_entries.average_st),
       national_win_rate = COALESCE(excluded.national_win_rate, race_entries.national_win_rate),
       national_top2_rate = COALESCE(excluded.national_top2_rate, race_entries.national_top2_rate),
       national_top3_rate = COALESCE(excluded.national_top3_rate, race_entries.national_top3_rate),
       local_win_rate = COALESCE(excluded.local_win_rate, race_entries.local_win_rate),
       local_top2_rate = COALESCE(excluded.local_top2_rate, race_entries.local_top2_rate),
       local_top3_rate = COALESCE(excluded.local_top3_rate, race_entries.local_top3_rate),
       motor_number = COALESCE(excluded.motor_number, race_entries.motor_number),
       motor_top2_rate = COALESCE(excluded.motor_top2_rate, race_entries.motor_top2_rate),
       motor_top3_rate = COALESCE(excluded.motor_top3_rate, race_entries.motor_top3_rate),
       boat_number_assigned = COALESCE(excluded.boat_number_assigned, race_entries.boat_number_assigned),
       boat_top2_rate = COALESCE(excluded.boat_top2_rate, race_entries.boat_top2_rate),
       boat_top3_rate = COALESCE(excluded.boat_top3_rate, race_entries.boat_top3_rate)`,
  ).run({
    $raceId: raceId,
    $racerId: entry.racerId,
    $boatNumber: entry.boatNumber,
    $racerClass: entry.racerClass ?? null,
    $racerWeight: entry.racerWeight ?? null,
    $flyingCount: entry.flyingCount ?? null,
    $lateCount: entry.lateCount ?? null,
    $averageSt: entry.averageSt ?? null,
    $nationalWinRate: entry.nationalWinRate ?? null,
    $nationalTop2Rate: entry.nationalTop2Rate ?? null,
    $nationalTop3Rate: entry.nationalTop3Rate ?? null,
    $localWinRate: entry.localWinRate ?? null,
    $localTop2Rate: entry.localTop2Rate ?? null,
    $localTop3Rate: entry.localTop3Rate ?? null,
    $motorNumber: entry.motorNumber ?? null,
    $motorTop2Rate: entry.motorTop2Rate ?? null,
    $motorTop3Rate: entry.motorTop3Rate ?? null,
    $boatNumberAssigned: entry.boatNumberAssigned ?? null,
    $boatTop2Rate: entry.boatTop2Rate ?? null,
    $boatTop3Rate: entry.boatTop3Rate ?? null,
  });
}

// --- Public API ---

/** Save race program data (stadiums, racers, races, entries) */
export function saveRaces(races: RaceData[], db?: Database): SaveResult {
  const database = db ?? getDatabase();
  const stats: SaveResult = { racesUpserted: 0, entriesUpserted: 0 };

  const transaction = database.transaction(() => {
    for (const race of races) {
      upsertStadium(
        database,
        race.stadiumId,
        race.stadiumName,
        race.stadiumPrefecture,
      );

      const raceId = upsertRace(database, race);
      stats.racesUpserted++;

      for (const entry of race.entries) {
        upsertRacer(database, {
          id: entry.racerId,
          name: entry.racerName,
          branch: entry.branch,
          birthplace: entry.birthplace,
          birthDate: entry.birthDate,
          class: entry.racerClass,
        });
        upsertRaceEntry(database, raceId, entry);
        stats.entriesUpserted++;
      }
    }
  });

  transaction();
  logger.info(
    `Saved: ${stats.racesUpserted} race(s), ${stats.entriesUpserted} entries`,
  );
  return stats;
}

/** Update race results (finish positions, start timings, conditions, payouts) */
export function saveRaceResults(
  results: RaceResultData[],
  db?: Database,
): void {
  const database = db ?? getDatabase();

  const transaction = database.transaction(() => {
    for (const result of results) {
      const race = database
        .query(
          `SELECT id FROM races
           WHERE stadium_id = $stadiumId AND race_date = $raceDate AND race_number = $raceNumber`,
        )
        .get({
          $stadiumId: result.stadiumId,
          $raceDate: result.raceDate,
          $raceNumber: result.raceNumber,
        }) as { id: number } | null;

      if (!race) {
        logger.warn(
          `Race not found: stadium=${result.stadiumId} date=${result.raceDate} R${result.raceNumber}`,
        );
        continue;
      }

      database
        .query(
          `UPDATE races SET
             weather = COALESCE($weather, weather),
             wind_speed = COALESCE($windSpeed, wind_speed),
             wind_direction = COALESCE($windDirection, wind_direction),
             wave_height = COALESCE($waveHeight, wave_height),
             temperature = COALESCE($temperature, temperature),
             water_temperature = COALESCE($waterTemperature, water_temperature),
             technique = COALESCE($technique, technique)
           WHERE id = $id`,
        )
        .run({
          $id: race.id,
          $weather: result.weather ?? null,
          $windSpeed: result.windSpeed ?? null,
          $windDirection: result.windDirection ?? null,
          $waveHeight: result.waveHeight ?? null,
          $temperature: result.temperature ?? null,
          $waterTemperature: result.waterTemperature ?? null,
          $technique: result.technique ?? null,
        });

      for (const entry of result.entries) {
        database
          .query(
            `UPDATE race_entries SET
               course_number = COALESCE($courseNumber, course_number),
               start_timing = COALESCE($startTiming, start_timing),
               finish_position = COALESCE($finishPosition, finish_position),
               race_time = COALESCE($raceTime, race_time)
             WHERE race_id = $raceId AND boat_number = $boatNumber`,
          )
          .run({
            $raceId: race.id,
            $boatNumber: entry.boatNumber,
            $courseNumber: entry.courseNumber ?? null,
            $startTiming: entry.startTiming ?? null,
            $finishPosition: entry.finishPosition ?? null,
            $raceTime: entry.raceTime ?? null,
          });
      }

      if (result.payouts && result.payouts.length > 0) {
        database
          .query("DELETE FROM race_payouts WHERE race_id = $raceId")
          .run({ $raceId: race.id });

        for (const payout of result.payouts) {
          database
            .query(
              `INSERT INTO race_payouts (race_id, bet_type, combination, payout)
               VALUES ($raceId, $betType, $combination, $payout)`,
            )
            .run({
              $raceId: race.id,
              $betType: payout.betType,
              $combination: payout.combination,
              $payout: payout.payout,
            });
        }
      }
    }
  });

  transaction();
  logger.info(`Updated results for ${results.length} race(s)`);
}

/** Update pre-race info (exhibition times, tilt, parts replacement) */
export function saveBeforeInfo(
  beforeInfoList: BeforeInfoData[],
  db?: Database,
): void {
  const database = db ?? getDatabase();

  const transaction = database.transaction(() => {
    for (const info of beforeInfoList) {
      const race = database
        .query(
          `SELECT id FROM races
           WHERE stadium_id = $stadiumId AND race_date = $raceDate AND race_number = $raceNumber`,
        )
        .get({
          $stadiumId: info.stadiumId,
          $raceDate: info.raceDate,
          $raceNumber: info.raceNumber,
        }) as { id: number } | null;

      if (!race) {
        logger.warn(
          `Race not found: stadium=${info.stadiumId} date=${info.raceDate} R${info.raceNumber}`,
        );
        continue;
      }

      for (const entry of info.entries) {
        database
          .query(
            `UPDATE race_entries SET
               exhibition_time = COALESCE($exhibitionTime, exhibition_time),
               tilt = COALESCE($tilt, tilt),
               exhibition_st = COALESCE($exhibitionSt, exhibition_st),
               stabilizer = $stabilizer,
               parts_replaced = COALESCE($partsReplaced, parts_replaced)
             WHERE race_id = $raceId AND boat_number = $boatNumber`,
          )
          .run({
            $raceId: race.id,
            $boatNumber: entry.boatNumber,
            $exhibitionTime: entry.exhibitionTime ?? null,
            $tilt: entry.tilt ?? null,
            $exhibitionSt: entry.exhibitionSt ?? null,
            $stabilizer: entry.stabilizer ? 1 : 0,
            $partsReplaced: entry.partsReplaced
              ? JSON.stringify(entry.partsReplaced)
              : null,
          });
      }
    }
  });

  transaction();
  logger.info(`Updated before-info for ${beforeInfoList.length} race(s)`);
}

/** Upsert racer course statistics */
export function saveRacerCourseStats(
  stats: RacerCourseStatsData[],
  db?: Database,
): void {
  const database = db ?? getDatabase();

  const transaction = database.transaction(() => {
    for (const stat of stats) {
      database
        .query(
          `INSERT INTO racer_course_stats (racer_id, course_number, win_rate, top2_rate, top3_rate, updated_at)
           VALUES ($racerId, $courseNumber, $winRate, $top2Rate, $top3Rate, datetime('now'))
           ON CONFLICT (racer_id, course_number) DO UPDATE SET
             win_rate = COALESCE(excluded.win_rate, racer_course_stats.win_rate),
             top2_rate = COALESCE(excluded.top2_rate, racer_course_stats.top2_rate),
             top3_rate = COALESCE(excluded.top3_rate, racer_course_stats.top3_rate),
             updated_at = datetime('now')`,
        )
        .run({
          $racerId: stat.racerId,
          $courseNumber: stat.courseNumber,
          $winRate: stat.winRate ?? null,
          $top2Rate: stat.top2Rate ?? null,
          $top3Rate: stat.top3Rate ?? null,
        });
    }
  });

  transaction();
  logger.info(`Saved ${stats.length} racer course stats`);
}

export interface OddsData {
  stadiumId: number;
  raceDate: string;
  raceNumber: number;
  entries: { betType: string; combination: string; odds: number }[];
}

/** Save race odds (delete + re-insert per race) */
export function saveOdds(oddsList: OddsData[], db?: Database): void {
  const database = db ?? getDatabase();

  const transaction = database.transaction(() => {
    for (const odds of oddsList) {
      const race = database
        .query(
          `SELECT id FROM races
           WHERE stadium_id = $stadiumId AND race_date = $raceDate AND race_number = $raceNumber`,
        )
        .get({
          $stadiumId: odds.stadiumId,
          $raceDate: odds.raceDate,
          $raceNumber: odds.raceNumber,
        }) as { id: number } | null;

      if (!race) continue;

      database
        .query("DELETE FROM race_odds WHERE race_id = $raceId")
        .run({ $raceId: race.id });

      for (const entry of odds.entries) {
        database
          .query(
            `INSERT INTO race_odds (race_id, bet_type, combination, odds)
             VALUES ($raceId, $betType, $combination, $odds)
             ON CONFLICT (race_id, bet_type, combination) DO UPDATE SET odds = excluded.odds`,
          )
          .run({
            $raceId: race.id,
            $betType: entry.betType,
            $combination: entry.combination,
            $odds: entry.odds,
          });
      }
    }
  });

  transaction();
  logger.info(`Saved odds for ${oddsList.length} race(s)`);
}

/** Check if a race has already been scraped (has finish_position data) */
export function isRaceScraped(
  stadiumId: number,
  raceDate: string,
  raceNumber: number,
  db?: Database,
): boolean {
  const database = db ?? getDatabase();
  const row = database
    .query(
      `SELECT re.finish_position FROM race_entries re
       JOIN races r ON r.id = re.race_id
       WHERE r.stadium_id = $stadiumId AND r.race_date = $raceDate AND r.race_number = $raceNumber
       AND re.finish_position IS NOT NULL
       LIMIT 1`,
    )
    .get({
      $stadiumId: stadiumId,
      $raceDate: raceDate,
      $raceNumber: raceNumber,
    });
  return row !== null;
}
