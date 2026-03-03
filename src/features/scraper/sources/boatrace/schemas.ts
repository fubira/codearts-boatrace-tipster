/** Zod validation schemas for parser output */

import { z } from "zod";

export const raceEntrySchema = z.object({
  racerId: z.number().int().positive(),
  boatNumber: z.number().int().min(1).max(6),
  racerName: z.string().min(1),
  racerClass: z.string().optional(),
  racerWeight: z.number().positive().optional(),
  flyingCount: z.number().int().min(0).optional(),
  lateCount: z.number().int().min(0).optional(),
  averageSt: z.number().optional(),
  nationalWinRate: z.number().min(0).max(100).optional(),
  nationalTop2Rate: z.number().min(0).max(100).optional(),
  nationalTop3Rate: z.number().min(0).max(100).optional(),
  localWinRate: z.number().min(0).max(100).optional(),
  localTop2Rate: z.number().min(0).max(100).optional(),
  localTop3Rate: z.number().min(0).max(100).optional(),
  motorNumber: z.number().int().positive().optional(),
  motorTop2Rate: z.number().min(0).max(100).optional(),
  motorTop3Rate: z.number().min(0).max(100).optional(),
  boatNumberAssigned: z.number().int().positive().optional(),
  boatTop2Rate: z.number().min(0).max(100).optional(),
  boatTop3Rate: z.number().min(0).max(100).optional(),
  branch: z.string().optional(),
  birthplace: z.string().optional(),
  birthDate: z.string().optional(),
});

export const raceDataSchema = z.object({
  stadiumId: z.number().int().min(1).max(24),
  stadiumName: z.string().min(1),
  stadiumPrefecture: z.string().optional(),
  raceDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  raceNumber: z.number().int().min(1).max(12),
  raceTitle: z.string().optional(),
  raceGrade: z.string().optional(),
  distance: z.number().int().positive().optional(),
  deadline: z.string().optional(),
  entries: z.array(raceEntrySchema).min(1).max(6),
});

export const raceResultEntrySchema = z.object({
  boatNumber: z.number().int().min(1).max(6),
  courseNumber: z.number().int().min(1).max(6).optional(),
  startTiming: z.number().optional(),
  finishPosition: z.number().int().min(1).max(6).optional(),
  raceTime: z.string().optional(),
});

export const payoutSchema = z.object({
  betType: z.string().min(1),
  combination: z.string().min(1),
  payout: z.number().int().positive(),
});

export const raceResultDataSchema = z.object({
  stadiumId: z.number().int().min(1).max(24),
  raceDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  raceNumber: z.number().int().min(1).max(12),
  weather: z.string().optional(),
  windSpeed: z.number().int().min(0).optional(),
  windDirection: z.number().int().optional(),
  waveHeight: z.number().int().min(0).optional(),
  temperature: z.number().optional(),
  waterTemperature: z.number().optional(),
  technique: z.string().optional(),
  entries: z.array(raceResultEntrySchema).min(1).max(6),
  payouts: z.array(payoutSchema).optional(),
});

export const beforeInfoEntrySchema = z.object({
  boatNumber: z.number().int().min(1).max(6),
  exhibitionTime: z.number().positive().optional(),
  tilt: z.number().optional(),
  exhibitionSt: z.number().optional(),
  stabilizer: z.boolean().optional(),
  partsReplaced: z.array(z.string()).optional(),
});

export const beforeInfoDataSchema = z.object({
  stadiumId: z.number().int().min(1).max(24),
  raceDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  raceNumber: z.number().int().min(1).max(12),
  entries: z.array(beforeInfoEntrySchema).min(1).max(6),
});
