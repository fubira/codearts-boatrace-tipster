/** Rate limiting: sleep between pages within the same venue */
export const COOLDOWN_BETWEEN_PAGES_MS = 100;

/** Rate limiting: sleep between venue-day groups */
export const COOLDOWN_BETWEEN_VENUES_MS = 1000;

/** Maximum number of races per venue per day */
export const MAX_RACES_PER_VENUE = 12;

/** Maximum concurrent venue scraping workers */
export const MAX_CONCURRENCY = 8;

/** Stadium codes and names (all 24 venues) */
export const STADIUMS: Record<string, string> = {
  "01": "桐生",
  "02": "戸田",
  "03": "江戸川",
  "04": "平和島",
  "05": "多摩川",
  "06": "浜名湖",
  "07": "蒲郡",
  "08": "常滑",
  "09": "津",
  "10": "三国",
  "11": "びわこ",
  "12": "住之江",
  "13": "尼崎",
  "14": "鳴門",
  "15": "丸亀",
  "16": "児島",
  "17": "宮島",
  "18": "徳山",
  "19": "下関",
  "20": "若松",
  "21": "芦屋",
  "22": "福岡",
  "23": "唐津",
  "24": "大村",
};

/** Stadium prefectures */
export const STADIUM_PREFECTURES: Record<string, string> = {
  "01": "群馬県",
  "02": "埼玉県",
  "03": "東京都",
  "04": "東京都",
  "05": "東京都",
  "06": "静岡県",
  "07": "愛知県",
  "08": "愛知県",
  "09": "三重県",
  "10": "福井県",
  "11": "滋賀県",
  "12": "大阪府",
  "13": "兵庫県",
  "14": "徳島県",
  "15": "香川県",
  "16": "岡山県",
  "17": "広島県",
  "18": "山口県",
  "19": "山口県",
  "20": "福岡県",
  "21": "福岡県",
  "22": "福岡県",
  "23": "佐賀県",
  "24": "長崎県",
};

export interface RaceParams {
  raceNumber: number;
  stadiumCode: string;
  date: string;
}

/** /race/racelist?rno=1&jcd=04&hd=20250115 */
export function raceListUrl(params: RaceParams): string {
  return `/owpc/pc/race/racelist?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /race/beforeinfo?rno=1&jcd=04&hd=20250115 */
export function beforeInfoUrl(params: RaceParams): string {
  return `/owpc/pc/race/beforeinfo?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /race/raceresult?rno=1&jcd=04&hd=20250115 */
export function raceResultUrl(params: RaceParams): string {
  return `/owpc/pc/race/raceresult?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /race/oddstf?rno=1&jcd=04&hd=20250115 — 単勝・複勝オッズ */
export function oddsTfUrl(params: RaceParams): string {
  return `/owpc/pc/race/oddstf?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /race/odds2tf?rno=1&jcd=04&hd=20250115 — 2連単・2連複オッズ */
export function odds2TfUrl(params: RaceParams): string {
  return `/owpc/pc/race/odds2tf?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /race/odds3t?rno=1&jcd=04&hd=20250115 — 3連単オッズ */
export function odds3TUrl(params: RaceParams): string {
  return `/owpc/pc/race/odds3t?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /race/odds3f?rno=1&jcd=04&hd=20250115 — 3連複オッズ */
export function odds3FUrl(params: RaceParams): string {
  return `/owpc/pc/race/odds3f?rno=${params.raceNumber}&jcd=${params.stadiumCode}&hd=${params.date}`;
}

/** /owpc/pc/race/index?hd=YYYYMMDD */
export function dailyScheduleUrl(date: string): string {
  return `/owpc/pc/race/index?hd=${date}`;
}

/** /owpc/pc/race/index?hd=YYYYMM01 — monthly schedule uses first day of month */
export function monthlyScheduleUrl(yearMonth: string): string {
  return `/owpc/pc/race/index?hd=${yearMonth}01`;
}
