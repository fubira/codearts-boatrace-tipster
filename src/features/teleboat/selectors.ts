/**
 * テレボート画面のセレクタ定義（PC版: ib.mbrace.or.jp）
 *
 * SPA 構造（jQuery + jsrender）。会場クリック→フォーム送信で画面遷移。
 */

export const TELEBOAT_URL = "https://ib.mbrace.or.jp/";

/** ログイン後のトップURL */
export const TOP_URL_PATTERN = "**/service/bet/top**";

export const LOGIN_SELECTORS = {
  subscriberNumber: "#memberNo",
  pin: "#pin",
  password: "#authPassword",
  loginButton: "#loginButton",
} as const;

export const MENU_SELECTORS = {
  /** 購入限度額の表示要素 */
  balance: "#currentBetLimitAmount",
  /** 残高更新ボタン */
  balanceReload: "#updateBalanceBtn a",
} as const;

export const STADIUM_SELECTORS = {
  /** 会場一覧コンテナ */
  stadiumList: "#jyoInfos .selectBox",
  /**
   * 会場アイテム。id="jyo{code}" (例: jyo18 = 徳山)
   * クリックすると todayForm が submit される
   */
  stadiumItem: "#jyoInfos .selectBox li",
  /** 会場IDプレフィックス — `#jyo${stadiumCode}` で特定 */
  stadiumIdPrefix: "#jyo",
} as const;

/** 投票画面（会場選択後）のセレクタ — TODO: 実画面で調査 */
export const RACE_SELECTORS = {
  raceList: "", // TODO
  raceItem: "", // TODO
  raceNum: "", // TODO
} as const;

export const BET_SELECTORS = {
  tanshoTab: "", // TODO
  boatNumber: "", // TODO
  amountInput: "", // TODO
  setButton: "", // TODO
} as const;

export const CONFIRM_SELECTORS = {
  voteList: "", // TODO
  courseRace: "", // TODO
  boatCombi: "", // TODO
  betStyle: "", // TODO
  totalInput: "", // TODO
  submitButton: "", // TODO
  cancelButton: "", // TODO: dry-run 時に使用
} as const;

/**
 * 会場コード → 会場名マッピング（BOAT.code から抽出）
 * runner の STADIUMS と同じマッピング
 */
export const STADIUM_CODES: Record<string, string> = {
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
