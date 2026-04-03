/**
 * テレボート画面のセレクタ定義
 *
 * TODO: テレボート SP 版の画面を手動調査して埋める
 */

export const TELEBOAT_URL = "https://tb.teleboat.jp/";

export const LOGIN_SELECTORS = {
  subscriberNumber: "", // TODO
  pin: "", // TODO
  password: "", // TODO
  loginButton: "", // TODO
} as const;

export const MENU_SELECTORS = {
  balance: "", // TODO
  purchaseMenu: "", // TODO
} as const;

export const STADIUM_SELECTORS = {
  stadiumList: "", // TODO
  stadiumItem: "", // TODO
} as const;

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
