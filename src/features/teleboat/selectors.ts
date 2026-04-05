/**
 * テレボート画面のセレクタ定義（PC版: ib.mbrace.or.jp）
 *
 * SPA 構造（jQuery + jsrender）。会場クリック→フォーム送信で画面遷移。
 * reCAPTCHA Enterprise が有効（invisible badge）。
 */

export const TELEBOAT_URL = "https://ib.mbrace.or.jp/";

/** ログイン後のトップURL */
export const TOP_URL_PATTERN = "**/service/bet/top**";

/** 投票画面URL */
export const BET_URL_PATTERN = "**/service/bet/betcom/**";

/** 投票確認画面URL */
export const BETCONF_URL_PATTERN = "**/service/bet/betconf**";

export const LOGIN_SELECTORS = {
  subscriberNumber: "#memberNo",
  pin: "#pin",
  password: "#authPassword",
  loginButton: "#loginButton",
  /** ログイン後に表示される「特別なお知らせ」モーダルの閉じるボタン */
  noticeCloseButton: "#newsoverviewdispCloseButton",
  /** 「すべて既読にする」チェックボックス */
  noticeAllRead: "#isAllread",
} as const;

export const MENU_SELECTORS = {
  /** 購入限度額の表示要素 */
  balance: "#currentBetLimitAmount",
  /** 残高更新ボタン */
  balanceReload: "#updateBalanceBtn a",
  /** 入金メニュー */
  charge: "#charge",
} as const;

/**
 * 入金モーダル
 *
 * フロー: メニュー「入金する」→ 金額入力 → パスワード入力 → 入金実行 → 完了
 */
export const CHARGE_SELECTORS = {
  /** 入金金額入力（千円単位 — 入力値 × 1000 = 実金額） */
  amountInput: "#chargeInstructAmt",
  /** 投票用パスワード入力 */
  betPassword: "#chargeBetPassword",
  /** 入金実行ボタン */
  executeButton: "#executeCharge",
  /** キャンセルボタン */
  cancelButton: "#closeCharge",
  /** 入金完了画面の閉じるボタン */
  closeCompButton: "#closeChargecomp",
} as const;

export const STADIUM_SELECTORS = {
  /** 会場一覧コンテナ */
  stadiumList: "#jyoInfos .selectBox",
  /** 会場アイテム。id="jyo{code}" (例: jyo18 = 徳山) */
  stadiumItem: "#jyoInfos .selectBox li",
  /** 会場IDプレフィックス — `#jyo${stadiumCode}` で特定 */
  stadiumIdPrefix: "#jyo",
} as const;

/**
 * レース選択（投票画面上部）
 * 会場クリック後に表示される。レースタブで切り替え。
 */
export const RACE_SELECTORS = {
  /** レースタブ — `#selRaceNo{nn}` (例: #selRaceNo01 = 1R) */
  raceTabPrefix: "#selRaceNo",
  /** 現在選択中のレースタブ */
  currentRace: "#raceSelection li.current",
} as const;

/**
 * 投票操作（賭式・艇番・金額）
 *
 * フロー: 賭式タブ → 投票方式タブ → 艇番クリック → 金額入力 → ベットリスト追加
 */
export const BET_SELECTORS = {
  /** 単勝タブ (kachishiki=1) */
  tanshoTab: "#betkati1",
  /** 3連単タブ (kachishiki=6) */
  sanrentanTab: "#betkati6",

  /** 通常投票タブ */
  normalBetWay: "#betway1",
  /** フォーメーション投票タブ */
  formationBetWay: "#betway4",

  /**
   * 通常投票の艇番ボタン — `#regbtn_{boatNumber}_{column}`
   * 単勝: column=1（1着列のみ）
   */
  normalBoatPrefix: "#regbtn_",
  /**
   * フォーメーション投票の艇番セル — `td.combiForma.x{boatNumber}.y{column}`
   * column: y1=1着, y2=2着, y3=3着
   * 選択済みで `.check` クラスが付与される
   */
  formationBoatCell: "td.combiForma",

  /** 購入金額入力欄（100円単位 — 入力値 × 100 = 実金額） */
  amountInput: "#amount",
  /** ベットリストに追加（通常投票） */
  normalAddToBetList: "#regAmountBtn",
  /** ベットリストに追加（フォーメーション投票） */
  formationAddToBetList: "#formaAmountBtn",
  /** フォーメーションのベット数表示 */
  formationBetCount: "#combiCount",
} as const;

/**
 * ベットリスト（画面右側）
 * ベット追加後に表示。投票入力完了で確認画面へ。
 */
export const BETLIST_SELECTORS = {
  /** 合計ベット数 */
  totalBetCount: ".inputCompletion .betNumber strong",
  /** 総購入金額 */
  totalAmount: "#totalAmount",
  /** 投票入力完了ボタン */
  submitButton: ".btnSubmit a",
  /** 全削除ボタン */
  allRemoveButton: ".betlistbtn.allremove",
} as const;

/**
 * 投票確認画面（/service/bet/betconf）
 * TODO: 確認画面の HTML を調査して埋める
 */
export const CONFIRM_SELECTORS = {
  /** 投票内容リスト */
  voteList: "", // TODO
  /** 投票用パスワード入力 */
  betPassword: "", // TODO
  /** 投票ボタン */
  submitButton: "", // TODO
  /** キャンセルボタン — dry-run 時に使用 */
  cancelButton: "", // TODO
} as const;

/**
 * 会場コード → 会場名マッピング（BOAT.code から抽出）
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
