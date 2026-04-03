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
 * フロー: 単勝タブ → 艇番クリック → 金額入力 → ベットリスト追加
 */
export const BET_SELECTORS = {
  /** 単勝タブ (kachishiki=1) */
  tanshoTab: "#betkati1",
  /**
   * 艇番ボタン — `#regbtn_{boatNumber}_{column}`
   * 単勝の場合 column=1（1着列のみ）
   * 選択済みで `.check` クラスが付与される
   */
  boatButtonPrefix: "#regbtn_",
  /** 購入金額入力欄（100円単位 — 入力値 × 100 = 実金額） */
  amountInput: "#amount",
  /** ベットリストに追加ボタン */
  addToBetList: "#regAmountBtn",
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
