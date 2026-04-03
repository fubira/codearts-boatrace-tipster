/** テレボートログイン認証情報 */
export interface TelebotCredentials {
  /** 加入者番号 */
  subscriberNumber: string;
  /** 暗証番号 */
  pin: string;
  /** 認証パスワード */
  password: string;
}

/** テレボート投票指示 */
export interface TelebotBetOrder {
  /** 場コード（2桁ゼロ埋め、例: "04" = 平和島） */
  stadiumCode: string;
  /** 場名 */
  stadiumName: string;
  /** レース番号 */
  raceNumber: number;
  /** 艇番（1-6） */
  boatNumber: number;
  /** 賭け式（現在は単勝固定） */
  betType: "tansho";
  /** 金額（100円単位） */
  amount: number;
}

/** 投票結果 */
export interface TelebotBetResult {
  order: TelebotBetOrder;
  success: boolean;
  /** スクリーンショットパス（証跡） */
  screenshotPath?: string;
  /** エラー時メッセージ */
  error?: string;
  /** 完了時刻 */
  completedAt: string;
  /** dry-run 実行だったか */
  dryRun: boolean;
}

/** テレボート残高情報 */
export interface TelebotBalance {
  /** 購入可能額 */
  availableBalance: number;
  /** 照会時刻 */
  queriedAt: string;
}
