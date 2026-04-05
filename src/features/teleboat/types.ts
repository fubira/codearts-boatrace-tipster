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
  /** 賭け式 */
  betType: "tansho" | "sanrentan";
  /** 1着に選択する艇番 */
  boats1st: number[];
  /** 2着に選択する艇番（3連単時） */
  boats2nd?: number[];
  /** 3着に選択する艇番（3連単時） */
  boats3rd?: number[];
  /** 1点あたりの金額（100円単位） */
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
