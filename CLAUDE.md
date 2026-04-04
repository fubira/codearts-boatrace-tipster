# boatrace-tipster

競艇予想AI - 機械学習（LightGBM）による競艇予想ソフトウェア

コマンドの使い方は README.md を参照。

## アーキテクチャ

- **TypeScript (Bun)**: CLI、スクレイピング、DB 管理、オーケストレーション
- **Python (uv)**: ML 学習・推論（LightGBM）、特徴量エンジニアリング
- tateyamakun と同じ技術スタック・パターンを踏襲

## ディレクトリ

- `src/cli/` - CLI エントリポイント・コマンド定義
- `src/cli/commands/` - コマンド格納先（scrape, scrape-odds, predict, analyze, run, data, backup）
- `src/features/scraper/` - スクレイピング関連
- `src/features/database/` - SQLite 管理（スキーマ、マイグレーション、ストレージ、整合性チェック）
- `src/features/runner/` - 自動運用デーモン（runner, race-scheduler, slack）
- `src/shared/` - 共有モジュール（ロガー、設定）
- `ml/` - Python ML エンジン（uv 管理）
- `ml/src/boatrace_tipster_ml/` - ML コアライブラリ
- `ml/scripts/` - 学習・分析スクリプト
- `scripts/` - 運用スクリプト（バックアップ、server-tune）
- `data/` - ランタイムデータ（SQLite DB、キャッシュ、バックアップ、stats snapshot）

## データベース

SQLite（WAL モード）。スキーマバージョン管理による自動マイグレーション（tateyamakun と同パターン）。書き込みは SQLite（スクレイパー）、読み込み分析は DuckDB（ML、SQLite を READ_ONLY ATTACH）。

主要テーブル: `races`, `race_entries`, `race_odds`, `race_payouts`

- `data test` でローカル整合性チェック（孤立レコード、重複、月次完全性等）
- `data verify` でサーバとの整合性比較（スキーマ、件数、ID sum、キャッシュ数）
- `data sync` でサーバとの同期（WAL チェックポイント → アトミック DB 入れ替え＋双方向キャッシュ同期）
- `data fingerprint` で DB 統計表示
- `backup` でローカルバックアップ（7 世代ローテーション）
- `scripts/backup.sh` で外部ストレージへの tar.gz バックアップ

サーバ接続には `.env` で `PRODUCT_SERVER` と `PRODUCT_DIR`（絶対パス）を設定する。

## ML

### モデル構成

二値分類（メイン）とランキング（サブ）の2モデル構成。

| モデル | 目的 | 手法 | 用途 |
|--------|------|------|------|
| 1号艇二値分類 | 1号艇が勝つか予測 | LGBMClassifier (28特徴量) | 単勝 EV 戦略 |
| 6艇ランキング | 着順予測 | LGBMRanker LambdaRank (25特徴量) | 2連複1点買い EV 戦略 |

### EV 戦略

単勝・2連複ともに `EV = model_prob × odds - 1 > 0` のレースのみ購入する。オッズは特徴量に含めず、モデルは独立に確率を推定する。

- **単勝 EV**: 二値分類モデルの確率 × 単勝オッズで判定
- **2連複 EV**: ランキングモデルの上位2艇を組み合わせ、softmax 確率 × 2連複オッズで判定

### 特徴量パイプライン

2つのモード: フルパイプライン（学習用）と snapshot パイプライン（推論用）。

**フルパイプライン**（`build_features_df()`）:
- DB 全件読み込み → 累積統計計算 → 日付フィルタ → 相対・交互作用特徴量
- Leak-safe: `cum_all - cum_daily` パターンで同一日レースを除外
- ローリング: cumsum+shift で O(n) ベクトル化（窓: 全体5日/コース別20日）
- 学習・バックテスト用。全データが必要なため 30-60 秒かかる

**snapshot パイプライン**（`build_features_from_snapshot()`）:
- 事前計算済み統計（`data/stats-snapshots/YYYY-MM-DD.db`）+ 当日エントリの JOIN
- フルパイプラインと同一の特徴量を ~4 秒で構築
- `snapshot.py`: 累積統計・モーター残差・ローリング日次集計を SQLite に保存
- `verify_snapshot.py` で全カラムの一致を検証可能
- runner 起動時に自動構築、30 日ローテーション

共通:
- 相対特徴量: レース内 z-score（`_race_zscore`）
- 交互作用: class×boat, wind×boat, kado×exhibition 等

### 特徴量定義

- `FEATURE_COLS` (feature_config.py): ランキングモデル用24特徴量
- `BOAT1_FEATURE_COLS` (boat1_features.py): 二値分類用28特徴量
- 特徴量の順序は model 互換性のため変更禁止（末尾に追加のみ）

### モデルの保存と推論

- モデルは `ml/models/trifecta_v1/` に保存（boat1/ + ranking/）
- `predict_trifecta.py --date DATE [--snapshot PATH]`: 3連単 X-noB1-noB1 予測、JSON 出力
- `backtest_trifecta.py --from DATE --to DATE`: 期間バックテスト
- `simulate_operations.py --model-dir models/trifecta_v1`: 運用シミュレーション
- CLI からは `bun run start predict` / `bun run start analyze` で呼び出す

### 漏洩管理

gate_bias / upset_rate は学習時に漏洩あり（intraday leakage）で木構造を改善し、評価/predict 時は `neutralize_leaked_features()` でレース内 mean に置換する。学習時の漏洩を除去してはならない。

### サーバーでの ML 実行

`scripts/server-tune.sh` 経由が必須（コード同期を自動で行う）。直接 ssh で uv run を叩くとコード不整合が発生する。

```bash
./scripts/server-tune.sh --setup              # 初回セットアップ
./scripts/server-tune.sh --trials 100         # Optuna 実行（ランキング）
./scripts/server-tune.sh --model boat1 --trials 100  # Optuna 実行（二値分類）
./scripts/server-tune.sh --watch              # ログ監視
./scripts/server-tune.sh --fetch              # 結果取得
```

## 自動運用デーモン（`run` コマンド）

`bun run start run` で起動するデーモン。締切時刻ベースでレースの状態遷移を管理する。

```
[待機] → 締切30分前 → [展示取得] → 締切3分前 → [オッズ取得+EV判定] → 締切5分超過 → [スキップ]
                                                                    → 締切10分後 → [結果取得] → [完了]
```

- 起動時: 会場 discover（リトライ30分） → 全レース出走表スクレイプ → stats snapshot 構築 → スケジュール表示
- ポーリング: 60秒間隔。EV 判定はキャッシュの prob + DB 最新 odds で再計算
- 途中起動: 締切5分超過の未処理レースは自動スキップ
- 通知: Slack Webhook（`SLACK_WEBHOOK_URL` 環境変数）。未設定時はコンソール出力
- 終了: 全レース完了で自動終了、または SIGINT/SIGTERM で graceful shutdown

### 設定（.env）

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## 開発ワークフロー

- TS ファイル編集後は必ず `bun run lint:fix` → lint/typecheck 確認
- Python ファイル編集後は `uv run pytest` で確認
- テストは Co-location（ソースと同じディレクトリ）

## スクレイパー

- プラグイン式アーキテクチャ（`Scraper` インターフェース + レジストリ）
- HTML キャッシュ（gzip 圧縮、YYYYMM サブディレクトリ分割）
- 並列処理: 会場間8並列 + 同一レース3ページ並列取得
- 部分 HTML 抽出（`extractSections`）による高速パース
- `--from-cache`: キャッシュからのみパース（HTTP fetch なし）
- `--cache-only`: HTML ダウンロードのみ（パースなし）
- 進捗ログ: 50 venue-day ごとにレート・キャッシュヒット率を表示
- HTTP fetch は `[WARN] [HTTP] GET ...` で明示的にログ出力
- Zod safeParse でパーサー出力を検証
- boatrace.jp は認証不要（tateyamakun との主な差異）
