# boatrace-tipster

競艇予想AI - 機械学習による競艇予想ソフトウェア

コマンドの使い方・開発コマンドは README.md を参照。

## アーキテクチャ

- **TypeScript (Bun)**: CLI、スクレイピング、DB 管理、オーケストレーション
- **Python (uv)**: ML 学習・推論（将来追加予定）
- tateyamakun と同じ技術スタック・パターンを踏襲

## ディレクトリ

- `src/cli/` - CLI エントリポイント・コマンド定義
- `src/cli/commands/` - コマンド格納先（scrape, verify, backup）
- `src/features/scraper/` - スクレイピング関連
- `src/features/database/` - SQLite 管理（スキーマ、マイグレーション、ストレージ、整合性チェック）
- `src/shared/` - 共有モジュール（ロガー、設定）
- `scripts/` - 運用スクリプト（バックアップ、キャッシュ移行）
- `data/` - ランタイムデータ（SQLite DB、キャッシュ、バックアップ）

## データベース

SQLite（WALモード）。スキーマバージョン管理による自動マイグレーション（tateyamakun と同パターン）。

- `bun run start verify` で整合性チェック（孤立レコード、重複、月次完全性等）
- `bun run start backup` でローカルバックアップ（7世代ローテーション）
- `scripts/backup.sh` で外部ストレージへの tar.gz バックアップ

## 開発ワークフロー

- TS ファイル編集後は必ず `bun run lint:fix` → lint/typecheck 確認
- テストは Co-location（ソースと同じディレクトリ）

## スクレイパー

- プラグイン式アーキテクチャ（`Scraper` インターフェース + レジストリ）
- HTMLキャッシュ（gzip 圧縮、YYYYMM サブディレクトリ分割）
- 並列処理: 会場間8並列 + 同一レース3ページ並列取得
- 部分HTML抽出（`extractSections`）による高速パース
- `--from-cache`: キャッシュからのみパース（HTTP fetch なし）
- `--cache-only`: HTML ダウンロードのみ（パースなし）
- 進捗ログ: 50 venue-day ごとにレート・キャッシュヒット率を表示
- HTTP fetch は `[WARN] [HTTP] GET ...` で明示的にログ出力
- Zod safeParse でパーサー出力を検証
- boatrace.jp は認証不要（tateyamakun との主な差異）
