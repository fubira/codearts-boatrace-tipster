# boatrace-tipster

競艇予想AI - 機械学習による競艇予想ソフトウェア

コマンドの使い方・開発コマンドは README.md を参照。

## アーキテクチャ

- **TypeScript (Bun)**: CLI、スクレイピング、DB 管理、オーケストレーション
- **Python (uv)**: ML 学習・推論（将来追加予定）
- tateyamakun と同じ技術スタック・パターンを踏襲

## ディレクトリ

- `src/cli/` - CLI エントリポイント・コマンド定義
- `src/cli/commands/` - コマンド格納先
- `src/features/scraper/` - スクレイピング関連
- `src/features/database/` - SQLite 管理（スキーマ、マイグレーション、ストレージ）
- `src/shared/` - 共有モジュール（ロガー、設定）
- `data/` - ランタイムデータ（SQLite DB、キャッシュ）

## データベース

SQLite（WALモード予定）。スキーマバージョン管理による自動マイグレーション（tateyamakun と同パターン）。

## 開発ワークフロー

- TS ファイル編集後は必ず `bun run lint:fix` → lint/typecheck 確認
- テストは Co-location（ソースと同じディレクトリ）

## 重点事項

- スクレイパーはプラグイン形式で構築予定
- HTMLキャッシュ（gzip 圧縮）でパーサー修正後の再取得を高速化
