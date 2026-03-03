# boatrace-tipster

競艇予想AI - 機械学習による競艇予想ソフトウェア

## セットアップ

```bash
bun install
```

## 使い方

```bash
# CLI ヘルプ
bun run start --help

# verbose モード
bun run start -v <command>

# スクレイピング
bun run start scrape -d 2025-01-15          # 指定日の全会場
bun run start scrape -m 202501              # 指定月
bun run start scrape -y 2025               # 指定年
bun run start scrape -d 2025-01-15 -s 04    # 平和島のみ
bun run start scrape --dry-run -d 2025-01-15  # DB 書き込みなし
bun run start scrape --cache-only -y 2025    # HTML ダウンロードのみ
bun run start scrape --force -d 2025-01-15   # キャッシュ無視で再取得
```

## 開発コマンド

```bash
# リント
bun run lint
bun run lint:fix

# 型チェック
bun run typecheck

# テスト
bun run test

# フォーマット
bun run format
```

## アーキテクチャ

- **TypeScript (Bun)**: CLI、スクレイピング、DB 管理
- **Python (uv)**: ML 学習・推論（将来）

## ディレクトリ構成

```
src/
  cli/             - CLI エントリポイント
  cli/commands/    - コマンド定義（scrape 等）
  features/
    scraper/       - スクレイピング（プラグイン式、gzip キャッシュ）
    database/      - SQLite 管理（スキーマ、マイグレーション、ストレージ）
  shared/          - 共有モジュール（ロガー、設定）
data/
  cache/           - HTML キャッシュ（gzip 圧縮）
  boatrace-tipster.db  - SQLite データベース
```
