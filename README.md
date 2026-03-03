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
  cli/           - CLI エントリポイント・コマンド定義
  features/
    scraper/     - スクレイピング
    database/    - SQLite 管理
  shared/        - 共有モジュール（ロガー、設定）
data/            - ランタイムデータ（DB、キャッシュ）
```
