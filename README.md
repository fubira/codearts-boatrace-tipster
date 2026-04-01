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

### スクレイピング

```bash
bun run start scrape -d 2025-01-15          # 指定日の全会場
bun run start scrape -m 202501              # 指定月
bun run start scrape -y 2025               # 指定年
bun run start scrape -d 2025-01-15 -s 04    # 平和島のみ
bun run start scrape -d 2025-01-15 -r 1,2,3 # レース番号指定
bun run start scrape --dry-run -d 2025-01-15  # DB 書き込みなし
bun run start scrape --cache-only -y 2025    # HTML ダウンロードのみ
bun run start scrape --from-cache -y 2025    # キャッシュからのみ投入（HTTP なし）
bun run start scrape --force -d 2025-01-15   # キャッシュ無視で再取得
```

### データ管理

```bash
bun run start verify              # DB 整合性チェック
bun run start backup              # ローカルバックアップ（7 世代保持）
bun run start backup -n 14        # 保持数を変更
```

### 外部バックアップ

```bash
./scripts/backup.sh /path/to/dest                 # フル（DB + キャッシュ）
./scripts/backup.sh --db-only /path/to/dest        # DB のみ
./scripts/backup.sh --rotate 21 --db-only /path/to/dest  # 21 日ローテーション
```

## 開発コマンド

```bash
bun run lint          # リント
bun run lint:fix      # リント自動修正
bun run typecheck     # 型チェック
bun run test          # テスト
bun run format        # フォーマット
```

## アーキテクチャ

- **TypeScript (Bun)**: CLI、スクレイピング、DB 管理
- **Python (uv)**: ML 学習・推論（将来）

## ディレクトリ構成

```
src/
  cli/             - CLI エントリポイント
  cli/commands/    - コマンド定義（scrape, verify, backup）
  features/
    scraper/       - スクレイピング（プラグイン式、gzip キャッシュ、YYYYMM 分割）
    database/      - SQLite 管理（スキーマ、マイグレーション、整合性チェック）
  shared/          - 共有モジュール（ロガー、設定）
scripts/           - 運用スクリプト（バックアップ、キャッシュ移行）
data/
  cache/           - HTML キャッシュ（gzip 圧縮、YYYYMM サブディレクトリ）
  backups/         - DB バックアップ
  boatrace-tipster.db  - SQLite データベース
```
