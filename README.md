# boatrace-tipster

競艇予想AI - 機械学習（LightGBM）による競艇予想ソフトウェア

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| CLI・データ収集 | TypeScript, Bun, Cheerio |
| データベース | SQLite (better-sqlite3, WAL), DuckDB (分析クエリ) |
| ML | Python, uv, LightGBM (Binary / LambdaRank) |

## セットアップ

```bash
bun install
cd ml && uv sync
```

## クイックスタート

```bash
# 1. データ収集
bun run start scrape -m 202501                # レース + 展示タイム
bun run start scrape-odds -m 202501           # オッズ

# 2. モデル学習・保存
uv run --directory ml python -m scripts.train_boat1_binary --save

# 3. 予測
bun run start predict -d 2026-04-02

# 4. バックテスト
bun run start analyze --from 2026-03-19 --to 2026-04-03
```

各コマンドの詳細は `--help` で確認できる。

## ドキュメント

| ガイド | 内容 |
|--------|------|
| [開発ガイド](docs/development-guide.md) | データ収集、モデル学習・実験、分析、EV 戦略、サーバーチューニング |

## スクレイピング

```bash
bun run start scrape -d 2025-01-15          # 指定日の全会場
bun run start scrape -m 202501              # 指定月
bun run start scrape -y 2025               # 指定年
bun run start scrape -d 2025-01-15 -r 1,2,3 # レース番号指定
bun run start scrape --dry-run -d 2025-01-15  # DB 書き込みなし
bun run start scrape --cache-only -y 2025    # HTML ダウンロードのみ
bun run start scrape --from-cache -y 2025    # キャッシュからのみ投入
bun run start scrape --force -d 2025-01-15   # キャッシュ無視で再取得
```

## データ管理

```bash
bun run start data test           # DB 整合性チェック
bun run start data fingerprint    # DB 統計表示
bun run start data verify         # ローカル vs サーバ比較
bun run start data sync           # DB + キャッシュをサーバと同期
bun run start backup              # ローカルバックアップ（7 世代保持）
```

`data sync` / `data verify` には `.env` で `PRODUCT_SERVER` と `PRODUCT_DIR` を設定する。

## 予測・分析

```bash
# モデル保存（初回 or 再学習時）
uv run --directory ml python -m scripts.train_boat1_binary --save

# 指定日の予測
bun run start predict -d 2026-04-02          # テーブル表示
bun run start predict -d 2026-04-02 --json   # JSON 出力
bun run start predict -d 2026-04-02 --all    # 全レース表示

# 期間バックテスト
bun run start analyze --from 2026-03-19 --to 2026-04-03
bun run start analyze --from 2026-03-01 --to 2026-04-03 --bankroll 100000 --bet-cap 5000
```

## ML 学習

```bash
# 単勝1号艇 二値分類（メインモデル）
uv run --directory ml python -m scripts.train_boat1_binary --mode single
uv run --directory ml python -m scripts.train_boat1_binary --mode wfcv
uv run --directory ml python -m scripts.train_boat1_binary --mode optuna --n-trials 100

# 6艇ランキング（LambdaRank、2連複EV戦略で使用）
uv run --directory ml python -m scripts.train_eval --mode single
uv run --directory ml python -m scripts.train_eval --mode wfcv

# サーバーでの Optuna 実行
./scripts/server-tune.sh --model boat1 --trials 100  # 二値分類
./scripts/server-tune.sh --trials 100                 # ランキング
./scripts/server-tune.sh --watch                      # ログ監視
./scripts/server-tune.sh --fetch                      # 結果取得
```

## プロジェクト構造

```
src/
  cli/              CLI エントリポイント・コマンド定義
  features/
    scraper/        スクレイピング（プラグイン式、gzip キャッシュ）
    database/       SQLite 管理（スキーマ、マイグレーション、整合性チェック）
  shared/           共有モジュール（ロガー、設定）
ml/                 Python ML エンジン（uv 管理、LightGBM）
  models/           保存済みモデル（boat1/model.pkl）
  src/boatrace_tipster_ml/
    features.py     特徴量パイプライン（leak-safe cumulative、ローリング）
    feature_config.py  特徴量定義、エンコーディング、相対/交互作用特徴量
    model.py        LambdaRank モデル、時系列分割
    evaluate.py     ランキング指標、配当 ROI シミュレーション
    boat1_features.py  1号艇二値分類用データ変形（6行→1行/レース）
    boat1_model.py  LGBMClassifier、EV 分析、閾値チューニング
  scripts/          学習・分析スクリプト
scripts/            運用スクリプト（backup, server-tune 等）
data/               SQLite DB、HTML キャッシュ、バックアップ
```

## 開発コマンド

```bash
bun run lint          # Biome チェック
bun run lint:fix      # 自動修正
bun run typecheck     # 型チェック
bun run test          # テスト
bun run format        # フォーマット
```

## ライセンス

Private
