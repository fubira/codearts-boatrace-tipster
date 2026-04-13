# boatrace-tipster

競艇予想AI - 機械学習（LightGBM LambdaRank）による P2 三連単戦略

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| CLI・データ収集 | TypeScript, Bun, Cheerio |
| データベース | SQLite (WAL), DuckDB (分析クエリ) |
| ML | Python, uv, LightGBM LambdaRank, Optuna |
| デプロイ | Docker, GHCR, watchtower |

## セットアップ

```bash
bun install
cd ml && uv sync
```

## クイックスタート

```bash
# 1. データ収集
bun run start scrape -m 202601           # 月単位
bun run start scrape-odds -m 202601

# 2. 予測（active モデルから自動解決）
bun run start predict -d "$(date +%F)"

# 3. 自動運用デーモン
bun run start run                         # DRY RUN
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
# 指定日の予測（active モデルを ml/models/active.json から解決）
bun run start predict -d "$(date +%F)"
bun run start predict -d "$(date +%F)" --json          # JSON 出力
bun run start predict -d "$(date +%F)" --use-snapshots # T-5 snapshot odds で再現

# OOS バックテスト（学習期間外のみ: p2_v2 は --from 2026-01-01 以降）
cd ml && uv run python -m scripts.daily_p2_summary --from 2026-01-01 --to "$(date +%F)"

# 会場別 / 期間別 / feature importance の診断
cd ml && uv run python scripts/analyze_model.py --from 2026-01-01 --to "$(date +%F)" --show-importance
cd ml && uv run python scripts/analyze_model.py --from 2026-01-01 --to "$(date +%F)" --split-by month
```

## 自動運用デーモン

```bash
bun run start run                                   # DRY RUN（デフォルト）
bun run start run --bet-cap 30000 --bankroll 70000  # サイジング指定
bun run start run --live                            # LIVE モード（将来の自動購入用）
```

締切時刻ベースで自動データ取得 → T-5 で predict → T-1 で drift 判定 → Slack 通知。
`.env` に `SLACK_WEBHOOK_URL` を設定すると Slack に通知が飛ぶ（未設定時はコンソール出力）。

## ML 学習・チューニング

サーバ側の Optuna 探索 (server-tune.sh) で並列実行・自動 prefix 採番される。

```bash
./scripts/server-tune.sh --trials 100                # 通常探索
./scripts/server-tune.sh --trials 100 --from-model models/p2_v2 --narrow
./scripts/server-tune.sh --watch                     # ログ監視
./scripts/server-tune.sh --fetch                     # 結果取得

# 上位 trial を dev モデルとして学習
cd ml && uv run python -m scripts.train_dev_model --tune-log <log> --trials 294
# 本番昇格は ml/models/active.json を書き換えるだけ
```

## プロジェクト構造

```
src/
  cli/              CLI エントリポイント・コマンド定義
  features/
    scraper/        スクレイピング（プラグイン式、gzip キャッシュ）
    database/       SQLite 管理（スキーマ、マイグレーション、整合性チェック）
    runner/         自動運用デーモン（スケジューラ、Slack 通知）
  shared/           共有モジュール（ロガー、設定、active model 解決）
ml/
  models/
    active.json     本番モデル指定（{"model": "p2_v2"}）
    .run-counter    dev model 命名カウンタ（整数1個）
    p2_v2/          本番モデル（active.json で参照）
    aa_294/         dev candidate（prefix は自動採番）
  src/boatrace_tipster_ml/
    features.py     特徴量パイプライン（leak-safe cumulative、ローリング）
    snapshot_features.py  事前計算済み統計からの高速特徴量構築
    model.py        LambdaRank モデル、時系列分割
    registry.py     active.json + .run-counter ヘルパ
  scripts/          学習・チューニング・推論スクリプト
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
