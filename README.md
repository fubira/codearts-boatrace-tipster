# boatrace-tipster

競艇予想AI - 機械学習（LightGBM LambdaRank）による P2 三連単戦略

## 運用ステータス

P2 1 号艇軸戦略は 2026-04-19 撤退、runner 停止中。scraper（データ収集）と
watchtower のみ稼働。以下の runner / 学習関連の記述は再開時の参照用として保持。
詳細は `CLAUDE.md` 冒頭と journal `2026-04-19_1343_roi-ceiling-and-strategic-retreat.md`。

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
# データ収集（サーバの scraper が自動実行中、手動実行する場合）
bun run start scrape -m 202601
bun run start scrape-odds -m 202601

# 予測（active モデルから自動解決、OOS バックテスト用）
bun run start predict -d "$(date +%F)"
```

## ドキュメント

| ガイド | 内容 |
|--------|------|
| [開発ガイド](docs/development-guide.md) | データ収集、モデル学習・実験、分析、EV 戦略、サーバーチューニング |

## スクレイピング

```bash
bun run start scrape -d 2025-01-15              # 指定日
bun run start scrape -m 202501                  # 指定月 (-y で年単位も可)
bun run start scrape -d 2025-01-15 -r 1,2,3     # レース番号指定
bun run start scrape --cache-only -y 2025       # HTML ダウンロードのみ
bun run start scrape --from-cache -y 2025       # キャッシュからパースのみ
bun run start scrape --force -d 2025-01-15      # キャッシュ無視で再取得
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

# OOS バックテスト（学習期間外のみ: 本番モデルの end_date 以降を指定）
cd ml && uv run python -m scripts.daily_p2_summary --from 2026-01-01 --to "$(date +%F)"

# 会場別 / 期間別 / feature importance の診断
cd ml && uv run python scripts/analyze_model.py --from 2026-01-01 --to "$(date +%F)" --show-importance
cd ml && uv run python scripts/analyze_model.py --from 2026-01-01 --to "$(date +%F)" --split-by month
```

## 自動運用デーモン（停止中）

runner は 2026-04-19 の撤退で停止、compose.yaml から削除済み。再開時の CLI:

```bash
bun run start run                                   # DRY RUN
bun run start run --bet-cap 30000 --bankroll 70000  # サイジング指定
bun run start run --live                            # LIVE モード
```

締切時刻ベースで自動データ取得 → T-5 で predict → T-1 で drift → Slack 通知
(`.env` の `SLACK_WEBHOOK_URL` 未設定時はコンソール出力)。

## ML 学習・チューニング（maintenance 停止中）

P2 戦略の tune / retrain は撤退に伴い停止。再開時は以下を参照:

```bash
./scripts/server-tune.sh --trials 400                # 通常探索 (overnight target)
./scripts/server-tune.sh --trials 400 --from-model models/<active> --narrow
./scripts/server-tune.sh --watch                     # ログ監視
./scripts/server-tune.sh --fetch                     # 結果取得 (Phase 1 + Phase 2)

# 上位 trial を dev モデルとして学習
cd ml && uv run python -m scripts.train_dev_model --tune-log <log> --trials <N>
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
    active.json     本番モデル指定（{"model": "<name>"}）
    .run-counter    dev model 命名カウンタ（整数1個）
    p2_v3/          本番モデル例（active.json で参照）
    am_476/         dev candidate 例（prefix は自動採番）
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
