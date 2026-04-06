# 開発ガイド

データ収集、モデル学習・実験、分析のためのコマンドリファレンス。

各コマンドの詳細オプションは `--help` で確認できる。

## データ収集

```bash
bun run start scrape -d 2025-01-15          # 日付指定（全会場）
bun run start scrape -m 202501              # 月単位
bun run start scrape -y 2025               # 年単位
bun run start scrape -d 2025-01-15 -s 04    # 会場指定（平和島=04）
bun run start scrape -d 2025-01-15 -r 1,2,3 # レース番号指定
```

取得済みレースは自動スキップ。`--force` で再取得（キャッシュも更新）、`--cache-only` でキャッシュから再パース、`--from-cache` でHTTP fetchなしのキャッシュ専用投入、`--dry-run` で確認のみ。

## データ管理

```bash
bun run start data test           # DB 整合性チェック（孤立レコード、重複、月次完全性）
bun run start data fingerprint    # DB 統計表示（テーブル件数、日付範囲）
bun run start data verify         # ローカル vs サーバ比較（スキーマ、件数、ID sum）
bun run start data sync           # サーバからプル（DB + キャッシュ）
bun run start data sync --push    # ローカルからサーバへプッシュ（DB のみ）
bun run start data sync --db-only      # DB のみ同期
bun run start data sync --cache-only   # キャッシュのみ同期
bun run start data sync --dry-run      # 実行せず確認のみ
bun run start backup              # ローカルバックアップ（7 世代ローテーション）
```

`data verify` / `data sync` には `.env` で `PRODUCT_SERVER` と `PRODUCT_DIR` を設定する。

### 外部バックアップ

```bash
./scripts/backup.sh /path/to/dest                 # フル（DB + キャッシュ）
./scripts/backup.sh --db-only /path/to/dest        # DB のみ
./scripts/backup.sh --rotate 21 --db-only /path/to/dest  # 21 日ローテーション
```

## ML: 3連単戦略（メイン）

### ハイパラ探索（Optuna）

`tune_trifecta.py` が WF-CV で3連単 X-allflow 戦略の Sharpe を最大化する。結果は model_meta.json に自動保存。

```bash
# ローカル実行
uv run --directory ml python -m scripts.tune_trifecta --trials 100

# サーバ実行（推奨）
./scripts/server-tune.sh --model trifecta --trials 100
```

### 本番モデル学習

model_meta.json のハイパラで学習し、モデルを保存する。

```bash
# ランキングモデル
uv run --directory ml python -m scripts.train_ranking --save --model-meta models/trifecta_v1/ranking

# 1号艇二値分類
uv run --directory ml python -m scripts.train_boat1_binary --mode single
```

### バックテスト

保存済み本番モデルで OOS 期間を評価する。WF-CV モードはハイパラ探索の検証専用。

```bash
# 期間指定（本番モデルをロード、再学習しない）
uv run --directory ml python -m scripts.backtest_trifecta --from 2026-01-01 --to 2026-04-08

# WF-CV（ハイパラ探索の検証専用）
uv run --directory ml python -m scripts.backtest_trifecta --wfcv

# EV 閾値スイープ
uv run --directory ml python -m scripts.backtest_trifecta --ev-sweep

# オプション
--model-dir models/trifecta_v1   # モデルディレクトリ（デフォルト）
--b1-threshold 0.42              # model_meta から自動読み込み
--ev-threshold 0.36              # model_meta から自動読み込み
```

### モンテカルロシミュレーション

保存済みモデルの OOS 統計に基づく収益・リスク予測。

```bash
# 保存済みモデルの OOS データから経験的パラメータを収集して実行
uv run --directory ml python -m scripts.simulate_monte_carlo --from-backtest

# 閾値比較
uv run --directory ml python -m scripts.simulate_monte_carlo --from-backtest \
  --compare "0.42:0.20,0.42:0.30,0.42:0.36" --all-flow

# パラメータ変更
uv run --directory ml python -m scripts.simulate_monte_carlo --bankroll 100000 --bet-cap 3000
```

### サーバーでの Optuna 実行

計算量の大きい Optuna 探索はサーバーで実行する。`server-tune.sh` がコード・DB同期 → nohup実行 → ログ管理を一括で行う。

```bash
./scripts/server-tune.sh --setup                        # 初回セットアップ
./scripts/server-tune.sh --model trifecta --trials 100  # 3連単戦略の探索
./scripts/server-tune.sh --model boat1 --trials 100     # 二値分類の探索
./scripts/server-tune.sh --status                       # 進捗確認
./scripts/server-tune.sh --watch                        # ログ監視（完了で自動終了）
./scripts/server-tune.sh --fetch                        # 結果ダウンロード

# オプション
--skip-sync            # コード・データ同期スキップ（DB を手動で送った場合等）
--warm-start           # 現行 model_meta のパラメータで初期化
--seed 43              # 乱数シード（デフォルト: 42、変更で探索範囲拡大）
--foreground           # SSH 接続維持モード
```

**注意**: サーバーでの ML 実行は必ず `server-tune.sh` 経由で行う。直接 ssh で `uv run` を叩くとコード不整合が発生する。

## ML: 特徴量パイプライン

### データフロー

```
SQLite DB
  ↓  DuckDB READ_ONLY ATTACH
全データ読み込み (races + race_entries + race_odds)
  ↓
歴史的特徴量（leak-safe: cum_all - cum_daily）
  ↓
ローリング特徴量（cumsum+shift O(n)）
  ↓
開催内特徴量（tournament_id ベース）
  ↓
漏洩特徴量（gate_bias, upset_rate）
  ↓
カテゴリカルエンコーディング
  ↓  日付フィルタ
相対特徴量（レース内 z-score）
  ↓
交互作用特徴量（class×boat, wind×boat 等）
  ↓
build_features_df()  → DataFrame 全列保持（二値分類用）
  ↓  prepare_feature_matrix()
ランキング用         → (X, y, meta) 24特徴量
  ↓  reshape_to_boat1()
1号艇二値分類用      → (X_b1, y_b1, meta_b1) 28特徴量
```

### 特徴量の追加方法

1. 歴史的特徴量: `features.py` に `_add_*()` 関数を追加し、`build_features_df()` 内で呼び出す
2. 相対/交互作用特徴量: `feature_config.py` の `compute_relative_features()` / `compute_interaction_features()` に追加
3. ランキングモデル用: `FEATURE_COLS` の末尾に追加（順序変更禁止）
4. 二値分類用: `boat1_features.py` の `BOAT1_FEATURE_COLS` と `reshape_to_boat1()` に追加

### Leak-safe パターン

同一日レースの結果漏洩を防ぐため、全ての歴史的特徴量で以下のパターンを使う:

```python
prior = cum_all - cum_daily      # 同一日を除外
prior_count = count_all - count_daily
result = prior / prior_count      # NaN if no prior data
```

### 漏洩特徴量（gate_bias / upset_rate）

学習時は漏洩あり（intraday cumsum）で木構造を改善し、評価時は `neutralize_leaked_features()` でレース内 mean に置換する。学習時の漏洩を除去してはならない。

## EV 戦略（3連単 X-allflow）

1号艇飛び予測時、非1号艇の1着固定 × 2-3着全流し（20点）。

```
1. b1_prob < b1_threshold → 1号艇が負けると判断
2. ランキングモデルの softmax 確率で1着 X を予測（非1号艇の最上位）
3. EV = model_prob / market_prob × 0.75 - 1
4. EV >= ev_threshold で購入
5. 20点: X-{全}-{全}（1着 X 固定、2-3着全流し）
```

- 閾値は model_meta.json から読み込み（b1_threshold, ev_threshold）
- 評価ロジックは `evaluate_trifecta_strategy()` に一本化（tune, backtest, MC が共用）
- オッズは特徴量に含めない（モデルは独立に確率推定）

## Stats Snapshot（推論高速化）

累積統計を事前計算し、予測時の DB フルスキャン（~30-60 秒）を ~4 秒に短縮する。runner 起動時に自動構築される。

```bash
# snapshot 構築（through-date = 予測日の前日）
uv run --directory ml python -m scripts.build_snapshot --through-date 2026-04-01

# snapshot を使った高速予測
uv run --directory ml python -m scripts.predict_trifecta --date 2026-04-02 \
  --snapshot data/stats-snapshots/2026-04-01.db

# snapshot の正確性検証（フルパイプラインと全カラム比較）
uv run --directory ml python -m scripts.verify_snapshot --date 2026-04-02
```

snapshot は `data/stats-snapshots/YYYY-MM-DD.db`（~51 MB）に日付別保存。30 日ローテーション。

## 開発コマンド

```bash
bun run lint          # Biome チェック
bun run lint:fix      # 自動修正
bun run typecheck     # 型チェック
bun run test          # TS テスト
bun run format        # フォーマット

# Python テスト
uv run --directory ml pytest
```
