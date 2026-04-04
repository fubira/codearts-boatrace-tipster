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
bun run start data sync           # サーバとの同期（DB + キャッシュ）
bun run start data sync --db-only      # DB のみ同期
bun run start data sync --cache-only   # キャッシュのみ同期
bun run start backup              # ローカルバックアップ（7 世代ローテーション）
```

`data verify` / `data sync` には `.env` で `PRODUCT_SERVER` と `PRODUCT_DIR` を設定する。

### 外部バックアップ

```bash
./scripts/backup.sh /path/to/dest                 # フル（DB + キャッシュ）
./scripts/backup.sh --db-only /path/to/dest        # DB のみ
./scripts/backup.sh --rotate 21 --db-only /path/to/dest  # 21 日ローテーション
```

## ML: モデル学習

### 1号艇二値分類（メインモデル）

1号艇が勝つかを予測する二値分類モデル。単勝 EV 戦略の核。

```bash
# 基本: single split 学習 + テスト評価
uv run --directory ml python -m scripts.train_boat1_binary --mode single

# Walk-Forward CV（4 folds × 2ヶ月）
uv run --directory ml python -m scripts.train_boat1_binary --mode wfcv --n-folds 4

# Optuna ハイパラ探索（EV>=0 ROI を最大化）
uv run --directory ml python -m scripts.train_boat1_binary --mode optuna --n-trials 100

# オプション
--n-estimators 500     # ツリー数
--learning-rate 0.05   # 学習率
--start-date 2023-01-01  # 学習開始日
--seed 42              # 乱数シード
--params '{"num_leaves":31}'  # LightGBM パラメータ上書き
```

出力: AUC、閾値別 hit 率/ROI、EV 分析（EV>=N での ROI・ベット数）、キャリブレーション、特徴量重要度。

### 6艇ランキング（LambdaRank）

6艇の着順を予測するランキングモデル。2連複1点買い EV 戦略で使用。

```bash
# single split
uv run --directory ml python -m scripts.train_eval --mode single

# Walk-Forward CV
uv run --directory ml python -m scripts.train_eval --mode wfcv --n-folds 4

# 特徴量重要度分析（split + gain + permutation）
uv run --directory ml python -m scripts.train_eval --mode importance

# Optuna（2連単 ROI 最大化）
uv run --directory ml python -m scripts.train_eval --mode optuna --n-trials 100

# オプション
--relevance linear|top_heavy|podium|win_only  # relevance scheme
```

### サーバーでの Optuna 実行

計算量の大きい Optuna 探索はサーバーで実行する。`server-tune.sh` がコード同期→nohup実行→ログ管理を一括で行う。

```bash
# 初回セットアップ（uv インストール + workspace 作成 + 依存解決）
./scripts/server-tune.sh --setup

# 実行（即座に返る、バックグラウンドで走る）
./scripts/server-tune.sh --trials 100

# 進捗確認
./scripts/server-tune.sh --status

# ログ監視（完了で自動終了）
./scripts/server-tune.sh --watch

# 結果ダウンロード
./scripts/server-tune.sh --fetch

# オプション
--server one           # SSH ホスト名（デフォルト: one）
--folds 4              # WF-CV fold 数
--fold-months 2        # fold 幅
--relevance top_heavy  # relevance scheme
--skip-sync            # コード・データ同期スキップ（2回目以降）
--foreground           # SSH 接続維持モード
```

サーバー設定は `scripts/server-tune.conf`（gitignored）で変更可能。`scripts/server-tune.conf.example` をコピーして使う。

**注意**: サーバーでの ML 実行は必ず `server-tune.sh` 経由で行う。直接 ssh で `uv run` を叩くとコード不整合が発生する。

## ML: 分析

### 1号艇飛びパターン分析

1号艇が負ける条件と、モデルの検出能力を分析する。

```bash
uv run --directory ml python -m scripts.analyze_boat1
uv run --directory ml python -m scripts.analyze_boat1 --relevance top_heavy
```

出力: 基本統計、モデル検出率、特徴量条件分析、複合条件、見逃し分析、生データ効果量（Cohen's d）、会場別/天候別飛率、イン屋影響。

### 2連単/2連複パターン検証

各パターン（2-3, 3-2 等）の二値分類モデルを個別に構築し、EV 戦略の有効性を検証する。

```bash
uv run --directory ml python -m scripts.train_exacta_binary
```

### ローリング窓サイズ最適化

ローリング特徴量の窓サイズ（race-days）を WF-CV で最適化する。

```bash
uv run --directory ml python -m scripts.grid_rolling
```

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
build_features()     → (X, y, meta) ランキング用 27特徴量
build_features_df()  → DataFrame 全列保持（二値分類用）
  ↓  reshape_to_boat1()
1号艇二値分類用    → (X_b1, y_b1, meta_b1) 24特徴量
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

## EV 戦略（3連単 X-noB1-noB1）

1号艇飛び予測時、非1号艇の1着固定 × 2-3着全流し（12点）。

```
1. b1_prob < b1_threshold → 1号艇が負けると判断
2. ランキングモデルの softmax 確率で1着 X を予測（非1号艇の最上位）
3. EV = model_prob / market_prob × 0.75 - 1
4. EV >= ev_threshold で購入
5. 12点: X-{非1,非X}-{非1,非X}
```

- オッズは特徴量に含めない（モデルは独立に確率推定）
- 3連単プール: 一般 ¥34M、G1 ¥108M → betCap ¥2,000 は全場安全

## モンテカルロシミュレーション

バックテスト統計に基づく収益・リスク予測。

```bash
# デフォルト（現在のWF-CV統計で10,000回）
uv run --directory ml python -m scripts.simulate_monte_carlo

# パラメータ変更
uv run --directory ml python -m scripts.simulate_monte_carlo --bankroll 100000 --bet-cap 3000

# 最新のバックテストから経験的パラメータを収集して実行
uv run --directory ml python -m scripts.simulate_monte_carlo --from-backtest --ev-threshold 0.33

# カスタム期間
uv run --directory ml python -m scripts.simulate_monte_carlo --days 7,14,30,60
```

## Stats Snapshot（推論高速化）

累積統計を事前計算し、予測時の DB フルスキャン（~30-60 秒）を ~4 秒に短縮する。runner 起動時に自動構築される。

```bash
# snapshot 構築（through-date = 予測日の前日）
uv run --directory ml python -m scripts.build_snapshot --through-date 2026-04-01

# snapshot を使った高速予測
uv run --directory ml python -m scripts.predict_trifecta --date 2026-04-02 \
  --snapshot data/stats-snapshots/2026-04-01.db

# snapshot の正確性検証（フルパイプラインと全カラム比較）
uv run --directory ml python -m scripts.verify_snapshot --date 2026-04-02 \
  --snapshot data/stats-snapshots/2026-04-01.db
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
