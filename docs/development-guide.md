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

取得済みレースは自動スキップ。`--force` で再取得（キャッシュも更新）、
`--cache-only` でキャッシュから再パース、`--from-cache` で HTTP fetch なしの
キャッシュ専用投入、`--dry-run` で確認のみ。

## データ管理

```bash
bun run start data test           # DB 整合性チェック
bun run start data fingerprint    # DB 統計表示
bun run start data verify         # ローカル vs サーバ比較
bun run start data sync           # サーバからプル（DB + キャッシュ + snapshot）
bun run start data sync --push    # ローカルからサーバへプッシュ
bun run start data sync --db-only      # DB のみ同期
bun run start data sync --cache-only   # キャッシュのみ同期
bun run start data sync --dry-run      # 実行せず確認のみ
bun run start backup              # ローカルバックアップ（7 世代ローテーション）
```

`data verify` / `data sync` には `.env` で `PRODUCT_SERVER` と `PRODUCT_DIR` を設定する。

### 外部バックアップ

```bash
./scripts/backup.sh /path/to/dest                       # フル
./scripts/backup.sh --db-only /path/to/dest             # DB のみ
./scripts/backup.sh --rotate 21 --db-only /path/to/dest # 21 世代ローテーション
```

## ML: P2 戦略

P2 戦略は LightGBM LambdaRank モデル単独。1号艇が1着になるレースで2-3着の
P2 パターン（最大2点）を購入する3連単戦略。

### モデル管理

```
ml/models/
├── active.json     ← 本番モデル指定（{"model": "<name>"}）
├── .run-counter    ← dev model 命名カウンタ（整数1個）
├── p2_v3/          ← 本番モデル例（active.json で参照）
├── p2_v2/          ← 旧本番モデル（rollback 用）
├── am_476/         ← dev candidate 例（prefix は自動採番）
├── ab_*/           ← 別 tune run の dev candidate
└── tune_result/    ← Optuna 探索結果
```

本番モデルの切替は `active.json` を 1 行書き換えるだけ。コード変更不要。

### Optuna ハイパラ探索

サーバ側 (server-tune.sh) で並列実行する。各 tune は local の `.run-counter`
から prefix を 1 つ消費し、由来する dev model 全てがその prefix を共有する。

```bash
./scripts/server-tune.sh --setup                       # 初回セットアップ
./scripts/server-tune.sh --trials 100                  # 通常探索（広域）
./scripts/server-tune.sh --trials 100 --from-model models/<active> --narrow
                                                       # seed 周辺を集中探索
./scripts/server-tune.sh --status                      # 進捗確認
./scripts/server-tune.sh --watch                       # ログ監視（完了で自動終了）
./scripts/server-tune.sh --fetch                       # 結果ダウンロード

# よく使うオプション
--fix-thresholds "gap23=0.13,ev=0.0,top3_conc=0.6"    # 閾値を固定して HP のみ探索
--from-model models/<active>                           # 既存モデル HP を warm-start
--narrow                                               # --from-model 周辺だけを探索
--n-jobs 2                                             # 並列 trial 数（default: 2）
--num-threads 4                                        # trial あたり LightGBM threads
--seed N                                               # 明示指定で再現性確保
--relevance podium                                     # relevance scheme 固定
--objective growth|kelly                               # 最適化目的（default: growth）
```

**注意**:
- サーバでの ML 実行は必ず `server-tune.sh` 経由で行う。直接 ssh で `uv run`
  を叩くとコード不整合が発生する
- `--seed` は無指定なら自動ランダム。明示指定は再現性確保したい時のみ
- `--n-jobs > 1` はサーバ専用（`BOATRACE_TUNE_PARALLEL=1` env var 必須、
  server-tune.sh が自動セット）

### dev モデルの学習・昇格

tune の trials.json から特定 trial を選んで本格学習する。tune の `run_prefix`
を自動継承するので `--prefix` 指定は通常不要。

```bash
# tune の上位 trial を dev モデルとして保存
cd ml && uv run python -m scripts.train_dev_model \
  --tune-log logs/tune/<timestamp>_server-tune.trials.json \
  --trials <N1>,<N2>,<N3>
# → models/<prefix>_<N>/ が作成される（prefix は trials.json の run_prefix）

# dev モデルの一覧
uv run python -m scripts.train_dev_model --list
```

本番昇格は `ml/models/active.json` の `model` フィールドを書き換えるだけ:

```json
{"model": "<model_name>"}
```

サーバ側に同期するときは scp + active.json 編集:

```bash
scp -r ml/models/<model_name> one:/home/one/boatrace-tipster/ml/models/
ssh one "echo '{\"model\": \"<model_name>\"}' > /home/one/boatrace-tipster/ml/models/active.json"
```

### バックテスト・OOS 評価

```bash
# 期間 OOS 評価（active モデル使用、daily 単位の summary）
cd ml && uv run python -m scripts.daily_p2_summary --from 2026-01-01 --to "$(date +%F)"

# 別モデルとの比較
cd ml && uv run python -m scripts.daily_p2_summary --from 2026-01-01 --to "$(date +%F)" \
  --model-dir models/<other_model>
```

`daily_p2_summary` は2つの path で評価する:
- **確定オッズ path**: 全レースを `race_odds`（最終確定）で評価。サンプル豊富
- **T-5 path**: `race_odds_snapshots` の T-5 オッズで評価（runner の予測時点を再現）

### Predict (T-5 再現含む)

```bash
# 本日の予測（active モデル）
bun run start predict -d "$(date +%F)"

# T-5 snapshot で再現（runner の T-5 判断と一致）
bun run start predict -d "$(date +%F)" --use-snapshots

# 別モデルで予測
bun run start predict -d "$(date +%F)" --model-dir ml/models/<other_model>

# JSON 出力
bun run start predict -d "$(date +%F)" --json
```

### 運用診断

日次・週次の状況確認スクリプト。DB の最新レースから自動で期間を解決するので引数なしでも動く。

```bash
# 1 日の P2 判定を bucket 別に分類 (BOUGHT / not_b1 / gap12_low / conc_low / gap23_low / ev_low)
# borderline skip と「閾値を緩めた場合の追加 hit」を表示
cd ml && uv run python -m scripts.analyze_decisions              # 最新日
cd ml && uv run python -m scripts.analyze_decisions --date 2026-04-14

# recent (default 7 日) vs baseline (default 30 日) で miss 内訳を比較
# 最近のハズレが過去と同じ分布かを確認、週次 health check
cd ml && uv run python -m scripts.compare_miss_patterns

# T-5 / T-1 drift / 確定オッズの 3 path を期間比較
# T-1 drop rate (T-5 EV+ → T-1 EV- に落ちた券の率) と per-day 内訳を表示
# 対象期間は race_odds_snapshots 存在日に限定 (2026-04-07 以降)
cd ml && uv run python -m scripts.analyze_t5_t1_drift \
    --from 2026-04-07 --to "$(date +%F)"
```

**使い分けの目安**:

- `analyze_decisions`: 日次、今日の判定が妥当かの振り返り
- `compare_miss_patterns`: 週次、近日の miss 分布が過去と一致するかの health check
- `analyze_t5_t1_drift`: 月次、モデルが選ぶ券が市場で削られやすくなっていないかの構造監視。snapshot データ 30 日以上蓄積後に信頼度が上がる

## ML: 特徴量パイプライン

P2 戦略では2つの特徴量構築 path がある:

- **フルパイプライン** (`build_features_df`): DB 全件読み込み → 累積統計
  → 日付フィルタ → 相対・交互作用特徴量。学習・バックテスト用。初回 ~20秒、
  2回目以降は pickle キャッシュで ~1秒
- **snapshot パイプライン** (`build_features_from_snapshot`): 事前計算済み
  統計 (`data/stats-snapshots/YYYY-MM-DD.db`) と当日エントリの JOIN。
  推論用（runner が使う）。~4秒

### Stats snapshot

```bash
# snapshot 構築（through-date = 予測日の前日）
cd ml && uv run python -m scripts.build_snapshot --through-date "$(date -d yesterday +%F)"

# snapshot とフルパイプラインの一致検証
cd ml && uv run python -m scripts.verify_snapshot --date "$(date +%F)"

# 3 path 一貫性テスト（backtest / predict full / predict snapshot）
cd ml && PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/test_predict_backtest_consistency.py
```

snapshot は `data/stats-snapshots/YYYY-MM-DD.db`（~51 MB）に日付別保存、
runner 起動時に自動構築・30 日ローテーション。

### 漏洩管理

- **予測時に不明な情報は特徴量に使わない**: `course_number`（実際の進入コース）
  は `boat_number` で代替
- `gate_bias` / `upset_rate` は学習時に漏洩あり（intraday）で木構造を改善し、
  評価/predict 時は `neutralize_leaked_features()` でレース内 mean に置換

## 開発コマンド

```bash
bun run lint          # Biome チェック
bun run lint:fix      # 自動修正
bun run typecheck     # 型チェック
bun run test          # TS テスト
bun run format        # フォーマット

# Python テスト
cd ml && uv run pytest
```
