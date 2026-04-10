# boatrace-tipster

競艇予想AI - 機械学習（LightGBM）による競艇予想ソフトウェア

## アーキテクチャ

- **TypeScript (Bun)**: CLI、スクレイピング、DB 管理、オーケストレーション
- **Python (uv)**: ML 学習・推論（LightGBM）、特徴量エンジニアリング
- tateyamakun と同じ技術スタック・パターンを踏襲

## ディレクトリ

- `src/cli/commands/` - コマンド格納先（scrape, scrape-daemon, scrape-odds, predict, analyze, run, data, backup）
- `src/features/scraper/` - スクレイピング関連（scrape-daemon 含む）
- `src/features/database/` - SQLite 管理（スキーマ、マイグレーション、ストレージ、整合性チェック）
- `src/features/runner/` - 自動運用デーモン（runner, race-scheduler, scrape-helpers, slack）
- `src/shared/` - 共有モジュール（ロガー、設定）
- `ml/src/boatrace_tipster_ml/` - ML コアライブラリ
- `ml/scripts/` - 学習・分析スクリプト
- `scripts/` - 運用スクリプト（バックアップ、server-tune）
- `data/` - ランタイムデータ（SQLite DB、キャッシュ、バックアップ、stats snapshot）

## CLI リファレンス

### TS CLI（`bun run start <command>`）

```
scrape -d DATE [-m YYYYMM] [-y YYYY] [-s STADIUM] [-r 1,2,3] [-l N]
       [--dry-run] [--force] [--no-cache] [--cache-only] [--from-cache]
       ※ --date/--month/--year のいずれか必須
       ※ --cache-only と --from-cache は排他

scrape-odds -d DATE [-m YYYYMM] [-y YYYY] [-s STADIUM]
            [--dry-run] [--no-cache] [--cache-only] [--from-cache]

predict -d DATE [--json] [--model-dir DIR] [--ev-threshold N]

analyze --from DATE --to DATE [--b1-threshold N] [--ev-threshold N] [--json]

scrape-daemon                    # 独立スクレイピングデーモン（runner 不要でデータ収集）

run [--dry-run(default)] [--live] [--ev-threshold N] [--bet-cap 2000] [--bankroll 70000]

data test                        # ローカル整合性チェック
data verify                      # サーバとの整合性比較
data sync [--db-only] [--cache-only] [--dry-run] [--push]  # サーバ同期（フラグなしで snapshot も同期）
data fingerprint                 # DB統計表示

backup [-n 7]                    # ローカルバックアップ（N世代ローテーション）
```

### Python ML（`cd ml && uv run python scripts/<script>`）

```
predict_trifecta.py --date DATE [--snapshot PATH] [--race-ids ID,...] [--use-snapshots]
                    [--model-dir models/trifecta_v1] [--b1-threshold N]
                    [--r2-threshold N]

backtest_trifecta.py --from DATE --to DATE [--b1-threshold N] [--ev-threshold N] [--json]
                     [--r2-threshold N] [--model-dir models/trifecta_v1] [--start-date DATE]
                     [--db-path PATH] [--weekly]
backtest_trifecta.py --wfcv [--n-folds 4] [--fold-months 2] [--ev-sweep]
backtest_trifecta.py --threshold-sweep --from DATE --to DATE [--model-dir DIR] [--json]

tune_trifecta.py --trials N [--seed 42] [--n-folds 4] [--fold-months 2] [--warm-start]
                 [--relevance SCHEME] [--objective growth|kelly|sharpe|profit] [--with-r2]
                 [--fix-thresholds "b1=0.35,ev=0.29"] [--validate-top 10]

train_ranking.py --save [--model-dir models/trifecta_v1/ranking] [--model-meta DIR]
                 [--n-estimators N] [--learning-rate N] [--relevance SCHEME] ...LGBMオーバーライド

train_boat1_binary.py --mode single|wfcv|optuna [--save] [--model-dir models/boat1]
                      [--n-estimators 500] [--learning-rate 0.05] [--seed 42]

build_snapshot.py --through-date DATE [--db-path PATH] [--output PATH]
verify_snapshot.py --date DATE [--db-path PATH] [--snapshot PATH]
daily_trifecta.py --date DATE | --from DATE --to DATE [--model-dir DIR] [--json]

simulate_monte_carlo.py [--from-backtest] [--bankroll 70000] [--unit-divisor 800]
                        [--bet-cap 2000] [--n-sims 10000] [--all-flow]
                        [--r2-threshold N] [--compare "b1:ev:r2ev,..."]

promote_model.py [--draft models/draft] [--prod models/trifecta_v1]
                 [--component ranking|boat1] [--yes]
```

### server-tune（`./scripts/server-tune.sh`）

```
--setup                          # 初回セットアップ
--model trifecta|ranking|boat1 --trials N  # Optuna 実行（default: trifecta）
--watch                          # ログ監視
--fetch                          # 結果取得
--with-r2                        # rank-2フォールバック有効化（default: 無効）
--fix-thresholds "b1=0.35,ev=0.29"  # 閾値固定でハイパラのみ探索（Phase 1）
--validate-top 10                   # 探索後にTop NをOOS検証（Phase 1.5）
```

### npm scripts

```
bun run lint       # biome check
bun run lint:fix   # biome check --write
bun run format     # biome format --write
bun run test       # bun test
bun run typecheck  # tsc --noEmit
```

## データベース

SQLite（WAL モード）。スキーマバージョン管理による自動マイグレーション。書き込みは SQLite（スクレイパー）、読み込み分析は DuckDB（ML、SQLite を READ_ONLY ATTACH）。

主要テーブル: `races`, `race_entries`, `race_odds`, `race_payouts`, `race_odds_snapshots`

サーバ接続には `.env` で `PRODUCT_SERVER` と `PRODUCT_DIR`（絶対パス）を設定する。

## ML

### モデルディレクトリ

```
models/
  draft/              ← 学習スクリプトのデフォルト保存先
    ranking/            model.pkl + model_meta.json
    boat1/              model.pkl + model_meta.json
  tune_result/        ← Optuna 探索結果の保存先
  trifecta_v1/        ← 本番モデル（runner/predict が読む）
    ranking/
    boat1/
```

- 学習: `train_ranking.py` / `train_boat1_binary.py` → `models/draft/` に保存
- 探索: `tune_trifecta.py` → `models/tune_result/` に保存
- 昇格: `promote_model.py` で draft → trifecta_v1 にコピー（確認プロンプト付き）
- 推論/評価: `predict_trifecta.py` / `backtest_trifecta.py` → `models/trifecta_v1/` を読む

### モデル構成

二値分類とランキングの2モデル構成。3連単 X-allflow 戦略で使用。

| モデル | 目的 | 手法 | 用途 |
|--------|------|------|------|
| 1号艇二値分類 | 1号艇が勝つか予測 | LGBMClassifier (32特徴量) | 1号艇飛び判定 |
| 6艇ランキング | 着順予測 | LGBMRanker LambdaRank (30特徴量) | 1着予測（非1号艇） |

### EV 戦略（3連単 X-allflow）

1号艇飛び予測時、非1号艇の1着固定 × 2-3着全流し（20点）。
`EV = model_prob / market_prob × 0.75 - 1 > threshold` で購入判断。オッズは特徴量に含めない。

- b1_prob < threshold → 1号艇が負けると判断
- ランキングモデルの softmax 確率で1着を予測
- **3連単オッズから逆算した市場確率**と比較して EV 判定（単勝オッズは使わない。プールが薄すぎて非本命を過大評価する）
- **Rank-2 フォールバック（デフォルト無効）**: rank-1 の EV が閾値未満のとき、rank-2（2番目の非1号艇）の EV が `r2_ev_threshold` 以上なら購入。短期は互角だが長期 MC で MaxDD 悪化・破産率上昇のため無効化。`--with-r2` で有効化
- 評価ロジックは `evaluate_trifecta_strategy()` に一本化（tune, backtest, MC が共用）

### 特徴量パイプライン

2つのモード: フルパイプライン（学習用）と snapshot パイプライン（推論用）。

**フルパイプライン**（`build_features_df()`）:
- DB 全件読み込み → 累積統計計算 → 日付フィルタ → 相対・交互作用特徴量
- Leak-safe: `cum_all - cum_daily` パターンで同一日レースを除外
- ローリング: cumsum+shift で O(n) ベクトル化（窓: 全体5日/コース別20日）
- 学習・バックテスト用。初回は全データ計算で ~20 秒、2回目以降は pickle キャッシュで ~1 秒（DB または feature コード変更で自動無効化）

**snapshot パイプライン**（`build_features_from_snapshot()`）:
- 事前計算済み統計（`data/stats-snapshots/YYYY-MM-DD.db`）+ 当日エントリの JOIN
- フルパイプラインと同一の特徴量を ~4 秒で構築
- runner 起動時に自動構築、30 日ローテーション

共通:
- 相対特徴量: レース内 z-score（`_race_zscore`）
- 交互作用: class×boat, wind×boat, kado×exhibition 等
- LambdaRank はレース内の6艇間の順位差を学習するため、レース内で変動しない値（stadium_id, wind等）は単体では無意味。交互作用やカテゴリカル指定で活かす

### 特徴量定義

- `FEATURE_COLS` (feature_config.py): ランキングモデル用特徴量
- `BOAT1_FEATURE_COLS` (boat1_features.py): 二値分類用特徴量
- 特徴量の順序は model 互換性のため変更禁止（末尾に追加のみ）

### モデルの保存と推論

- モデルは `ml/models/trifecta_v1/` に保存（boat1/ + ranking/）
- `model_meta.json` がハイパラ・閾値の唯一の真実の源（全スクリプトがここから読む）
- 学習と運用は分離。同じモデルでも DB 内容の差で累積特徴量が変わる。学習はローカル→モデルをデプロイ。運用側は保存モデルで推論のみ

### 漏洩管理

gate_bias / upset_rate は学習時に漏洩あり（intraday leakage）で木構造を改善し、評価/predict 時は `neutralize_leaked_features()` でレース内 mean に置換する。学習時の漏洩を除去してはならない。

## ML 運用ルール

- **評価は OOS のみ**: 保存済みモデルの評価は学習期間外（OOS）でのみ行う。インサンプル期間は過学習で膨らむため評価に使わない
- **本番モデル学習は `--end-date 2026-01-01` 必須**: `train_ranking.py` / `train_boat1_binary.py` はデフォルトで全データを使う。OOS 期間（2026-01〜）にデータが漏れるので、本番モデル保存時は必ず `--end-date` を指定する
- **WF-CV は探索専用**: WF-CV は毎回モデルを再学習するため、保存済み本番モデルの評価には使わない。ハイパラ探索=WF-CV、本番モデル評価=保存済みモデルで `backtest_trifecta.py --from --to`
- **Optuna seed**: TPESampler seed=42 固定だと同一 trials 数で同じ結果になる。探索拡大には seed を変えるか trials を増やす
- **WF-CV 直列実行**: WF-CV を並列実行するとメモリ・CPU が溢れてマシンがフリーズする。1つずつ
- **MC の --from-backtest**: `simulate_monte_carlo.py` のデフォルト値は楽観的。実モデルの数値を使うため必ず `--from-backtest` を使う
- **特徴量変更は Optuna 再探索とセット**: 特徴量を変えるとスコア分布が変わり旧閾値が最適でなくなる。特徴量変更 → 学習 → Optuna（ハイパラ+閾値再探索） → MC確認を1サイクルとする。旧閾値での ablation 結果を最終判断にしない
- **特徴量の事前評価は NDCG/Top1 で行う**: Optuna 再探索は高コスト。特徴量追加前にまず同一ハイパラで WF-CV の NDCG@6 / Top-1 精度を N feat vs N+1 feat で比較する。差がなければ情報量がないので Optuna に進まない。差があれば Optuna → MC のフルサイクルへ
- **Optuna 目的関数は growth を使う**: Sharpe は「量で安定」に収束し過剰購入、Kelly は「質で ROI」に収束し過少購入。growth = `mean(log(1 + daily_fold_profit / bankroll))` が量×質を直接最適化する。購入数を減らす方向の最適化は実運用で買えないリスクがあるため避ける
- **Optuna は 2 フェーズ反復探索**: WF-CV はハイパラと閾値を同時に最適化すると、fold 再学習によりモデルが閾値レジームに適応し、固定モデル OOS で乖離する。Phase 1 の固定閾値と Phase 2 の最適閾値がずれるため、反復で収束させる:
  1. Round 1: Phase 1(`--fix-thresholds b1=0.35,ev=0.29` + `--validate-top 10`) → Phase 2(`--threshold-sweep`) → 最適閾値を取得
  2. Round 2: Phase 1(Round 1 の最適閾値で固定) → Phase 1.5 → Phase 2 → 収束確認
  3. 閾値が変わらなくなったら終了（通常 2-3 ラウンド）

## 自動運用デーモン

2プロセス構成。同一イメージで起動コマンドを分岐する。

| デーモン | コマンド | 役割 |
|---------|---------|------|
| scrape-daemon | `bun run start scrape-daemon` | データ収集（before-info, odds, results） |
| runner | `bun run start run` | ML 推論 + EV 判定 + 購入判断 |

runner は scrape-daemon が DB に書いたデータを読んで動作する。scrape-daemon 必須。

### runner の状態モデル（データ有無ベース）

```
[waiting] → 締切7分前 → [active] → 購入判断完了 → [decided] → 結果処理 → [done]
                          ↑ DB にデータがあれば処理、なければ次回ポーリングで再チェック
```

- `active` 内の処理は DB 状態で判断（状態の巻き戻しなし）:
  - T-5 odds がある AND 予測キャッシュなし → 予測実行
  - T-1 以内 AND T-3+T-1 odds がある → drift 外挿 + 購入判断
  - 締切超過 → auto-skip
- 起動時: `setupDay()` で DB からスケジュール読み込み → stats snapshot 構築
- ポーリング: 10秒間隔（T-1 前は5秒）
- 通知: Slack Webhook（`SLACK_WEBHOOK_URL` 環境変数）。未設定時はコンソール出力
- 終了: 全レース完了で自動終了、または SIGINT/SIGTERM で graceful shutdown
- **最初のレースは 8:30 JST の場合がある**（モーニングレース）。10:30 開始と思い込まない

### 設定（.env）

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## 開発ワークフロー

- TS ファイル編集後は必ず `bun run lint:fix` → lint/typecheck 確認
- Python ファイル編集後は `uv run pytest` で確認
- テストは Co-location（ソースと同じディレクトリ）
- パーサー・データ変換は構造的性質をテストする（parseOdds3T 全データ汚染の教訓）
- **サーバ変更コマンド**（data sync, data push）は実行前にソース確認。--dry-run が全パスで参照されているか確認する

## スクレイパー

- プラグイン式アーキテクチャ（`Scraper` インターフェース + レジストリ）
- HTML キャッシュ（gzip 圧縮、YYYYMM サブディレクトリ分割）
- 並列処理: 会場間8並列 + 同一レース3ページ並列取得
- 部分 HTML 抽出（`extractSections`）による高速パース
- `--from-cache`: キャッシュからのみパース（HTTP fetch なし）
- `--cache-only`: HTML ダウンロードのみ（パースなし）
- Zod safeParse でパーサー出力を検証
- boatrace.jp は認証不要（tateyamakun との主な差異）
