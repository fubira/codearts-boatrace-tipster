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

run [--dry-run(default)] [--live] [--ev-threshold N] [--bet-cap 30000]
    [--unit-divisor 200] [--bankroll 70000]

data test                        # ローカル整合性チェック
data verify                      # サーバとの整合性比較
data sync [--db-only] [--cache-only] [--dry-run] [--push]  # サーバ同期（フラグなしで snapshot も同期）
data fingerprint                 # DB統計表示

backup [-n 7]                    # ローカルバックアップ（N世代ローテーション）
```

### Python ML（`cd ml && uv run python scripts/<script>`）

```
predict_p2.py --date DATE [--snapshot PATH] [--race-ids ID,...] [--use-snapshots]
              [--model-dir models/<active>] [--db-path PATH]

tune_p2.py --trials N [--seed 42] [--n-folds 4] [--fold-months 2]
           [--relevance SCHEME] [--objective growth|kelly]
           [--fix-thresholds "gap23=0.13,top3_conc=0.6,gap12=0.04"]
           [--from-model models/<active>[,models/x]] [--narrow]
           [--n-jobs N] [--num-threads N]
           ※ --n-jobs > 1 は BOATRACE_TUNE_PARALLEL=1 必須（server-tune.sh が自動セット）

train_ranking.py --save [--model-dir models/draft/ranking] [--model-meta DIR]
                 [--end-date DATE] [--n-estimators N] [--learning-rate N]
                 [--relevance SCHEME] ...LGBMオーバーライド

build_snapshot.py --through-date DATE [--db-path PATH] [--output PATH]
verify_snapshot.py --date DATE [--db-path PATH] [--snapshot PATH]
daily_p2_summary.py --from DATE --to DATE [--model-dir DIR] [--db-path PATH]

analyze_model.py --from DATE --to DATE [--model-dir DIR]
                 [--split-by none|quarter|month] [--stadium CSV]
                 [--show-importance] [--json]

threshold_sweep.py --from DATE --to DATE [--model-dir DIR]
                   [--axis conc|ev|gap23|gap12] [--start N] [--stop N] [--step N]

# 診断ツール（gap12 のような新フィルタ軸の cliff 探索 / 開催日効果検証）
filter_axis_scan.py --from DATE --to DATE [--model-dir DIR]
                    --axis gap34|p1|entropy   # 5分位層別 + 閾値スイープで cliff を探す
tournament_day_analysis.py --from DATE --to DATE [--model-dir DIR]
                           # 開催日(1-6) で baseline 性能を層別、月別再現性チェック付き
seed_stability_check.py --tune-log PATH --top-n N --seeds 42,100,200,300,400 \
                        --from DATE --to DATE [--gap12-th 0.04]
                        # tune の上位 N 個を K seed で再学習 → OOS 安定性検証 (winner's curse 検出)
                        # ディスクに model を残さず in-memory で完結 (~30s/trial-seed)

# 運用診断ツール（日次 / 週次で状況確認）
analyze_decisions.py [--date YYYY-MM-DD] [--model-dir DIR]
                     # 1 日の P2 判定を bucket (BOUGHT/gap12_low/conc_low/gap23_low/ev_low) 別に分類、
                     # hit 率・borderline skip・"閾値を緩めた場合の追加 hit" を表示
                     # --date 省略で DB 最新日付。日次レビューに使う
compare_miss_patterns.py [--recent-days 7] [--baseline-days 30] [--to YYYY-MM-DD]
                         # recent vs baseline で miss 内訳 (boat1 NOT 1st / ordering wrong) を比較
                         # 「最近のハズレが過去と同じ分布か」を確認、週次 health check
                         # --recent-from / --recent-to / --baseline-from / --baseline-to で手動指定可

analyze_t5_t1_drift.py --from DATE --to DATE [--model-dir DIR]
                       # T-5 買い / T-1 drift 後 / 確定オッズ参考の 3 path で購入数・hit・ROI・P/L を比較
                       # T-1 drop rate (T-5 で EV+ → T-1 で EV- に落ちた tickets 率) も計算
                       # 期間は race_odds_snapshots 存在日 (2026-04-07 以降) に限定。
                       # 「モデルの選ぶ券が市場で削られやすいか」を定期監視 (月次 health check)

simulate_p2_mc.py --from DATE --to DATE [--model-dir DIR]
                  [--bankroll N] [--unit-divisor N] [--bet-cap N]
                  [--n-sims N] [--seed N] [--days 30,90,180,365]

promote_model.py [--draft models/draft] [--prod models/<active>]
                 [--component ranking] [--yes]

# 旧戦略スクリプト（backtest 用に温存、runner からは呼ばれない）
predict_trifecta.py / backtest_trifecta.py / tune_trifecta.py
```

### server-tune（`./scripts/server-tune.sh`）

P2 戦略 Optuna 探索をサーバで実行。**Phase 1 (tune) + Phase 2 (seed stability) をサーバで連続実行**し、kick 1 回 + fetch 1 回で完結する。

```
--setup                          # 初回セットアップ
--trials N                       # Optuna 実行 (Phase 1)
--watch                          # ログ監視
--fetch                          # 結果取得（log + trials.json、Phase 2 の ranking まで log に含まれる）
--fix-thresholds "gap23=0.13,top3_conc=0.6,gap12=0.04"  # 閾値固定でハイパラのみ探索
--from-model models/<active>  # 既存モデルHPをseedとして投入（カンマ区切り可）
--narrow                         # --from-model の最初のモデル周辺だけ探索
--n-jobs N                       # 並列 trial 数 (default: 2、サーバ専用)
--num-threads N                  # trial あたり LightGBM threads (default: 2、i7-6700 の半分)
--relevance podium               # relevance scheme 固定
--seed N                         # random seed (default: 自動ランダム、明示で再現性確保)

# Phase 2 (hit_pct top-N × 5 seed stability check、Phase 1 完了後にサーバで連続実行)
--phase2 N                       # hit_pct 上位 N を評価 (default: max(15, trials/15))
--no-phase2                      # Phase 2 を無効化 (Phase 1 だけ実行)
--phase2-from DATE               # OOS 評価期間 開始 (default: 2026-01-01)
--phase2-to DATE                 # OOS 評価期間 終了 (default: 今日)
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
  active.json         ← 本番モデル指定 ({"model": "<name>"})
  .run-counter        ← dev model 命名カウンタ（整数1個）
  draft/              ← 学習スクリプトのデフォルト保存先
    ranking/            model.pkl + model_meta.json
  tune_result/        ← Optuna 探索結果の保存先
  p2_v3/              ← 本番候補モデル (active.json で 1 つを選択)
    ranking/
  am_476/, ab_*/      ← dev candidate (prefix は自動採番)
```

- **active model**: `models/active.json` の `model` フィールドが本番。runner / predict / config はここを唯一の真実の源として参照する。切り替えは JSON を1行書き換えるだけ
- **runner は active.json を per-prediction で再読する**: `getActiveModelDir()` は `runPrediction` 内で呼ばれるので、active.json を書き換えると **次の predict から新モデルが使われる**。runner 再起動は不要。ただし `loadModelStrategy()` (gap12/conc/gap23/ev/div/cap/bankroll) は `run.ts` 起動時に 1 回だけ読まれるので、**strategy 値を変えたい場合は runner 再起動が必要**
- **dev prefix counter**: `models/.run-counter` は整数1個。`registry.py:next_prefix()` がこれを読んで `aa, ab, ac, ..., az, ba, ..., zz` を採番（fixed 2-letter base-26）
- 学習: `train_ranking.py` → `models/draft/` に保存
- 探索: `tune_p2.py` → `models/tune_result/` に保存
- 昇格: `promote_model.py` で draft → active model にコピー
- 推論: `predict_p2.py` → active model を読む

### 本番サーバへのモデル同期

モデルは Docker イメージに含めず、ボリュームマウントで配信する (tag push → CI → ghcr.io → watchtower auto-pull はコードのみ)。新モデルを本番 runner に反映する手順:

```bash
# 1. モデルディレクトリを rsync (ローカル → サーバ)
rsync -av ml/models/p2_v3/ one:${PRODUCT_DIR}/ml/models/p2_v3/

# 2. md5 で転送整合性を確認
ssh one "md5sum ${PRODUCT_DIR}/ml/models/p2_v3/ranking/model.pkl"
md5sum ml/models/p2_v3/ranking/model.pkl  # ローカル側、両方一致を確認

# 3. active.json を atomic に更新
ssh one 'cat > '${PRODUCT_DIR}'/ml/models/active.json.new <<EOF
{
  "model": "p2_v3"
}
EOF
mv '${PRODUCT_DIR}'/ml/models/active.json.new '${PRODUCT_DIR}'/ml/models/active.json'
```

**注意**:
- rsync は非破壊 (新 dir を追加するだけ)、前モデルは残る
- active.json の書き換えは `.new` 経由の `mv` で atomic に行う (読み込み中の半端な状態を防ぐ)
- hot swap (次の prediction から切替) は **モデルディレクトリだけ** 反映される。**strategy パラメータ変更 (div / bankroll / threshold)** が必要な場合は **runner 停止 → 設定変更 → 再起動** が必要
- 同じ model_meta.json の strategy 値で HP だけ差し替える場合は hot swap で十分

### モデル構成

Non-odds 特徴量のランキングモデル単体で完結。b1 二値分類モデルは不要。

| モデル | 目的 | 手法 | 用途 |
|--------|------|------|------|
| 6艇ランキング | 着順予測 | LGBMRanker LambdaRank (Non-odds 21特徴量) | 1着・2着予測 |

### EV 戦略（3連単 P2 adaptive）

**概要**: B1 が1着になるレースで、2-3着を P2 パターン（2点）で購入。3連単オッズから券ごとに EV 判定し、EV+ の券だけ adaptive に購入。

**フィルタ**（順序: 上から評価し早期 continue）:
1. `strategy.excluded_stadiums` に含まれる会場は全スキップ（採算性の構造的に悪い会場を除外）
2. **欠場検知**: 3連単 odds の 1 位置艇集合が 6 未満なら `withdrawal` で skip。`MIN_COMBOS_FOR_WITHDRAWAL_DETECTION=20` で partial loading の誤検知を防ぐ
3. Top-1 予測が1号艇であること
4. **gap12**（`P_rank1 - P_rank2`）≥ 閾値 → モデルが 1 号艇 1 着に自信を持っているレースのみ。低 gap12 は P2 戦略のコア前提が崩れている領域
5. top3_concentration（`(P_rank2+P_rank3)/(1-P_rank1)`）≥ 閾値 → top3 が下位から分離しているレースを選ぶ
6. gap23（2位-3位のスコア差）≥ 閾値 → 2着予測に自信があるレースを選ぶ
7. 各券の 3連単オッズから `EV = model_prob × odds × 0.75 − 1 ≥ 閾値`（0.75 は控除率 25% の補正だが、`odds` が既に控除後の値なので **二重カウント**になっている。結果 EV=0 は「真の期待値 +33%」を要求する過剰保守設定で、EV=−0.25 が真の損益分岐点。詳細は memory `knowledge_ev_formula_semantics.md`）

**買い方**: P2 = 1-(rank2,rank3)-(rank2,rank3)。最大2点、EV- の券はスキップ。
**ベットサイジング**: `unit = bankroll / divisor`（default: 200）、cap 3万円。`unit × 券数` が1レースの投資額
**EV 判定**: 単勝オッズは使わない（プール薄い）。**各券の3連単オッズ**から直接市場確率を計算
**探索**: `tune_p2.py` で gap12・gap23・top3_conc・EV 閾値・ハイパラを同時最適化（growth 目的関数）
**返還処理**: `race_entries.finish_position IS NULL` の艇を refunded boat とみなし、その艇を含む券は stake が bankroll に戻る（loss 計上しない）。フライング失格・欠場どちらにも対応

### 特徴量パイプライン

2つのモード: フルパイプライン（学習用）と snapshot パイプライン（推論用）。

**フルパイプライン**（`build_features_df()`）:
- DB 全件読み込み → 累積統計計算 → 日付フィルタ → 相対・交互作用特徴量
- `course_number` は常に `boat_number`（枠なり前提）。実コースは `actual_course_number` に保持し `course_taking_rate` の計算にのみ使用
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

`boatrace_tipster_ml/feature_config.py` が単一の真実源。

- `FEATURES`: P2 用 Non-odds 21特徴量。tune_p2 / train_ranking / predict_p2 / _p2_decision / 各診断スクリプトが参照
- `FEATURE_COLS`: 旧戦略用32特徴量（backtest_trifecta.py 等で使用）

### モデルの保存と推論

- モデルは `ml/models/<name>/ranking/` に保存。本番選択は `models/active.json` 経由
- `model_meta.json` がハイパラ・閾値の唯一の真実の源（全スクリプトがここから読む）
- 学習と運用は分離。同じモデルでも DB 内容の差で累積特徴量が変わる。学習はローカル→モデルをデプロイ。運用側は保存モデルで推論のみ

### 漏洩管理

- **予測時に不明な情報は特徴量に使わない**: `course_number`（実際の進入コース）はレース後にしか確定しないため、`boat_number` で代替する。`COALESCE` 等でフォールバックすると backtest でだけ実コースが漏洩する
- gate_bias / upset_rate は学習時に漏洩あり（intraday leakage）で木構造を改善し、評価/predict 時は `neutralize_leaked_features()` でレース内 mean に置換する。学習時の漏洩を除去してはならない

## ML 運用ルール

- **評価は OOS のみ**: 保存済みモデルの評価は学習期間外（OOS）でのみ行う。インサンプル期間は過学習で膨らむため評価に使わない。本番モデルの `end_date` は `model_meta.json` の `training.end_date` を確認し、分析期間は必ずそれ以降で指定する
- **本番モデル学習は `--end-date 2026-01-01` 必須**: `train_ranking.py` はデフォルトで全データを使う。OOS 期間（2026-01〜）にデータが漏れるので、本番モデル保存時は必ず `--end-date` を指定する
- **WF-CV は探索専用**: WF-CV は毎回モデルを再学習するため、保存済み本番モデルの評価には使わない。ハイパラ探索=WF-CV、本番モデル評価=保存済みモデルで `daily_p2_summary.py` または `analyze_model.py` (会場別 / 期間別 / feature importance)
- **Optuna seed**: TPESampler seed=42 固定だと同一 trials 数で同じ結果になる。探索拡大には seed を変えるか trials を増やす
- **WF-CV 直列実行**: WF-CV を並列実行するとメモリ・CPU が溢れてマシンがフリーズする。1つずつ
- **MC は P2 用 `simulate_p2_mc.py` を使う**: 旧 `simulate_monte_carlo.py` は X-allflow 戦略前提なので P2 には使えない。P2 では `analyze_model` の OOS 結果からパラメータを自動抽出する `simulate_p2_mc.py` を使うこと（hit_rate/bets_per_day/tickets_per_bet/payout 分布を実モデル実測値から取得）
- **特徴量変更は Optuna 再探索とセット**: 特徴量を変えるとスコア分布が変わり旧閾値が最適でなくなる。特徴量変更 → 学習 → Optuna（ハイパラ+閾値再探索） → MC確認を1サイクルとする。旧閾値での ablation 結果を最終判断にしない
- **特徴量の事前評価は NDCG/Top1 で行う**: Optuna 再探索は高コスト。特徴量追加前にまず同一ハイパラで WF-CV の NDCG@6 / Top-1 精度を N feat vs N+1 feat で比較する。差がなければ情報量がないので Optuna に進まない。差があれば Optuna → MC のフルサイクルへ
- **Optuna 目的関数は growth を使う**: Sharpe は「量で安定」に収束し過剰購入、Kelly は「質で ROI」に収束し過少購入。growth = `mean(log(1 + daily_fold_profit / bankroll))` が量×質を直接最適化する。購入数を減らす方向の最適化は実運用で買えないリスクがあるため避ける
- **Optuna 閾値 gap12/conc/gap23 は構造的な値として固定**: `gap12=0.04 / top3_conc=0.60 / gap23=0.13` は podium relevance で HP 非依存で P/L 最適。HP 変更時に再探索しない。`--fix-thresholds "gap23=0.13,top3_conc=0.6,gap12=0.04"` を tune の標準。**例外**: 新 relevance_scheme (top_heavy 等) を試す時は sweep 必須。**定期再確認**: 新 OOS 30 日蓄積後に全 HP 共通で sweep し構造変化がないか確認 (年 1-2 回)
- **EV 閾値は構造ではなく要実測**: 2026-04-17 の am_476 実測で P/L 最適点は **EV=-0.22**（full OOS +50,570 / late OOS +28,730）。現行 `ev=0.0` は EV 公式の 0.75 補正による控除率二重カウントで「真の期待値 +33%」を要求する過剰保守設定。運用変更は dry-run で empirical hit 実証を経て段階的に（`knowledge_ev_formula_semantics.md` 参照）
- **tune_p2 は early stopping を使わない**: `early_stopping_rounds=None` 必須。1 ヶ月 val window は noise dominated で best_iter が train の微小変化で 2x 跳ねるため。full training で WF-CV と production の training regime を一致させる
- **Tune は 2 phase 構成**: Phase 1 = WF-CV (1 seed, growth objective)、Phase 2 = **hit_pct 上位 N** を 5 seed で再評価し `stability_score = mean - std` で ranking。**Phase 1 の growth/P/L/Kelly で selection しない** (2026-04-17 az/ba run 計測で Kelly は Phase 2 stab と負相関 ρ≈-0.2、hit_pct が +0.33-0.56 で最強の単一 signal と判明)。Phase 2 は `server-tune.sh --trials N` 実行時にサーバで Phase 1 完了後に自動連続実行される (`--phase2` 省略時は `max(15, trials/15)` で auto-scale)
- **full OOS データを使い regime-robust metric で選定する**: full OOS の mean P/L は high-variance HP (幸運な window で稼ぐ) を favor するので、**min-based / worst-case-based metric** (例: `min(calendar month P/L)`、全月 P/L > 0) を full data に適用する。評価 window の切り方は **カレンダー月** を使い、指針上「都合よく」定めた regime 境界 (観察で見えた特定日付) は使わない。seed_stability_check が完全な暦月に自動分割して per-window stats を出す
- **live 切替前に dry-run で empirical hit 実証**: 新モデル / 新 config の live 運用入れ替えには、数値分析 (BT / MC / OOS / Phase 2) だけでなく **dry-run で実際に hit が出ることを確認**する。1-2 週間の観測期間で複数回の empirical hit がない限り live 切替を保留。詳細は memory `feedback_empirical_hit_validation.md`

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
  - T-5 odds がある AND 予測キャッシュなし → predict_p2 実行
  - T-1 以内 AND T-1 odds がある → P2 券のオッズ直接ルックアップ + EV 再計算 + 購入判断
  - 締切超過 → auto-skip
- 起動時: `setupDay()` で DB からスケジュール読み込み → stats snapshot 構築 → 同日 snapshot があれば bets/results/skipCounts を `runner-state.json` から復元
- **状態永続化**: `data/runner-state.json` に bankroll + today snapshot (bets/results/skipCounts/t1DroppedTickets) を atomic に保存。BetDecision には ticket combos も含まれるので、再起動後の result phase でも返還検知・hit 検知が完全復元できる。day rollover で snapshot は破棄、bankroll のみ保持
- ポーリング: 10秒間隔（T-1 前は5秒）
- 通知: Slack Webhook（`SLACK_WEBHOOK_URL` 環境変数）。未設定時はコンソール出力。result tag は WIN/REFUND/LOSE
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
- **ML リリース前**: P2 戦略は ranking 単体のため、旧 trifecta 用一貫性テストは廃止。predict_p2 の動作確認は `predict_p2.py --date DATE --model-dir models/<active> --json` で実施する
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
