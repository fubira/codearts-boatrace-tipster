"""Analyze boat 1 loss patterns to improve anti-favorite strategy.

Identifies conditions under which boat 1 (1号艇) loses, and checks
which patterns the current model fails to detect.

Usage:
    uv run --directory ml python -m scripts.analyze_boat1 [options]

Options:
    --relevance SCHEME  Relevance scheme (default: top_heavy)
"""

import argparse
import time

import numpy as np
import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.evaluate import evaluate_model
from boatrace_tipster_ml.features import build_features
from boatrace_tipster_ml.model import time_series_split, train_model

DB_PATH = DEFAULT_DB_PATH
FIELD_SIZE = 6


def _build_boat1_df(
    model, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame
) -> pd.DataFrame:
    """Build a DataFrame with per-race boat 1 analysis columns."""
    scores = model.predict(X)

    df = meta[["race_id", "boat_number"]].copy()
    df["score"] = scores
    df["actual_pos"] = y.values

    # Merge feature columns needed for analysis
    for col in [
        "exhibition_time", "rel_exhibition_time", "rel_exhibition_st",
        "rel_national_win_rate", "tourn_st_delta", "tourn_exhibition_delta",
        "tourn_avg_position", "stadium_course_win_rate", "course_number",
        "kado_x_exhibition", "class_x_boat", "boat_number",
    ]:
        if col in X.columns:
            df[col] = X[col].values

    n_races = len(df) // FIELD_SIZE
    scores_2d = df["score"].values.reshape(n_races, FIELD_SIZE)
    actual_2d = df["actual_pos"].values.reshape(n_races, FIELD_SIZE)
    boats_2d = df["boat_number"].values.reshape(n_races, FIELD_SIZE)

    # Predicted rank per boat
    pred_order = np.argsort(-scores_2d, axis=1)
    pred_ranks = np.empty_like(pred_order)
    rows = np.arange(n_races)[:, None]
    pred_ranks[rows, pred_order] = np.arange(1, FIELD_SIZE + 1)

    # Find boat 1 in each race
    boat1_col = np.argmax(boats_2d == 1, axis=1)  # column index of boat 1

    result = pd.DataFrame({
        "race_id": df["race_id"].values.reshape(n_races, FIELD_SIZE)[:, 0],
        "boat1_actual_pos": actual_2d[np.arange(n_races), boat1_col],
        "boat1_pred_rank": pred_ranks[np.arange(n_races), boat1_col],
        "boat1_score": scores_2d[np.arange(n_races), boat1_col],
        "max_other_score": np.array([
            np.max(np.delete(scores_2d[i], boat1_col[i])) for i in range(n_races)
        ]),
    })
    result["boat1_lost"] = result["boat1_actual_pos"] > 1
    result["model_predicts_loss"] = result["boat1_pred_rank"] > 1
    result["score_gap"] = result["boat1_score"] - result["max_other_score"]

    # Merge boat 1 features
    boat1_rows = df[df["boat_number"] == 1].reset_index(drop=True)
    for col in [
        "rel_exhibition_time", "rel_exhibition_st",
        "tourn_st_delta", "tourn_exhibition_delta",
        "tourn_avg_position", "stadium_course_win_rate",
        "course_number", "kado_x_exhibition", "class_x_boat",
    ]:
        if col in boat1_rows.columns:
            result[f"b1_{col}"] = boat1_rows[col].values

    return result


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def analyze_basic_stats(b1: pd.DataFrame) -> None:
    """Basic boat 1 win/loss statistics."""
    _section("1号艇 基本統計")
    n = len(b1)
    n_won = (b1["boat1_actual_pos"] == 1).sum()
    n_lost = b1["boat1_lost"].sum()
    print(f"レース数: {n}")
    print(f"1号艇勝利: {n_won} ({n_won/n:.1%})")
    print(f"1号艇敗北: {n_lost} ({n_lost/n:.1%})")

    print(f"\n1号艇の着順分布:")
    for pos in range(1, 8):
        cnt = (b1["boat1_actual_pos"] == pos).sum()
        bar = "█" * int(cnt / n * 100)
        print(f"  {pos}着: {cnt:5d} ({cnt/n:5.1%}) {bar}")


def analyze_model_detection(b1: pd.DataFrame) -> None:
    """How well the model detects boat 1 losses."""
    _section("モデルの1号艇飛び検出")
    lost = b1[b1["boat1_lost"]]
    n_lost = len(lost)

    detected = lost[lost["model_predicts_loss"]]
    n_detected = len(detected)
    print(f"1号艇敗北: {n_lost} レース")
    print(f"モデルが検出 (pred_rank > 1): {n_detected} ({n_detected/n_lost:.1%})")
    print(f"見逃し (pred_rank == 1): {n_lost - n_detected} ({(n_lost - n_detected)/n_lost:.1%})")

    # Detection by confidence (score_gap)
    print(f"\n検出率 by score_gap しきい値:")
    for pct in [0, 10, 25, 50, 75]:
        if pct == 0:
            threshold = -999
        else:
            threshold = np.percentile(b1["score_gap"], pct)
        # "score_gap < threshold" means model is less confident about boat 1
        suspicious = lost[lost["score_gap"] < threshold]
        print(f"  p{pct:>2d} (gap < {threshold:+.3f}): 検出={len(suspicious):4d}/{n_lost} ({len(suspicious)/n_lost:.1%})")


def analyze_feature_conditions(b1: pd.DataFrame) -> None:
    """Analyze conditions that correlate with boat 1 losses."""
    _section("飛び条件の分析")

    conditions = []

    # --- Condition 1: Exhibition ST ---
    if "b1_rel_exhibition_st" in b1.columns:
        print("\n--- 展示ST (rel_exhibition_st) ---")
        print("  正値 = 同レース内で遅い ST")
        for thr in [0.0, 0.3, 0.5, 0.8, 1.0]:
            mask = b1["b1_rel_exhibition_st"] > thr
            n_match = mask.sum()
            if n_match < 10:
                continue
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  rel_ex_st > {thr:.1f}: {n_match:4d}R  飛率={loss_rate:.1%}  検出={detected:.1%}")
            conditions.append(("rel_exhibition_st", thr, n_match, loss_rate, detected))

    # --- Condition 2: Exhibition time (relative) ---
    if "b1_rel_exhibition_time" in b1.columns:
        print("\n--- 展示タイム (rel_exhibition_time) ---")
        print("  正値 = 同レース内で遅い（悪い）")
        for thr in [0.0, 0.3, 0.5, 0.8, 1.0]:
            mask = b1["b1_rel_exhibition_time"] > thr
            n_match = mask.sum()
            if n_match < 10:
                continue
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  rel_ex_time > {thr:.1f}: {n_match:4d}R  飛率={loss_rate:.1%}  検出={detected:.1%}")
            conditions.append(("rel_exhibition_time", thr, n_match, loss_rate, detected))

    # --- Condition 3: Tournament ST delta ---
    if "b1_tourn_st_delta" in b1.columns:
        print("\n--- 開催内ST変化 (tourn_st_delta) ---")
        print("  正値 = 開催内でSTが悪化している")
        valid = b1["b1_tourn_st_delta"].notna()
        for thr in [0.0, 0.01, 0.02, 0.03]:
            mask = valid & (b1["b1_tourn_st_delta"] > thr)
            n_match = mask.sum()
            if n_match < 10:
                continue
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  tourn_st_delta > {thr:.2f}: {n_match:4d}R  飛率={loss_rate:.1%}  検出={detected:.1%}")

    # --- Condition 4: Tournament exhibition delta ---
    if "b1_tourn_exhibition_delta" in b1.columns:
        print("\n--- 開催内展示タイム変化 (tourn_exhibition_delta) ---")
        print("  正値 = 開催内で展示タイムが悪化")
        valid = b1["b1_tourn_exhibition_delta"].notna()
        for thr in [0.0, 0.05, 0.10, 0.15]:
            mask = valid & (b1["b1_tourn_exhibition_delta"] > thr)
            n_match = mask.sum()
            if n_match < 10:
                continue
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  tourn_ex_delta > {thr:.2f}: {n_match:4d}R  飛率={loss_rate:.1%}  検出={detected:.1%}")

    # --- Condition 5: Stadium course win rate ---
    if "b1_stadium_course_win_rate" in b1.columns:
        print("\n--- 会場コース勝率 (stadium_course_win_rate) ---")
        print("  低い = この会場で1コースが弱い")
        valid = b1["b1_stadium_course_win_rate"].notna()
        for thr in [0.60, 0.55, 0.50, 0.45]:
            mask = valid & (b1["b1_stadium_course_win_rate"] < thr)
            n_match = mask.sum()
            if n_match < 10:
                continue
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  stadium_win_rate < {thr:.2f}: {n_match:4d}R  飛率={loss_rate:.1%}  検出={detected:.1%}")

    # --- Condition 6: Course number != 1 (前付けされた) ---
    if "b1_course_number" in b1.columns:
        print("\n--- コース (course_number) ---")
        mask = b1["b1_course_number"] > 1
        n_match = mask.sum()
        if n_match >= 5:
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  1号艇が1コース以外: {n_match}R  飛率={loss_rate:.1%}  検出={detected:.1%}")
        else:
            print(f"  1号艇が1コース以外: {n_match}R（少なすぎて分析不可）")

    # --- Condition 7: class_x_boat (相手の質) ---
    if "b1_class_x_boat" in b1.columns:
        print("\n--- クラス×枠番 (class_x_boat) ---")
        print("  低い = 1号艇のクラスが低いか外枠（不利）")
        for pct in [25, 50]:
            thr = np.percentile(b1["b1_class_x_boat"].dropna(), pct)
            mask = b1["b1_class_x_boat"] < thr
            n_match = mask.sum()
            if n_match < 10:
                continue
            loss_rate = b1.loc[mask, "boat1_lost"].mean()
            detected = b1.loc[mask & b1["boat1_lost"], "model_predicts_loss"].mean()
            print(f"  class_x_boat < p{pct} ({thr:.1f}): {n_match:4d}R  飛率={loss_rate:.1%}  検出={detected:.1%}")


def analyze_compound_conditions(b1: pd.DataFrame) -> None:
    """Analyze compound conditions for higher detection rates."""
    _section("複合条件の分析")

    def _report(label: str, mask: pd.Series) -> None:
        n_match = mask.sum()
        if n_match < 10:
            print(f"  {label}: {n_match}R（少なすぎ）")
            return
        loss_rate = b1.loc[mask, "boat1_lost"].mean()
        lost_in_group = mask & b1["boat1_lost"]
        n_lost_detected = (lost_in_group & b1["model_predicts_loss"]).sum()
        n_lost_in_group = lost_in_group.sum()
        detect_rate = n_lost_detected / n_lost_in_group if n_lost_in_group > 0 else 0
        print(f"  {label}: {n_match:4d}R  飛率={loss_rate:.1%}  "
              f"飛び={n_lost_in_group}  検出={n_lost_detected}({detect_rate:.1%})")

    has_st = "b1_rel_exhibition_st" in b1.columns
    has_ex = "b1_rel_exhibition_time" in b1.columns
    has_td = "b1_tourn_st_delta" in b1.columns
    has_sw = "b1_stadium_course_win_rate" in b1.columns
    has_te = "b1_tourn_exhibition_delta" in b1.columns

    # Compound: slow ST + slow exhibition
    if has_st and has_ex:
        print("\n--- 展示ST遅い AND 展示タイム遅い ---")
        for st_thr in [0.0, 0.3, 0.5]:
            for ex_thr in [0.0, 0.3, 0.5]:
                mask = (b1["b1_rel_exhibition_st"] > st_thr) & (b1["b1_rel_exhibition_time"] > ex_thr)
                _report(f"ST>{st_thr:.1f} & TIME>{ex_thr:.1f}", mask)

    # Compound: slow ST + tournament ST worsening
    if has_st and has_td:
        print("\n--- 展示ST遅い AND 開催内ST悪化 ---")
        valid_td = b1["b1_tourn_st_delta"].notna()
        for st_thr in [0.0, 0.3]:
            for td_thr in [0.0, 0.01, 0.02]:
                mask = valid_td & (b1["b1_rel_exhibition_st"] > st_thr) & (b1["b1_tourn_st_delta"] > td_thr)
                _report(f"ST>{st_thr:.1f} & tourST>{td_thr:.2f}", mask)

    # Compound: slow ST + weak venue
    if has_st and has_sw:
        print("\n--- 展示ST遅い AND 会場コース勝率低い ---")
        valid_sw = b1["b1_stadium_course_win_rate"].notna()
        for st_thr in [0.0, 0.3]:
            for sw_thr in [0.55, 0.50]:
                mask = valid_sw & (b1["b1_rel_exhibition_st"] > st_thr) & (b1["b1_stadium_course_win_rate"] < sw_thr)
                _report(f"ST>{st_thr:.1f} & venue<{sw_thr:.2f}", mask)

    # Compound: exhibition time worsening + slow exhibition
    if has_te and has_ex:
        print("\n--- 開催内展示悪化 AND 展示タイム遅い ---")
        valid_te = b1["b1_tourn_exhibition_delta"].notna()
        for te_thr in [0.0, 0.05]:
            for ex_thr in [0.0, 0.3]:
                mask = valid_te & (b1["b1_tourn_exhibition_delta"] > te_thr) & (b1["b1_rel_exhibition_time"] > ex_thr)
                _report(f"tourEX>{te_thr:.2f} & TIME>{ex_thr:.1f}", mask)

    # Triple compound: slow ST + slow time + tournament worsening
    if has_st and has_ex and has_td:
        print("\n--- 展示ST遅い AND タイム遅い AND 開催内悪化 ---")
        valid_td = b1["b1_tourn_st_delta"].notna()
        mask = valid_td & (b1["b1_rel_exhibition_st"] > 0.0) & (b1["b1_rel_exhibition_time"] > 0.0) & (b1["b1_tourn_st_delta"] > 0.0)
        _report("ST>0 & TIME>0 & tourST>0", mask)
        mask = valid_td & (b1["b1_rel_exhibition_st"] > 0.3) & (b1["b1_rel_exhibition_time"] > 0.3) & (b1["b1_tourn_st_delta"] > 0.0)
        _report("ST>0.3 & TIME>0.3 & tourST>0", mask)


def analyze_missed_losses(b1: pd.DataFrame) -> None:
    """Analyze characteristics of boat 1 losses the model missed."""
    _section("見逃し分析（モデルが1位予測したのに飛んだケース）")

    missed = b1[b1["boat1_lost"] & ~b1["model_predicts_loss"]]
    detected = b1[b1["boat1_lost"] & b1["model_predicts_loss"]]

    if len(missed) == 0 or len(detected) == 0:
        print("データ不足")
        return

    print(f"見逃し: {len(missed)} レース")
    print(f"検出済: {len(detected)} レース")

    # Compare feature distributions
    compare_cols = [
        ("b1_rel_exhibition_st", "展示ST"),
        ("b1_rel_exhibition_time", "展示タイム"),
        ("b1_tourn_st_delta", "開催内ST変化"),
        ("b1_tourn_exhibition_delta", "開催内展示変化"),
        ("b1_stadium_course_win_rate", "会場勝率"),
        ("b1_class_x_boat", "クラス×枠番"),
    ]

    print(f"\n{'特徴量':<20s}  {'見逃しmean':>10s} {'検出済mean':>10s} {'差':>8s}")
    print("-" * 60)
    for col, label in compare_cols:
        if col not in b1.columns:
            continue
        m_mean = missed[col].mean()
        d_mean = detected[col].mean()
        diff = m_mean - d_mean
        print(f"  {label:<18s}  {m_mean:>10.4f} {d_mean:>10.4f} {diff:>+8.4f}")

    # What % of missed losses had "should have been detectable" features
    print(f"\n見逃しの中で各条件に該当する割合:")
    if "b1_rel_exhibition_st" in missed.columns:
        pct = (missed["b1_rel_exhibition_st"] > 0.3).mean()
        print(f"  展示ST > 0.3:      {pct:.1%}")
    if "b1_rel_exhibition_time" in missed.columns:
        pct = (missed["b1_rel_exhibition_time"] > 0.3).mean()
        print(f"  展示タイム > 0.3:  {pct:.1%}")
    if "b1_tourn_st_delta" in missed.columns:
        valid = missed["b1_tourn_st_delta"].notna()
        if valid.sum() > 0:
            pct = (missed.loc[valid, "b1_tourn_st_delta"] > 0).mean()
            print(f"  開催内ST悪化:      {pct:.1%}")


def analyze_win_vs_loss_patterns(b1: pd.DataFrame) -> None:
    """Compare all feature distributions between boat 1 wins and losses.

    Reverse-engineers which features distinguish win/loss outcomes
    by computing effect size (Cohen's d) for each feature.
    """
    _section("勝ち/負けパターン逆引き分析")

    won = b1[~b1["boat1_lost"]]
    lost = b1[b1["boat1_lost"]]
    print(f"勝ち: {len(won)}R  負け: {len(lost)}R")

    # Compare all b1_ features
    feature_cols = [c for c in b1.columns if c.startswith("b1_")]
    if not feature_cols:
        print("  分析対象の特徴量がありません")
        return

    results = []
    for col in feature_cols:
        w = won[col].dropna()
        l = lost[col].dropna()
        if len(w) < 10 or len(l) < 10:
            continue
        w_mean = w.mean()
        l_mean = l.mean()
        pooled_std = np.sqrt((w.std()**2 + l.std()**2) / 2)
        if pooled_std == 0:
            continue
        cohens_d = (l_mean - w_mean) / pooled_std
        results.append({
            "feature": col.replace("b1_", ""),
            "win_mean": w_mean,
            "loss_mean": l_mean,
            "diff": l_mean - w_mean,
            "cohens_d": cohens_d,
            "abs_d": abs(cohens_d),
        })

    results.sort(key=lambda x: -x["abs_d"])

    print(f"\n  効果量ランキング (Cohen's d: 負け - 勝ち)")
    print(f"  {'特徴量':<28s} {'勝ちmean':>9s} {'負けmean':>9s} {'差':>8s} {'Cohen d':>8s}")
    print("  " + "-" * 70)
    for r in results:
        direction = "↑飛" if r["cohens_d"] > 0 else "↓飛"
        print(f"  {r['feature']:<28s} {r['win_mean']:>9.4f} {r['loss_mean']:>9.4f} "
              f"{r['diff']:>+8.4f} {r['cohens_d']:>+7.3f} {direction}")


def analyze_raw_data_patterns() -> None:
    """Analyze raw DB data patterns for boat 1 win/loss.

    Looks at features not yet in the model to find new signal sources.
    """
    _section("生データからの新規シグナル探索")

    conn = get_connection(DB_PATH)
    try:
        df = conn.execute("""
            SELECT
                r.id AS race_id,
                r.stadium_id,
                r.race_grade,
                r.weather,
                r.wind_speed,
                r.wind_direction,
                r.wave_height,
                r.temperature,
                r.water_temperature,
                r.race_number,
                re.boat_number,
                COALESCE(re.course_number, re.boat_number) AS course_number,
                re.racer_class,
                re.racer_weight,
                re.exhibition_time,
                re.exhibition_st,
                re.average_st,
                re.start_timing,
                re.national_win_rate,
                re.national_top2_rate,
                re.motor_top2_rate,
                re.motor_top3_rate,
                re.boat_top2_rate,
                re.local_win_rate,
                re.finish_position,
                re.flying_count,
                re.late_count,
                re.tilt,
                re.stabilizer
            FROM db.races r
            JOIN db.race_entries re ON re.race_id = r.id
            WHERE r.race_date >= '2025-10-01'
            ORDER BY r.race_date, r.id, re.boat_number
        """).fetchdf()
    finally:
        conn.close()

    # Keep only complete races
    rc = df.groupby("race_id").size()
    valid = rc[rc == 6].index
    df = df[df["race_id"].isin(valid)]

    # Split boat 1 and others
    b1 = df[df["boat_number"] == 1].copy()
    others = df[df["boat_number"] > 1].copy()
    b1["lost"] = b1["finish_position"] > 1

    # Per-race aggregates of opponents
    opp_agg = others.groupby("race_id").agg(
        opp_max_national_win=("national_win_rate", "max"),
        opp_mean_national_win=("national_win_rate", "mean"),
        opp_max_motor=("motor_top3_rate", "max"),
        opp_min_exhibition_time=("exhibition_time", "min"),
        opp_min_exhibition_st=("exhibition_st", "min"),
        opp_n_a1=("racer_class", lambda x: (x == "A1").sum()),
        opp_max_local_win=("local_win_rate", "max"),
        opp_mean_exhibition_st=("exhibition_st", "mean"),
    ).reset_index()

    b1m = b1.merge(opp_agg, on="race_id", how="left")
    won = b1m[~b1m["lost"]]
    lost = b1m[b1m["lost"]]

    # Analyze conditions
    analysis_cols = [
        # Boat 1 own features
        ("exhibition_time", "展示タイム"),
        ("exhibition_st", "展示ST"),
        ("average_st", "平均ST"),
        ("national_win_rate", "全国勝率"),
        ("motor_top3_rate", "モーターTop3率"),
        ("racer_weight", "体重"),
        ("flying_count", "F数"),
        ("late_count", "L数"),
        ("start_timing", "実ST"),
        ("tilt", "チルト"),
        # Race conditions
        ("wind_speed", "風速"),
        ("wave_height", "波高"),
        ("temperature", "気温"),
        ("water_temperature", "水温"),
        ("race_number", "レース番号"),
        # Opponent features
        ("opp_max_national_win", "相手最高勝率"),
        ("opp_mean_national_win", "相手平均勝率"),
        ("opp_max_motor", "相手最高モーター"),
        ("opp_min_exhibition_time", "相手最速展示"),
        ("opp_min_exhibition_st", "相手最速ST"),
        ("opp_n_a1", "相手A1人数"),
        ("opp_max_local_win", "相手最高地元勝率"),
        ("opp_mean_exhibition_st", "相手平均ST"),
    ]

    results = []
    for col, label in analysis_cols:
        if col not in b1m.columns:
            continue
        w = won[col].dropna()
        l = lost[col].dropna()
        if len(w) < 10 or len(l) < 10:
            continue
        w_mean = w.mean()
        l_mean = l.mean()
        pooled_std = np.sqrt((w.std()**2 + l.std()**2) / 2)
        if pooled_std == 0:
            continue
        cohens_d = (l_mean - w_mean) / pooled_std
        results.append({
            "col": col,
            "label": label,
            "win_mean": w_mean,
            "loss_mean": l_mean,
            "cohens_d": cohens_d,
            "abs_d": abs(cohens_d),
        })

    results.sort(key=lambda x: -x["abs_d"])

    print(f"\n  勝ち: {len(won)}R  負け: {len(lost)}R")
    print(f"\n  効果量ランキング（生データ）")
    print(f"  {'特徴量':<20s} {'勝ちmean':>10s} {'負けmean':>10s} {'Cohen d':>8s} {'方向':>6s}")
    print("  " + "-" * 62)
    for r in results:
        direction = "↑飛" if r["cohens_d"] > 0 else "↓飛"
        print(f"  {r['label']:<20s} {r['win_mean']:>10.4f} {r['loss_mean']:>10.4f} "
              f"{r['cohens_d']:>+7.3f}  {direction}")

    # Stadium-specific analysis
    print(f"\n--- 会場別 1号艇飛率 ---")
    stadium_stats = b1.groupby("stadium_id").agg(
        total=("lost", "count"),
        losses=("lost", "sum"),
    ).reset_index()
    stadium_stats["loss_rate"] = stadium_stats["losses"] / stadium_stats["total"]
    stadium_stats = stadium_stats.sort_values("loss_rate", ascending=False)
    overall_loss = b1["lost"].mean()
    for _, row in stadium_stats.head(10).iterrows():
        delta = row["loss_rate"] - overall_loss
        print(f"  会場{int(row['stadium_id']):>2d}: 飛率={row['loss_rate']:.1%} ({int(row['total'])}R) "
              f"{'↑' if delta > 0.02 else '↓' if delta < -0.02 else '='}{abs(delta):.1%}")

    # Weather impact
    print(f"\n--- 天候別 1号艇飛率 ---")
    weather_stats = b1.groupby("weather").agg(
        total=("lost", "count"),
        losses=("lost", "sum"),
    ).reset_index()
    weather_stats["loss_rate"] = weather_stats["losses"] / weather_stats["total"]
    for _, row in weather_stats.iterrows():
        if row["total"] >= 10:
            print(f"  {row['weather'] or '不明'}: 飛率={row['loss_rate']:.1%} ({int(row['total'])}R)")

    # Wind direction impact
    print(f"\n--- 風向別 1号艇飛率 ---")
    wind_stats = b1.groupby("wind_direction").agg(
        total=("lost", "count"),
        losses=("lost", "sum"),
    ).reset_index()
    wind_stats["loss_rate"] = wind_stats["losses"] / wind_stats["total"]
    wind_stats = wind_stats.sort_values("loss_rate", ascending=False)
    for _, row in wind_stats.iterrows():
        if row["total"] >= 20:
            print(f"  {row['wind_direction'] or '不明'}: 飛率={row['loss_rate']:.1%} ({int(row['total'])}R)")

    # Actual ST comparison (boat 1 vs winner when boat 1 lost)
    print(f"\n--- 実ST: 飛び時の1号艇 vs 勝者 ---")
    lost_races = b1m[b1m["lost"]]["race_id"]
    lost_entries = df[df["race_id"].isin(lost_races)]
    winners = lost_entries[lost_entries["finish_position"] == 1]
    b1_lost_st = b1m[b1m["lost"]]["start_timing"].dropna()
    winner_st = winners["start_timing"].dropna()
    if len(b1_lost_st) > 0 and len(winner_st) > 0:
        print(f"  1号艇の実ST平均: {b1_lost_st.mean():.4f}s")
        print(f"  勝者の実ST平均:  {winner_st.mean():.4f}s")
        print(f"  差: {b1_lost_st.mean() - winner_st.mean():+.4f}s")

        # Distribution of ST diff
        b1_lost_indexed = b1m[b1m["lost"]][["race_id", "start_timing"]].set_index("race_id")
        winner_indexed = winners[["race_id", "start_timing"]].set_index("race_id")
        common = b1_lost_indexed.index.intersection(winner_indexed.index)
        if len(common) > 0:
            st_diff = b1_lost_indexed.loc[common, "start_timing"] - winner_indexed.loc[common, "start_timing"]
            print(f"  ST差（1号艇-勝者）分布:")
            for pct in [25, 50, 75, 90]:
                print(f"    p{pct}: {np.percentile(st_diff.dropna(), pct):+.4f}s")
            late_pct = (st_diff > 0).mean()
            print(f"  1号艇がSTで負けている割合: {late_pct:.1%}")


def analyze_inya_impact(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, model) -> None:
    """Analyze impact of イン屋 (aggressive course-takers) on boat 1 losses.

    Loads course_taking_rate from the feature pipeline and checks
    if presence of high-rate outer boats correlates with boat 1 losses.
    """
    _section("イン屋の影響分析")

    # We need the raw feature data before FEATURE_COLS filtering
    # Rebuild a minimal dataset from DB
    conn = get_connection(DB_PATH)
    try:
        # Get course_taking_rate from the full feature pipeline
        # We'll compute it here from the raw data
        rows = conn.execute("""
            SELECT re.race_id, re.boat_number,
                   COALESCE(re.course_number, re.boat_number) AS course_number,
                   re.finish_position
            FROM db.races r
            JOIN db.race_entries re ON re.race_id = r.id
            WHERE r.race_date >= '2025-10-01'
            ORDER BY r.race_date, r.id, re.id
        """).fetchdf()
    finally:
        conn.close()

    if rows.empty:
        print("テスト期間のデータなし")
        return

    # Check if course_number != boat_number (front-taking happened)
    rows["took_inner"] = (rows["course_number"] < rows["boat_number"]).astype(int)

    # Per race: did any outer boat (boat >= 3) take an inner course?
    race_ids = rows["race_id"].unique()

    # Races where front-taking happened (any boat got a course < its boat number)
    ft_races = rows[rows["took_inner"] == 1]["race_id"].unique()
    n_ft = len(ft_races)
    n_total = len(race_ids)

    # Filter to complete races (6 entries)
    race_counts = rows.groupby("race_id").size()
    valid = race_counts[race_counts == 6].index
    rows = rows[rows["race_id"].isin(valid)]

    # Boat 1 outcomes
    b1_rows = rows[rows["boat_number"] == 1].set_index("race_id")

    # Races with front-taking
    ft_set = set(ft_races)
    b1_rows["has_front_taking"] = b1_rows.index.isin(ft_set)
    b1_rows["lost"] = b1_rows["finish_position"] > 1

    # Compare
    ft_group = b1_rows[b1_rows["has_front_taking"]]
    no_ft_group = b1_rows[~b1_rows["has_front_taking"]]

    print(f"テスト期間 前付け発生: {n_ft}/{n_total} ({n_ft/n_total:.1%})")
    if len(ft_group) > 0:
        print(f"  前付けあり: 1号艇飛率 = {ft_group['lost'].mean():.1%} ({len(ft_group)}R)")
    if len(no_ft_group) > 0:
        print(f"  前付けなし: 1号艇飛率 = {no_ft_group['lost'].mean():.1%} ({len(no_ft_group)}R)")

    # Deeper: boat 1 lost its course (got pushed to course 2+)
    b1_course = rows[rows["boat_number"] == 1][["race_id", "course_number"]].set_index("race_id")
    b1_lost_course = b1_course[b1_course["course_number"] > 1]
    print(f"\n  1号艇がコース1を失った: {len(b1_lost_course)}R")
    if len(b1_lost_course) > 0:
        lost_ids = set(b1_lost_course.index)
        b1_in_lost = b1_rows.loc[b1_rows.index.isin(lost_ids)]
        if len(b1_in_lost) > 0:
            print(f"    → 飛率: {b1_in_lost['lost'].mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze boat 1 loss patterns")
    parser.add_argument("--relevance", default="top_heavy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Relevance: {args.relevance}")
    print()

    t0 = time.time()
    X, y, meta = build_features(DB_PATH)
    print(f"Features: {X.shape[0]} entries ({X.shape[0]//6} races), {X.shape[1]} features")

    splits = time_series_split(X, y, meta)
    for name, data in splits.items():
        n_races = len(data["X"]) // 6
        print(f"  {name}: {n_races} races")

    print("\nTraining model...")
    model, metrics = train_model(
        splits["train"]["X"], splits["train"]["y"], splits["train"]["meta"],
        splits["val"]["X"], splits["val"]["y"], splits["val"]["meta"],
        relevance_scheme=args.relevance,
    )
    print(f"  Trained in {time.time() - t0:.1f}s")

    # Evaluate baseline
    print("\nBaseline evaluation:")
    result = evaluate_model(
        model, splits["test"]["X"], splits["test"]["y"], splits["test"]["meta"],
        db_path=DB_PATH,
    )
    acc = result["topNAccuracy"]
    print(f"  Top1={acc['1']:.1%} nDCG={result['avgNDCG']:.4f}")
    if "payoutROI" in result:
        for name, s in result["payoutROI"].items():
            print(f"  {name}: ROI={s['recoveryRate']:.1%}")

    # Build boat 1 analysis DataFrame
    b1 = _build_boat1_df(
        model, splits["test"]["X"], splits["test"]["y"], splits["test"]["meta"]
    )

    analyze_basic_stats(b1)
    analyze_model_detection(b1)
    analyze_feature_conditions(b1)
    analyze_compound_conditions(b1)
    analyze_missed_losses(b1)
    analyze_win_vs_loss_patterns(b1)
    analyze_raw_data_patterns()
    analyze_inya_impact(X, y, meta, model)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
