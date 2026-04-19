"""1 号艇の予測確率と実際の勝率の信頼性を区分別に集計する。

モデルが 1 号艇を過信しているか (= 予測確率が高い区分で実際の勝率が
それに届かないか) を直接可視化する。全レースとフィルタ通過レースの
両方で集計し、P2 フィルタが過信を強めているかも確認する。

Usage:
    uv run python -m scripts.analyze_reliability \\
        --from 2026-01-01 --to 2026-04-18 --model-dir models/p2_v3
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
from scripts.analyze_model import evaluate_period

FIELD_SIZE = 6


def _compute_boat1_prob_and_hit(
    model, meta: dict, df, from_date: str, to_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (p1, hit, gap12, conc, gap23, b1_is_rank1, race_ids)."""
    features = meta["feature_columns"]
    test = df[(df["race_date"] >= from_date) & (df["race_date"] < to_date)].copy()
    X = test[features].copy()
    fill_nan_with_means(X, meta)
    scores = model.predict(X)

    n = len(X) // FIELD_SIZE
    s2 = scores.reshape(n, FIELD_SIZE)
    b2 = test["boat_number"].values.reshape(n, FIELD_SIZE).astype(int)
    y2 = test["finish_position"].values.reshape(n, FIELD_SIZE)

    # softmax
    ex = np.exp(s2 - s2.max(axis=1, keepdims=True))
    mp = ex / ex.sum(axis=1, keepdims=True)

    # 1 号艇の位置 (boat_number=1) を各レースで特定
    # boat_number 配列で == 1 のインデックスを取る
    boat1_idx = np.argmax(b2 == 1, axis=1)  # (n,)
    p1 = mp[np.arange(n), boat1_idx]  # (n,) 1 号艇 softmax 確率

    # 実際の 1 着 boat (finish_position == 1 の boat)
    winner_pos = np.argmin(y2, axis=1)  # finish_position==1 のインデックス
    winner_boat = b2[np.arange(n), winner_pos]
    hit_boat1 = (winner_boat == 1).astype(int)

    # 予測ランキング (スコア降順) から top 3 の softmax prob
    po = np.argsort(-s2, axis=1)  # (n, 6)
    p_rank1 = mp[np.arange(n), po[:, 0]]
    p_rank2 = mp[np.arange(n), po[:, 1]]
    p_rank3 = mp[np.arange(n), po[:, 2]]
    gap12 = p_rank1 - p_rank2
    top3_conc = (p_rank2 + p_rank3) / (1 - p_rank1 + 1e-10)
    gap23 = p_rank2 - p_rank3

    # P2 フィルタ: 1 号艇が予測ランキング 1 位
    b1_is_rank1 = np.take_along_axis(b2, po[:, :1], axis=1).flatten() == 1

    race_ids = test["race_id"].values.reshape(n, FIELD_SIZE)[:, 0].astype(int)

    return p1, hit_boat1, gap12, top3_conc, gap23, b1_is_rank1, race_ids


def _print_reliability(
    label: str, p1: np.ndarray, hit: np.ndarray, buckets: list[tuple[float, float]],
) -> None:
    print(f"\n--- {label} ({len(p1)} races) ---")
    print(
        f"  {'予測確率区分':<14}  {'件数':>6}  {'1 号艇 1 着数':>12}  "
        f"{'実勝率':>8}  {'予測平均':>8}  {'乖離':>8}"
    )
    for lo, hi in buckets:
        mask = (p1 >= lo) & (p1 < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        hit_rate = float(hit[mask].mean()) * 100
        pred_mean = float(p1[mask].mean()) * 100
        diff = hit_rate - pred_mean
        bucket_label = f"{lo * 100:>3.0f}〜{hi * 100:>3.0f}%"
        print(
            f"  {bucket_label:<14}  {n:>6}  {int(hit[mask].sum()):>12}  "
            f"{hit_rate:>7.1f}%  {pred_mean:>7.1f}%  {diff:>+7.1f}pt"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--model-dir", default="models/p2_v3")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    model = load_model(Path(args.model_dir) / "ranking")
    meta = load_model_meta(Path(args.model_dir) / "ranking")

    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    p1, hit, gap12, conc, gap23, b1_rank1, race_ids = _compute_boat1_prob_and_hit(
        model, meta, df, args.from_date, args.to_date,
    )

    buckets = [
        (0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01),
    ]

    st = meta.get("strategy", {})
    conc_th = st.get("top3_conc_threshold", 0.6)
    gap12_th = st.get("gap12_min_threshold", 0.04)
    gap23_th = st.get("gap23_threshold", 0.13)

    print(f"=== 1 号艇 予測確率 vs 実勝率 ===")
    print(f"モデル: {args.model_dir}")
    print(f"期間: {args.from_date} ~ {args.to_date}")
    print(f"フィルタ値: gap12>={gap12_th} conc>={conc_th} gap23>={gap23_th}")

    # 全レース
    _print_reliability("全レース", p1, hit, buckets)

    # 1 号艇が予測 1 位のレースのみ
    mask_rank1 = b1_rank1
    _print_reliability(
        "1 号艇が予測 1 位", p1[mask_rank1], hit[mask_rank1], buckets,
    )

    # P2 フィルタ通過 (b1_rank1 + gap12 + conc + gap23)
    mask_p2 = (
        b1_rank1 & (gap12 >= gap12_th) & (conc >= conc_th) & (gap23 >= gap23_th)
    )
    _print_reliability(
        "P2 フィルタ通過", p1[mask_p2], hit[mask_p2], buckets,
    )

    # まとめ
    print("\n=== まとめ ===")
    for label, mask in [
        ("全レース", np.ones_like(p1, dtype=bool)),
        ("1 号艇が予測 1 位", mask_rank1),
        ("P2 フィルタ通過", mask_p2),
    ]:
        n = int(mask.sum())
        if n == 0:
            continue
        hr = float(hit[mask].mean()) * 100
        pm = float(p1[mask].mean()) * 100
        print(
            f"  {label:<18} 件数 {n:>5}  実勝率 {hr:>5.1f}%  予測平均 {pm:>5.1f}%  "
            f"乖離 {hr - pm:+5.1f}pt"
        )

    # 予測確率区分 × 実 ROI (P2 購入レースのみ)
    print("\n=== 予測確率区分 × 実 ROI (P2 購入) ===")
    conn = get_connection(args.db_path)
    odds_rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds "
        "WHERE bet_type='3連単'"
    ).fetchall()
    conn.close()
    odds = {(int(r[0]), r[1]): float(r[2]) for r in odds_rows}

    purchases, _ = evaluate_period(
        model, meta, df, odds, args.from_date, args.to_date,
    )
    rid_to_p1 = dict(zip(race_ids.tolist(), p1.tolist()))

    class Bucket:
        def __init__(self):
            self.races = 0
            self.wins = 0
            self.cost = 0.0
            self.payout = 0.0

    bucket_stats: dict[str, Bucket] = {}
    for (lo, hi) in buckets:
        bucket_stats[f"{int(lo*100):>2}〜{int(hi*100):>3}%"] = Bucket()

    for p in purchases:
        p1_val = rid_to_p1.get(p.race_id)
        if p1_val is None:
            continue
        for (lo, hi) in buckets:
            if lo <= p1_val < hi:
                b = bucket_stats[f"{int(lo*100):>2}〜{int(hi*100):>3}%"]
                b.races += 1
                b.wins += 1 if p.won else 0
                b.cost += p.cost
                b.payout += p.payout
                break

    print(
        f"  {'予測区分':<10}  {'件数':>6}  {'的中':>5}  {'的中率':>7}  "
        f"{'購入':>9}  {'払戻':>9}  {'ROI':>7}  {'P/L':>9}"
    )
    total_cost = 0.0
    total_payout = 0.0
    for label, b in bucket_stats.items():
        if b.races == 0:
            continue
        hit_rate = 100 * b.wins / b.races
        roi = 100 * b.payout / b.cost if b.cost else 0
        pl = b.payout - b.cost
        total_cost += b.cost
        total_payout += b.payout
        print(
            f"  {label:<10}  {b.races:>6}  {b.wins:>5}  {hit_rate:>6.1f}%  "
            f"{int(b.cost):>9,}  {int(b.payout):>9,}  {roi:>6.1f}%  {int(pl):>+9,}"
        )
    total_roi = 100 * total_payout / total_cost if total_cost else 0
    print(
        f"  {'合計':<10}  {'':>6}  {'':>5}  {'':>7}  "
        f"{int(total_cost):>9,}  {int(total_payout):>9,}  {total_roi:>6.1f}%  "
        f"{int(total_payout - total_cost):>+9,}"
    )


if __name__ == "__main__":
    main()
