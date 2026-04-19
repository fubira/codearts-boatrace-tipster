"""1 号艇 racer のコース 1 勝率 × 2 号艇強度別の勝率と ROI を分析。

マスター観察: 若手は 1 号艇で勝ったり負けたりするが、ベテランは勝って
当たり前。1 号艇 racer の絶対コース 1 勝率が低い race では「飛ぶ」可能性が
高い。強い 2 号艇の番組と組み合わせると edge が見える可能性あり。

Usage:
    uv run python -m scripts.analyze_boat1_skill \\
        --from 2026-01-01 --to 2026-04-18 --model-dir models/p2_v3
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import defaultdict
from pathlib import Path

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from scripts.analyze_model import evaluate_period


def _load_entries(db_path: str, race_ids: list[int]) -> dict[int, dict[int, dict]]:
    if not race_ids:
        return {}
    conn = get_connection(db_path)
    placeholders = ",".join(["?"] * len(race_ids))
    rows = conn.execute(
        f"SELECT race_id, boat_number, racer_class, national_win_rate, "
        f"local_win_rate FROM db.race_entries WHERE race_id IN ({placeholders})",
        list(race_ids),
    ).fetchall()
    conn.close()
    out: dict[int, dict[int, dict]] = defaultdict(dict)
    for race_id, boat, cls, nwr, lwr in rows:
        out[int(race_id)][int(boat)] = {
            "class": cls or "?",
            "national_win_rate": float(nwr) if nwr is not None else 0.0,
            "local_win_rate": float(lwr) if lwr is not None else 0.0,
        }
    return out


def _load_boat1_course_winrate(
    db_path: str, race_ids: list[int], df,
) -> dict[int, float]:
    """1 号艇 racer の「過去 1 コース勝率」を race_id 単位で返す。

    features dataframe の racer_course_win_rate カラムを使う (race_id と
    boat_number=1 の行)。
    """
    if "racer_course_win_rate" not in df.columns:
        return {}
    rid_set = set(race_ids)
    sub = df[(df["race_id"].isin(rid_set)) & (df["boat_number"] == 1)]
    return dict(zip(sub["race_id"].astype(int), sub["racer_course_win_rate"].astype(float)))


def _bucket_boat1_rate(r: float) -> str:
    if r < 0.15:
        return "0〜15%"
    if r < 0.25:
        return "15〜25%"
    if r < 0.35:
        return "25〜35%"
    if r < 0.45:
        return "35〜45%"
    if r < 0.55:
        return "45〜55%"
    return "55%+"


def _bucket_wr_diff(d: float) -> str:
    if d >= 2.0:
        return "b1 >> b2 (+2.0 以上)"
    if d >= 1.0:
        return "b1 > b2 (+1.0〜2.0)"
    if d >= 0.0:
        return "b1 ≈ b2 (+0〜1.0)"
    if d >= -1.0:
        return "b1 < b2 (-1.0〜0)"
    return "b1 << b2 (-1.0 以下) 強い 2 号艇"


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
    if not purchases:
        print("No buy races.")
        return

    race_ids = [p.race_id for p in purchases]
    entries = _load_entries(args.db_path, race_ids)
    boat1_wr_map = _load_boat1_course_winrate(args.db_path, race_ids, df)

    # 軸 B: 1 号艇 course 勝率 → bucket 別集計
    print(f"\n=== 1 号艇 course 勝率別の buy race 集計 ===")
    print(f"モデル: {args.model_dir} / 期間: {args.from_date} ~ {args.to_date}")
    print(f"Buy races: {len(purchases)}")
    print(f"\n{'course1 勝率':<10}  {'件数':>5}  {'1 着':>4}  {'hit%':>6}  "
          f"{'購入':>8}  {'払戻':>8}  {'ROI':>7}  {'P/L':>8}")
    b1_stats: dict[str, dict] = defaultdict(
        lambda: {"races": 0, "wins": 0, "cost": 0, "payout": 0}
    )
    for p in purchases:
        wr = boat1_wr_map.get(p.race_id)
        if wr is None:
            continue
        bucket = _bucket_boat1_rate(wr)
        s = b1_stats[bucket]
        s["races"] += 1
        s["wins"] += 1 if p.won else 0
        s["cost"] += p.cost
        s["payout"] += p.payout
    for bucket in ["0〜15%", "15〜25%", "25〜35%", "35〜45%", "45〜55%", "55%+"]:
        s = b1_stats.get(bucket, {"races": 0, "wins": 0, "cost": 0, "payout": 0})
        if s["races"] == 0:
            continue
        hit = 100 * s["wins"] / s["races"]
        roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
        pl = s["payout"] - s["cost"]
        print(f"{bucket:<10}  {s['races']:>5}  {s['wins']:>4}  {hit:>5.1f}%  "
              f"{int(s['cost']):>8,}  {int(s['payout']):>8,}  {roi:>6.1f}%  "
              f"{int(pl):>+8,}")

    # 軸 A × 軸 B: 2 号艇強度 × 1 号艇 course 勝率のクロス集計
    print(f"\n=== クロス集計: 2 号艇強度 × 1 号艇 course 勝率 → hit% / ROI ===")
    cross_stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"races": 0, "wins": 0, "cost": 0, "payout": 0}
    )
    for p in purchases:
        wr = boat1_wr_map.get(p.race_id)
        if wr is None:
            continue
        e = entries.get(p.race_id, {})
        wr1 = e.get(1, {}).get("national_win_rate", 0.0)
        wr2 = e.get(2, {}).get("national_win_rate", 0.0)
        key = (_bucket_wr_diff(wr1 - wr2), _bucket_boat1_rate(wr))
        s = cross_stats[key]
        s["races"] += 1
        s["wins"] += 1 if p.won else 0
        s["cost"] += p.cost
        s["payout"] += p.payout

    print(f"\n{'2 号艇強度':<30}  {'c1 勝率':<10}  {'件数':>5}  "
          f"{'hit%':>6}  {'ROI':>7}  {'P/L':>8}")
    wr_order = [
        "b1 >> b2 (+2.0 以上)",
        "b1 > b2 (+1.0〜2.0)",
        "b1 ≈ b2 (+0〜1.0)",
        "b1 < b2 (-1.0〜0)",
        "b1 << b2 (-1.0 以下) 強い 2 号艇",
    ]
    b1_order = ["0〜15%", "15〜25%", "25〜35%", "35〜45%", "45〜55%", "55%+"]
    for wr_b in wr_order:
        for b1_b in b1_order:
            s = cross_stats.get((wr_b, b1_b))
            if s is None or s["races"] < 5:
                continue
            hit = 100 * s["wins"] / s["races"]
            roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
            pl = s["payout"] - s["cost"]
            print(f"{wr_b:<30}  {b1_b:<10}  {s['races']:>5}  "
                  f"{hit:>5.1f}%  {roi:>6.1f}%  {int(pl):>+8,}")


if __name__ == "__main__":
    main()
