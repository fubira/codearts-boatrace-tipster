"""P2 buy race の月別 × {会場 / グレード / 天候 / 水温帯} の hit% と P/L。

2 月 hard regime の構造要因を探る。月別で会場特性 / race_grade /
天候要素の差を見る。

Usage:
    uv run python -m scripts.analyze_february_regime \\
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


def _load_race_meta(db_path: str, race_ids: list[int]) -> dict[int, dict]:
    if not race_ids:
        return {}
    conn = get_connection(db_path)
    placeholders = ",".join(["?"] * len(race_ids))
    rows = conn.execute(
        f"SELECT r.id, r.race_date, r.stadium_id, s.name, "
        f"       r.race_grade, r.race_title, r.weather, r.wind_speed, "
        f"       r.water_temperature "
        f"FROM db.races r JOIN db.stadiums s ON r.stadium_id = s.id "
        f"WHERE r.id IN ({placeholders})",
        list(race_ids),
    ).fetchall()
    conn.close()
    out: dict[int, dict] = {}
    for (rid, date, sid, sname, grade, title, weather, wind, wtemp) in rows:
        out[int(rid)] = {
            "date": str(date),
            "stadium": sname,
            "grade": grade or "?",
            "weather": weather or "?",
            "wind": int(wind) if wind is not None else 0,
            "water_temperature": float(wtemp) if wtemp is not None else 0.0,
        }
    return out


def _print_table(
    title: str, grouped: dict[str, dict], sort_key=lambda kv: -kv[1]["races"],
) -> None:
    print(f"\n--- {title} ---")
    print(
        f"  {'group':<18}  {'races':>5}  {'hit':>4}  {'hit%':>6}  "
        f"{'ROI':>7}  {'P/L':>8}"
    )
    for label, s in sorted(grouped.items(), key=sort_key):
        if s["races"] == 0:
            continue
        hit = 100 * s["wins"] / s["races"]
        roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
        pl = s["payout"] - s["cost"]
        print(
            f"  {label:<18}  {s['races']:>5}  {s['wins']:>4}  {hit:>5.1f}%  "
            f"{roi:>6.1f}%  {int(pl):>+8,}"
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
    meta_map = _load_race_meta(args.db_path, race_ids)

    def _blank_stat():
        return {"races": 0, "wins": 0, "cost": 0.0, "payout": 0.0}

    # 月別 × 各軸の集計
    monthly_stats: dict[str, dict] = defaultdict(_blank_stat)
    month_stadium: dict[tuple[str, str], dict] = defaultdict(_blank_stat)
    month_grade: dict[tuple[str, str], dict] = defaultdict(_blank_stat)
    month_weather: dict[tuple[str, str], dict] = defaultdict(_blank_stat)
    month_wtemp: dict[tuple[str, str], dict] = defaultdict(_blank_stat)

    def _wtemp_bucket(t: float) -> str:
        if t <= 0:
            return "?"
        if t < 10:
            return "<10℃"
        if t < 15:
            return "10〜15℃"
        if t < 20:
            return "15〜20℃"
        if t < 25:
            return "20〜25℃"
        return "25℃+"

    for p in purchases:
        m = meta_map.get(p.race_id)
        if m is None:
            continue
        month = m["date"][:7]
        for (target, key) in [
            (monthly_stats, month),
            (month_stadium, (month, m["stadium"])),
            (month_grade, (month, m["grade"])),
            (month_weather, (month, m["weather"])),
            (month_wtemp, (month, _wtemp_bucket(m["water_temperature"]))),
        ]:
            s = target[key]
            s["races"] += 1
            s["wins"] += 1 if p.won else 0
            s["cost"] += p.cost
            s["payout"] += p.payout

    print(f"=== P2 buy race: 月別 + 軸別 集計 (期間 {args.from_date} ~ {args.to_date}) ===")
    print(f"Buy races: {len(purchases)}")

    _print_table("月別", monthly_stats, sort_key=lambda kv: kv[0])

    print("\n--- 月別 × 会場 (各月 top 5 races 数) ---")
    by_month_st: dict[str, list] = defaultdict(list)
    for (month, st), s in month_stadium.items():
        by_month_st[month].append((st, s))
    for month in sorted(by_month_st.keys()):
        items = sorted(by_month_st[month], key=lambda kv: -kv[1]["races"])[:5]
        print(f"\n  [{month}]")
        print(
            f"    {'stadium':<10}  {'races':>5}  {'hit%':>6}  {'ROI':>7}  {'P/L':>8}"
        )
        for st, s in items:
            if s["races"] == 0:
                continue
            hit = 100 * s["wins"] / s["races"]
            roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
            pl = s["payout"] - s["cost"]
            print(
                f"    {st:<10}  {s['races']:>5}  {hit:>5.1f}%  {roi:>6.1f}%  {int(pl):>+8,}"
            )

    print("\n--- 月別 × race_grade ---")
    for month in sorted(set(k[0] for k in month_grade.keys())):
        print(f"\n  [{month}]")
        print(f"    {'grade':<10}  {'races':>5}  {'hit%':>6}  {'ROI':>7}  {'P/L':>8}")
        items = [(g, s) for (m, g), s in month_grade.items() if m == month]
        for g, s in sorted(items, key=lambda kv: -kv[1]["races"]):
            if s["races"] == 0:
                continue
            hit = 100 * s["wins"] / s["races"]
            roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
            pl = s["payout"] - s["cost"]
            print(
                f"    {g:<10}  {s['races']:>5}  {hit:>5.1f}%  {roi:>6.1f}%  {int(pl):>+8,}"
            )

    print("\n--- 月別 × 水温帯 ---")
    for month in sorted(set(k[0] for k in month_wtemp.keys())):
        print(f"\n  [{month}]")
        print(f"    {'water_temp':<10}  {'races':>5}  {'hit%':>6}  {'ROI':>7}  {'P/L':>8}")
        items = [(g, s) for (m, g), s in month_wtemp.items() if m == month]
        for g, s in sorted(items, key=lambda kv: str(kv[0])):
            if s["races"] == 0:
                continue
            hit = 100 * s["wins"] / s["races"]
            roi = 100 * s["payout"] / s["cost"] if s["cost"] else 0
            pl = s["payout"] - s["cost"]
            print(
                f"    {g:<10}  {s['races']:>5}  {hit:>5.1f}%  {roi:>6.1f}%  {int(pl):>+8,}"
            )


if __name__ == "__main__":
    main()
