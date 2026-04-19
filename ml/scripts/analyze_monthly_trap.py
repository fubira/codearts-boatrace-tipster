"""月別に「1 枠に下手な選手を入れるレース」の比率と結果分布を集計。

仮説: 2 月 (hard regime) で 1 号艇の skill が低い番組が多く配置される
ことで、model が外しやすくなっているのではないか。

指標:
  - 1 号艇 national_win_rate の平均 / 低勝率 (<5.0) racer の比率
  - 1 号艇 1 着率
  - 2-1-X (2 号艇 1 着 & 1 号艇 2 着) 比率
  - 1 号艇 racer_class 分布

Usage:
    uv run python -m scripts.analyze_monthly_trap \\
        --from 2026-01-01 --to 2026-04-18
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    rows = conn.execute(
        """
        SELECT r.id, r.race_date, re.boat_number, re.finish_position,
               re.racer_class, re.national_win_rate
        FROM db.races r JOIN db.race_entries re ON r.id = re.race_id
        WHERE r.race_date BETWEEN ? AND ?
        ORDER BY r.id, re.boat_number
        """,
        (args.from_date, args.to_date),
    ).fetchall()
    conn.close()

    # Group per race
    races: dict[int, dict] = defaultdict(dict)
    for rid, date, boat, fin, cls, nwr in rows:
        r = races[int(rid)]
        r["date"] = str(date)
        r.setdefault("boats", {})[int(boat)] = {
            "fin": int(fin) if fin is not None else None,
            "class": cls or "?",
            "nwr": float(nwr) if nwr is not None else 0.0,
        }

    # Monthly aggregation
    monthly: dict[str, dict] = defaultdict(
        lambda: {
            "races": 0,
            "boat1_nwr_sum": 0.0,
            "boat1_low_nwr_count": 0,  # boat1 national_win_rate < 5.0
            "boat1_b2_lower_nwr_count": 0,  # boat1 nwr < boat2 nwr
            "boat1_1st": 0,
            "boat2_1st": 0,
            "b2_b1_pattern": 0,  # 2-1-X
            "class_dist": defaultdict(int),  # boat1 class
        }
    )

    for rid, r in races.items():
        boats = r.get("boats", {})
        if 1 not in boats or 2 not in boats:
            continue
        b1 = boats[1]
        b2 = boats[2]
        month = r["date"][:7]  # "2026-01"
        m = monthly[month]
        m["races"] += 1
        m["boat1_nwr_sum"] += b1["nwr"]
        if b1["nwr"] < 5.0 and b1["nwr"] > 0:
            m["boat1_low_nwr_count"] += 1
        if b1["nwr"] < b2["nwr"]:
            m["boat1_b2_lower_nwr_count"] += 1
        if b1["fin"] == 1:
            m["boat1_1st"] += 1
        if b2["fin"] == 1:
            m["boat2_1st"] += 1
            if b1["fin"] == 2:
                m["b2_b1_pattern"] += 1
        m["class_dist"][b1["class"]] += 1

    print(f"=== 月別 trap 番組比率 (期間: {args.from_date} ~ {args.to_date}) ===")
    print(
        f"\n{'月':<8}  {'races':>5}  {'b1 nwr 平均':>11}  "
        f"{'b1 低 nwr%':>11}  {'b1<b2 nwr%':>11}  "
        f"{'b1 1 着%':>9}  {'b2 1 着%':>9}  {'2-1-X%':>8}"
    )
    for month in sorted(monthly.keys()):
        m = monthly[month]
        n = m["races"]
        if n == 0:
            continue
        avg_nwr = m["boat1_nwr_sum"] / n
        low_pct = 100 * m["boat1_low_nwr_count"] / n
        b1lt = 100 * m["boat1_b2_lower_nwr_count"] / n
        b1_1st = 100 * m["boat1_1st"] / n
        b2_1st = 100 * m["boat2_1st"] / n
        trap = 100 * m["b2_b1_pattern"] / n
        print(
            f"{month:<8}  {n:>5}  {avg_nwr:>10.2f}  "
            f"{low_pct:>10.1f}%  {b1lt:>10.1f}%  "
            f"{b1_1st:>8.1f}%  {b2_1st:>8.1f}%  {trap:>7.1f}%"
        )

    print(f"\n--- 1 号艇 class 月別分布 ---")
    classes = ["A1", "A2", "B1", "B2"]
    print(f"{'月':<8}  " + "  ".join(f"{c:>7}" for c in classes))
    for month in sorted(monthly.keys()):
        m = monthly[month]
        n = m["races"]
        if n == 0:
            continue
        pcts = [f"{100*m['class_dist'].get(c,0)/n:>6.1f}%" for c in classes]
        print(f"{month:<8}  " + "  ".join(pcts))


if __name__ == "__main__":
    main()
