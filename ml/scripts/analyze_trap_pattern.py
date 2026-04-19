"""Analyze "trap program" exposure for P2 buy races.

Board programming sometimes places a weak racer in boat 1 and skilled
racers in boats 2-3 to bait boat-1 bias. This script measures how often
the model's P2 buys fall into that trap by looking at:

  1. First-place boat distribution among buy races (vs all races baseline)
  2. When boat 2 wins (2-X-Y), which boat takes 2nd/3rd
  3. Class combinations of boat 1 and boat 2 on buy races + the boat-1
     1st-place rate per (class1, class2) pair

Usage:
    uv run python -m scripts.analyze_trap_pattern \\
        --from 2026-01-01 --to 2026-04-18 --model-dir models/p2_v3
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter, defaultdict
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

    buy_races = [(p.race_id, p.hit_combo) for p in purchases]
    if not buy_races:
        print("No buy races in period.")
        return

    race_ids = [r[0] for r in buy_races]
    entries = _load_entries(args.db_path, race_ids)

    # 1. First-place boat distribution (buy races vs baseline)
    first_boat_counts: Counter[int] = Counter()
    for _, hit in buy_races:
        first_boat_counts[int(hit.split("-")[0])] += 1

    conn = get_connection(args.db_path)
    baseline_rows = conn.execute(
        "SELECT r.id, re.boat_number FROM db.races r "
        "JOIN db.race_entries re ON r.id = re.race_id "
        "WHERE r.race_date BETWEEN ? AND ? AND re.finish_position = 1",
        (args.from_date, args.to_date),
    ).fetchall()
    conn.close()
    baseline_counts: Counter[int] = Counter(int(r[1]) for r in baseline_rows)
    total_baseline = len(baseline_rows)

    print(f"\n=== Trap program exposure analysis ===")
    print(f"Model: {args.model_dir}")
    print(f"Period: {args.from_date} ~ {args.to_date}")
    print(f"Buy races: {len(buy_races)}, Baseline races: {total_baseline}")

    print(f"\n--- 1 着 boat 分布: buy race vs baseline ---")
    print(f"  {'boat':>4}  {'baseline %':>10}  {'buy %':>8}  {'diff pt':>8}")
    for b in range(1, 7):
        base_pct = (
            100 * baseline_counts.get(b, 0) / total_baseline if total_baseline else 0
        )
        buy_pct = 100 * first_boat_counts.get(b, 0) / len(buy_races)
        diff = buy_pct - base_pct
        print(f"  {b:>4}  {base_pct:>9.1f}%  {buy_pct:>7.1f}%  {diff:>+7.1f}pt")

    # 2. When boat 2 wins on buy races: which boat is 2nd
    second_when_2wins: Counter[int] = Counter()
    third_when_2wins: Counter[int] = Counter()
    b2_wins = 0
    for _, hit in buy_races:
        parts = [int(x) for x in hit.split("-")]
        if parts[0] == 2:
            b2_wins += 1
            second_when_2wins[parts[1]] += 1
            third_when_2wins[parts[2]] += 1

    if b2_wins > 0:
        print(f"\n--- 2 号艇 1 着 race ({b2_wins}/{len(buy_races)}、{100*b2_wins/len(buy_races):.1f}%) の 2 着 boat 分布 ---")
        print(f"  {'2 着':>4}  {'count':>6}  {'% of b2 wins':>14}")
        for b in range(1, 7):
            c = second_when_2wins.get(b, 0)
            pct = 100 * c / b2_wins if b2_wins else 0
            tag = " ← trap pattern" if b == 1 else ""
            print(f"  {b:>4}  {c:>6}  {pct:>13.1f}%{tag}")

    # 3. class(1) × class(2) breakdown + boat-1 hit rate
    class_grid: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "boat1_first": 0}
    )
    for rid, hit in buy_races:
        e = entries.get(rid, {})
        c1 = e.get(1, {}).get("class", "?")
        c2 = e.get(2, {}).get("class", "?")
        class_grid[(c1, c2)]["count"] += 1
        if int(hit.split("-")[0]) == 1:
            class_grid[(c1, c2)]["boat1_first"] += 1

    print(f"\n--- 1 号艇 class × 2 号艇 class: 1 号艇 1 着率 (buy race) ---")
    print(f"  {'class1':>6}  {'class2':>6}  {'count':>6}  {'b1 1st':>7}  {'hit%':>6}")
    for (c1, c2), stats in sorted(
        class_grid.items(), key=lambda kv: -kv[1]["count"]
    ):
        cnt = stats["count"]
        hits = stats["boat1_first"]
        pct = 100 * hits / cnt if cnt else 0.0
        print(f"  {c1:>6}  {c2:>6}  {cnt:>6}  {hits:>7}  {pct:>5.1f}%")

    # 4. national_win_rate diff between boat 1 and boat 2
    # 「1 号艇が 2 号艇より弱い」番組で 1 号艇 hit 率はどうなるか
    wr_diff_buckets: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "boat1_first": 0}
    )
    for rid, hit in buy_races:
        e = entries.get(rid, {})
        wr1 = e.get(1, {}).get("national_win_rate", 0.0)
        wr2 = e.get(2, {}).get("national_win_rate", 0.0)
        diff = wr1 - wr2
        if diff >= 2.0:
            bucket = "b1 >> b2 (+2.0 以上)"
        elif diff >= 1.0:
            bucket = "b1 > b2 (+1.0〜2.0)"
        elif diff >= 0.0:
            bucket = "b1 ≈ b2 (+0〜1.0)"
        elif diff >= -1.0:
            bucket = "b1 < b2 (-1.0〜0)"
        else:
            bucket = "b1 << b2 (-1.0 以下) ← trap?"
        wr_diff_buckets[bucket]["count"] += 1
        if int(hit.split("-")[0]) == 1:
            wr_diff_buckets[bucket]["boat1_first"] += 1

    print(f"\n--- national_win_rate 差 (b1-b2) 別の 1 号艇 1 着率 ---")
    order = [
        "b1 >> b2 (+2.0 以上)",
        "b1 > b2 (+1.0〜2.0)",
        "b1 ≈ b2 (+0〜1.0)",
        "b1 < b2 (-1.0〜0)",
        "b1 << b2 (-1.0 以下) ← trap?",
    ]
    print(f"  {'bucket':<36}  {'count':>6}  {'b1 1st':>7}  {'hit%':>6}")
    for b in order:
        stats = wr_diff_buckets.get(b, {"count": 0, "boat1_first": 0})
        cnt = stats["count"]
        hits = stats["boat1_first"]
        pct = 100 * hits / cnt if cnt else 0.0
        print(f"  {b:<36}  {cnt:>6}  {hits:>7}  {pct:>5.1f}%")


if __name__ == "__main__":
    main()
