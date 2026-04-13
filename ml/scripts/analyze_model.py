"""Analyze a P2 model's OOS performance across stadiums and time.

Unified diagnostic tool that replaces earlier one-off check scripts.
Use cases:
- Stadium-level breakdown to detect underperforming venues
- Time-windowed trends (quarter / month) to detect model degradation
- Feature importance to understand what the model relies on
- Before/after comparison after adding features or retraining

Usage:
    uv run python scripts/analyze_model.py --from 2026-01-01 --to 2026-04-13
    uv run python scripts/analyze_model.py --from 2025-07-01 --to 2026-04-13 --split-by quarter
    uv run python scripts/analyze_model.py --from 2026-01-01 --to 2026-04-13 --show-importance
    uv run python scripts/analyze_model.py --model-dir models/p2_v1 --from 2026-01-01 --to 2026-04-13
"""

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import fill_nan_with_means, load_model, load_model_meta
from boatrace_tipster_ml.registry import get_active_model_dir

FIELD_SIZE = 6


@dataclass
class Purchase:
    race_id: int
    race_date: str
    stadium_id: int
    top3_conc: float
    gap23: float
    tickets: list = field(default_factory=list)  # [(combo, odds, ev)]
    hit_combo: str = ""
    won: bool = False
    b1_won: bool = False
    cost: float = 0.0
    payout: float = 0.0


def _trifecta_prob(p, i1, i2, i3):
    a, b, c = p[i1], p[i2], p[i3]
    if a >= 1 or (a + b) >= 1:
        return 0.0
    return a * (b / (1 - a)) * (c / (1 - a - b))


def evaluate_period(
    model, meta, df, odds, from_date: str, to_date: str
) -> tuple[list[Purchase], dict[int, int]]:
    """Run model predictions on a period and apply P2 strategy filter.

    Returns:
      purchases: list of Purchase records (only those passing all filters)
      total_by_stadium: total races per stadium_id in the period (for context)
    """
    features = meta["feature_columns"]
    st = meta.get("strategy", {})
    conc_th = st.get("top3_conc_threshold", 0.0)
    gap23_th = st.get("gap23_threshold", 0.0)
    gap12_th = st.get("gap12_min_threshold", 0.0)
    ev_th = st.get("ev_threshold", 0.0)
    excluded = set(st.get("excluded_stadiums") or [])

    test = df[(df["race_date"] >= from_date) & (df["race_date"] < to_date)].copy()
    X = test[features].copy()
    fill_nan_with_means(X, meta)
    scores = model.predict(X)

    n = len(X) // FIELD_SIZE
    s2 = scores.reshape(n, FIELD_SIZE)
    b2 = test["boat_number"].values.reshape(n, FIELD_SIZE).astype(int)
    ri = test["race_id"].values.reshape(n, FIELD_SIZE)[:, 0].astype(int)
    sid_arr = test["stadium_id"].values.reshape(n, FIELD_SIZE)[:, 0].astype(int)
    dates = test["race_date"].values.reshape(n, FIELD_SIZE)[:, 0]
    y2 = test["finish_position"].values.reshape(n, FIELD_SIZE)

    po = np.argsort(-s2, axis=1)
    tb = np.take_along_axis(b2, po, axis=1)
    ex = np.exp(s2 - s2.max(axis=1, keepdims=True))
    mp = ex / ex.sum(axis=1, keepdims=True)
    ao = np.argsort(y2, axis=1)
    ab = np.take_along_axis(b2, ao, axis=1).astype(int)

    purchases: list[Purchase] = []
    total_by_stadium: dict[int, int] = defaultdict(int)

    for i in range(n):
        sid = int(sid_arr[i])
        total_by_stadium[sid] += 1

        if sid in excluded:
            continue
        if tb[i, 0] != 1:
            continue
        p1 = float(mp[i, po[i, 0]])
        p2 = float(mp[i, po[i, 1]])
        p3 = float(mp[i, po[i, 2]])
        gap12 = p1 - p2
        if gap12 < gap12_th:
            continue
        top3_conc = (p2 + p3) / (1 - p1 + 1e-10)
        gap23 = p2 - p3
        if top3_conc < conc_th or gap23 < gap23_th:
            continue

        rid = int(ri[i])
        r2, r3 = int(tb[i, 1]), int(tb[i, 2])
        a1, a2, a3 = int(ab[i, 0]), int(ab[i, 1]), int(ab[i, 2])
        hit = f"{a1}-{a2}-{a3}"

        tks: list[tuple[str, float, float]] = []
        for combo, ia, ib, ic in [
            (f"1-{r2}-{r3}", po[i, 0], po[i, 1], po[i, 2]),
            (f"1-{r3}-{r2}", po[i, 0], po[i, 2], po[i, 1]),
        ]:
            o = odds.get((rid, combo), 0)
            if o <= 0:
                continue
            m2 = _trifecta_prob(mp[i], ia, ib, ic)
            ev = m2 / (1 / o) * 0.75 - 1
            if ev >= ev_th:
                tks.append((combo, o, ev))
        if not tks:
            continue

        cost = len(tks) * 100
        payout = 0.0
        won = False
        for combo, o, _ in tks:
            if combo == hit:
                won = True
                payout = o * 100
                break

        purchases.append(
            Purchase(
                race_id=rid, race_date=str(dates[i]), stadium_id=sid,
                top3_conc=top3_conc, gap23=gap23, tickets=tks,
                hit_combo=hit, won=won, b1_won=(a1 == 1),
                cost=cost, payout=payout,
            )
        )

    return purchases, dict(total_by_stadium)


def aggregate(purchases: list[Purchase], key=None) -> dict:
    """Group purchases and compute summary metrics for each group."""
    groups: dict = defaultdict(list)
    for p in purchases:
        k = key(p) if key else "all"
        groups[k].append(p)

    out: dict = {}
    for k, ps in groups.items():
        n = len(ps)
        w = sum(1 for p in ps if p.won)
        b1 = sum(1 for p in ps if p.b1_won)
        cost = sum(p.cost for p in ps)
        pay = sum(p.payout for p in ps)
        hit_odds = [p.payout / 100 for p in ps if p.won]
        out[k] = {
            "n": n, "w": w, "b1_wins": b1,
            "cost": cost, "payout": pay,
            "hit_pct": 100 * w / n if n else 0.0,
            "b1_pct": 100 * b1 / n if n else 0.0,
            "roi_pct": 100 * pay / cost if cost else 0.0,
            "pl": pay - cost,
            "median_payout": float(np.median(hit_odds)) if hit_odds else 0.0,
            "mean_payout": float(np.mean(hit_odds)) if hit_odds else 0.0,
        }
    return out


def _load_stadium_names(db_path: str) -> dict[int, str]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, name FROM db.stadiums ORDER BY id"
    ).fetchall()
    conn.close()
    return {int(r[0]): r[1] for r in rows}


def print_row(label: str, m: dict) -> None:
    print(
        f"  {label:<20} {m['n']:>5} {m['w']:>4} {m['hit_pct']:>5.1f}% "
        f"{m['roi_pct']:>5.0f}% {m['b1_pct']:>5.0f}% {m['pl']:>+9,.0f}"
    )


def print_importance(model, features: list[str], top_k: int = 25) -> None:
    print("\n=== Feature importance ===")
    imps = model.feature_importances_
    total = sum(imps)
    rank = sorted(zip(features, imps), key=lambda x: -x[1])
    for i, (f, imp) in enumerate(rank[:top_k]):
        marker = " <- bc_*" if f.startswith("bc_") else ""
        print(f"  {i+1:2d}. {f:<32} {imp:>6d} ({100*imp/total:>5.1f}%){marker}")
    bc_sum = sum(imp for f, imp in zip(features, imps) if f.startswith("bc_"))
    print(f"\n  bc_* 合計: {bc_sum} ({100*bc_sum/total:.1f}% of total)")


def period_key(p: Purchase, split_by: str) -> str:
    if split_by == "quarter":
        y, m = int(p.race_date[:4]), int(p.race_date[5:7])
        q = (m - 1) // 3 + 1
        return f"{y}Q{q}"
    # month
    return p.race_date[:7]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose a P2 model's OOS performance"
    )
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument(
        "--model-dir", default=None,
        help="Default: active model from active.json",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--split-by", choices=["none", "quarter", "month"], default="none",
        help="Split time into periods for trend analysis",
    )
    parser.add_argument(
        "--stadium", default=None,
        help="Comma-separated stadium IDs to focus on (filters output)",
    )
    parser.add_argument("--show-importance", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir or get_active_model_dir()
    print(f"Model: {model_dir}/ranking", file=sys.stderr)
    model = load_model(f"{model_dir}/ranking")
    meta = load_model_meta(f"{model_dir}/ranking")
    stadium_names = _load_stadium_names(args.db_path)

    print("Loading features...", file=sys.stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path)

    print("Loading odds...", file=sys.stderr)
    conn = get_connection(args.db_path)
    rows = conn.execute(
        "SELECT race_id, combination, odds FROM db.race_odds WHERE bet_type='3連単'"
    ).fetchall()
    conn.close()
    odds = {(int(r[0]), r[1]): float(r[2]) for r in rows}

    purchases, total_by_stadium = evaluate_period(
        model, meta, df, odds, args.from_date, args.to_date
    )

    # --stadium filter: focus on specific stadium IDs
    focus_stadiums: set[int] | None = None
    if args.stadium:
        focus_stadiums = {int(s) for s in args.stadium.split(",")}
        purchases = [p for p in purchases if p.stadium_id in focus_stadiums]
        total_by_stadium = {
            sid: n for sid, n in total_by_stadium.items() if sid in focus_stadiums
        }

    if args.json:
        by_stadium = aggregate(purchases, key=lambda p: p.stadium_id)
        result = {
            "from": args.from_date, "to": args.to_date,
            "model": model_dir,
            "n_purchases": len(purchases),
            "overall": aggregate(purchases).get("all", {}),
            "by_stadium": {
                stadium_names.get(sid, f"S{sid}"): m
                for sid, m in sorted(by_stadium.items())
            },
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Header
    print(f"\n=== {args.from_date} ~ {args.to_date} ===")
    print(f"Model: {model_dir}/ranking")
    st = meta.get("strategy", {})
    conc = st.get("top3_conc_threshold")
    gap23 = st.get("gap23_threshold")
    ev = st.get("ev_threshold")
    gap12 = st.get("gap12_min_threshold", 0.0)
    if conc is not None and gap23 is not None and ev is not None:
        print(
            f"Strategy: gap12>={gap12:.3f} conc>={conc:.4f} "
            f"gap23>={gap23:.3f} ev>={ev:.3f}"
        )
    else:
        print("Strategy: (incomplete strategy section)")
    excluded = st.get("excluded_stadiums") or []
    if excluded:
        excl_names = [stadium_names.get(s, str(s)) for s in excluded]
        print(f"Excluded stadiums: {', '.join(excl_names)}")

    # Overall summary
    overall = aggregate(purchases).get("all")
    print("\n=== Overall ===")
    if not overall:
        print("  (no purchases)")
    else:
        print(
            f"  {overall['n']}R, W {overall['w']} "
            f"({overall['hit_pct']:.1f}%), "
            f"B1勝率 {overall['b1_pct']:.0f}%, "
            f"ROI {overall['roi_pct']:.0f}%, "
            f"P/L {overall['pl']:+,.0f}"
        )
        if overall["w"] > 0:
            print(
                f"  配当: median ¥{overall['median_payout']*100:,.0f}, "
                f"mean ¥{overall['mean_payout']*100:,.0f}"
            )

    # Per-stadium breakdown (sorted by ROI descending)
    print("\n=== By stadium (sorted by ROI) ===")
    print(
        f"  {'name(総R)':<20} {'買':>5} {'W':>4} "
        f"{'Hit%':>6} {'ROI':>6} {'B1%':>6} {'P/L':>10}"
    )
    by_stadium = aggregate(purchases, key=lambda p: p.stadium_id)
    for sid in sorted(by_stadium, key=lambda s: -by_stadium[s]["roi_pct"]):
        name = stadium_names.get(sid, f"S{sid}")
        total_r = total_by_stadium.get(sid, 0)
        label = f"{name}({total_r})"
        print_row(label, by_stadium[sid])

    # Stadiums with zero purchases (excluded or no filter-passing races)
    silent_stadiums = [
        sid for sid in sorted(total_by_stadium)
        if sid not in by_stadium and total_by_stadium[sid] > 0
    ]
    if silent_stadiums:
        excl_set = set(excluded)
        print("\n=== Zero-purchase stadiums ===")
        for sid in silent_stadiums:
            name = stadium_names.get(sid, f"S{sid}")
            tag = " (excluded)" if sid in excl_set else ""
            print(f"  {name} ({total_by_stadium[sid]}R){tag}")

    # Time split
    if args.split_by != "none":
        print(f"\n=== By {args.split_by} ===")
        print(
            f"  {'period':<20} {'買':>5} {'W':>4} "
            f"{'Hit%':>6} {'ROI':>6} {'B1%':>6} {'P/L':>10}"
        )
        by_period = aggregate(
            purchases, key=lambda p: period_key(p, args.split_by)
        )
        for k in sorted(by_period):
            print_row(k, by_period[k])

    if args.show_importance:
        print_importance(model, meta["feature_columns"])


if __name__ == "__main__":
    main()
