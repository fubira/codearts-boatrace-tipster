"""Sweep a strategy threshold on a fixed model over OOS period.

Supports sweeping `top3_conc_threshold`, `ev_threshold`, or
`gap23_threshold` without retraining. Other strategy values are taken
from the model's own `model_meta.json`.

Usage:
    cd ml && uv run python -m scripts.threshold_sweep --from 2026-01-01 --to "$(date +%F)"
    cd ml && uv run python -m scripts.threshold_sweep --from 2026-01-01 --to "$(date +%F)" \
        --axis ev --start 0.0 --stop 0.5 --step 0.05
    cd ml && uv run python -m scripts.threshold_sweep --from 2026-01-01 --to "$(date +%F)" \
        --axis conc --start 0.30 --stop 0.80 --step 0.05
"""

import argparse
import contextlib
import copy
import io
import sys

from boatrace_tipster_ml.db import DEFAULT_DB_PATH, get_connection
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import load_model, load_model_meta
from boatrace_tipster_ml.registry import get_active_model_dir
from scripts.analyze_model import aggregate, evaluate_period

AXIS_META_KEY = {
    "conc": "top3_conc_threshold",
    "ev": "ev_threshold",
    "gap23": "gap23_threshold",
    "gap12": "gap12_min_threshold",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--axis", choices=list(AXIS_META_KEY.keys()), default="conc",
        help="Which strategy threshold to sweep",
    )
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--stop", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    args = parser.parse_args()

    defaults = {
        "conc": (0.30, 0.80, 0.05),
        "ev": (-0.20, 0.60, 0.05),
        "gap23": (0.00, 0.30, 0.02),
        "gap12": (0.00, 0.20, 0.02),
    }
    d_start, d_stop, d_step = defaults[args.axis]
    start = args.start if args.start is not None else d_start
    stop = args.stop if args.stop is not None else d_stop
    step = args.step if args.step is not None else d_step
    meta_key = AXIS_META_KEY[args.axis]

    model_dir = args.model_dir or get_active_model_dir()
    print(f"Model: {model_dir}/ranking", file=sys.stderr)
    model = load_model(f"{model_dir}/ranking")
    meta_orig = load_model_meta(f"{model_dir}/ranking")

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

    cur_value = meta_orig.get("strategy", {}).get(meta_key)
    print(f"\n=== {args.axis} sweep on {model_dir}/ranking ===")
    print(f"Period: {args.from_date} ~ {args.to_date}")
    print(f"Current model {meta_key}: {cur_value}")
    print(f"Sweep: {start} → {stop} step {step}")
    print()

    header = (
        f"{args.axis:>8} {'n':>5} {'W':>4} {'Hit%':>6} {'ROI':>6} "
        f"{'P/L':>10} {'配当med':>9} {'B1%':>5}"
    )
    print(header)
    print("-" * len(header))

    vals: list[float] = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 6))
        v += step
    if cur_value is not None and not any(abs(cur_value - x) < 1e-6 for x in vals):
        vals.append(round(cur_value, 6))
        vals.sort()

    for val in vals:
        meta = copy.deepcopy(meta_orig)
        meta.setdefault("strategy", {})[meta_key] = val
        purchases, _ = evaluate_period(
            model, meta, df, odds, args.from_date, args.to_date
        )
        m = aggregate(purchases).get("all")
        if not m:
            print(f"{val:>8.4f} {'-':>5} {'-':>4} {'-':>6} {'-':>6} {'-':>10}")
            continue
        mark = (
            " ← current"
            if cur_value is not None and abs(val - cur_value) < 1e-6
            else ""
        )
        print(
            f"{val:>8.4f} {m['n']:>5} {m['w']:>4} "
            f"{m['hit_pct']:>5.1f}% {m['roi_pct']:>5.0f}% "
            f"{m['pl']:>+9,.0f} "
            f"¥{m['median_payout'] * 100:>7,.0f} "
            f"{m['b1_pct']:>4.0f}%{mark}"
        )


if __name__ == "__main__":
    main()
