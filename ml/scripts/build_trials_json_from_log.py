"""Build a trials.json sidecar from an old tune log that predates the
trials.json saving feature (8378c12, 2026-04-12 19:48).

Parses lines of the form:
  [I 2026-04-11 21:43:35,405] Trial 294 finished with value: 0.0030 and parameters: {...}

and the "Top N" summary block:
  #294: growth=0.003004 kelly=0.4502 ROI=158% P/L=+50,550 n=734 rel=podium ...

to produce a trials.json compatible with parse_tune_log() in train_dev_model.py.

Usage:
    cd ml && uv run python scripts/build_trials_json_from_log.py \\
        ../logs/tune/2026-04-12_1713_server-tune.log
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

TRIAL_LINE_RE = re.compile(
    r"Trial (\d+) finished with value: ([\d.eE+-]+) and parameters: (\{[^}]*\})"
)
SUMMARY_RE = re.compile(
    r"#\s*(\d+):\s+growth=([\d.eE+-]+)\s+kelly=([\d.eE+-]+)\s+ROI=(\d+)%\s+P/L=([+\-\d,]+)\s+n=(\d+)\s+rel=(\w+)"
)


def main():
    log_path = Path(sys.argv[1])
    out_path = log_path.with_suffix(".trials.json")
    if out_path.exists():
        print(f"ERROR: {out_path} already exists", file=sys.stderr)
        sys.exit(1)

    text = log_path.read_text()

    # Extract per-trial params from optuna log lines
    trials_by_num: dict[int, dict] = {}
    for m in TRIAL_LINE_RE.finditer(text):
        num = int(m.group(1))
        value = float(m.group(2))
        params_str = m.group(3)
        try:
            params = ast.literal_eval(params_str)
        except Exception as e:
            print(f"WARN: cannot parse params for trial {num}: {e}", file=sys.stderr)
            continue
        trials_by_num[num] = {
            "number": num,
            "value": value,
            "state": "COMPLETE",
            "params": params,
            "user_attrs": {},
        }

    # Augment with summary info (growth/kelly/ROI/profit/races/relevance)
    for m in SUMMARY_RE.finditer(text):
        num = int(m.group(1))
        if num not in trials_by_num:
            continue
        trials_by_num[num]["user_attrs"].update({
            "growth": float(m.group(2)),
            "kelly": float(m.group(3)),
            "mean_roi": float(m.group(4)) / 100,
            "profit": float(m.group(5).replace(",", "")),
            "total_races": int(m.group(6)),
            "relevance": m.group(7),
        })

    # Try to detect fix_thresholds from the "Fixed thresholds:" line
    fix_match = re.search(r"Fixed thresholds:\s+(\{[^}]*\})", text)
    fix_thresholds = ast.literal_eval(fix_match.group(1)) if fix_match else {}

    out = {
        "created_at": "from-log",
        "n_trials": max(trials_by_num.keys()) + 1 if trials_by_num else 0,
        "seed": None,
        "fix_thresholds": fix_thresholds,
        "best_value": None,
        "best_trial": None,
        "trials": list(trials_by_num.values()),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} with {len(trials_by_num)} trials, "
          f"{sum(1 for t in trials_by_num.values() if t['user_attrs'])} have summary",
          file=sys.stderr)


if __name__ == "__main__":
    main()
