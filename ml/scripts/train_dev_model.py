"""Train dev candidate models from Optuna tune logs.

Parses HP from tune log, trains with specified trials, saves to
models/dev/<prefix>_<trial>/ranking/, and updates registry.json.

Prefix naming: aa, ab, ac, ..., az, ba, bb, ... (lowercase for ease of typing;
display can use upper case). Each Optuna run gets one prefix.

Usage:
    # Register new Optuna run and save top trials (auto-allocate prefix)
    uv run python -m scripts.train_dev_model \\
        --tune-log logs/tune/2026-04-12_1713_server-tune.log \\
        --trials 266,294,28

    # Use specific prefix (e.g., re-training within same run)
    uv run python -m scripts.train_dev_model \\
        --tune-log PATH --trials 294 --prefix aa

    # List current dev models
    uv run python -m scripts.train_dev_model --list
"""

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import save_model, save_model_meta, train_model
from boatrace_tipster_ml.registry import next_prefix as registry_next_prefix
from boatrace_tipster_ml.registry import peek_prefix
from scripts.tune_p2 import FEATURES

FIELD_SIZE = 6
MODELS_DIR = Path("models")


def is_candidate_dir(name: str) -> bool:
    """True if name matches dev candidate pattern (e.g., 'aa_294')."""
    return bool(re.fullmatch(r"[a-z]{2}_\d+", name))


def parse_tune_log(log_path: Path) -> dict:
    """Parse Optuna log. Prefers trials.json if available, else falls back
    to log text parsing.

    Returns: {
        "trials": {num: {"growth", "params", "user_attrs"}},
        "fix_thresholds": dict,
        "n_trials": int, "seed": int, "best_growth": float,
    }
    """
    if not log_path.exists():
        print(f"ERROR: Log not found: {log_path}")
        sys.exit(1)

    # Prefer trials.json (includes user_attrs like best_iter).
    # Look for <log_basename>.trials.json adjacent to the log first, then
    # fall back to models/tune_result/trials.json.
    trials_json_candidates = [
        log_path.with_suffix(".trials.json"),
        MODELS_DIR / "tune_result" / "trials.json",
    ]
    for trials_json in trials_json_candidates:
        if not trials_json.exists():
            continue
        try:
            with open(trials_json) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as err:
            print(
                f"  WARN: failed to read {trials_json}: {err}. "
                f"Falling back to next source.",
                file=sys.stderr,
            )
            continue
        trials = {
            t["number"]: {
                "growth": t["value"],
                "params": t["params"],
                "user_attrs": t.get("user_attrs", {}),
            }
            for t in data.get("trials", [])
        }
        return {
            "trials": trials,
            "fix_thresholds": data.get("fix_thresholds", {}),
            "n_trials": data.get("n_trials"),
            "seed": data.get("seed"),
            "run_prefix": data.get("run_prefix"),
            "best_growth": data.get("best_value"),
            "source": str(trials_json),
        }

    # Fallback: parse log text (no user_attrs available)
    with open(log_path) as f:
        log = f.read()

    cmd_m = re.search(r"Command:\s+(.*)", log)
    cmd = cmd_m.group(1) if cmd_m else ""
    trials_m = re.search(r"--trials\s+(\d+)", cmd)
    seed_m = re.search(r"--seed\s+(\d+)", cmd)
    fix_m = re.search(r"--fix-thresholds\s+'?([^'\"]+)", cmd)

    fix_thresholds = {}
    if fix_m:
        for pair in fix_m.group(1).split(","):
            k, v = pair.strip().split("=")
            fix_thresholds[k.strip()] = float(v.strip())

    trial_re = re.compile(
        r"Trial (\d+) finished with value: ([\-\d\.e]+) and parameters: (\{[^}]+\})"
    )
    trials = {}
    for m in trial_re.finditer(log):
        num = int(m.group(1))
        val = float(m.group(2))
        params_str = m.group(3).replace("'", '"')
        try:
            params = json.loads(params_str)
            trials[num] = {"growth": val, "params": params, "user_attrs": {}}
        except json.JSONDecodeError:
            continue

    best_m = re.search(r"Best Growth:\s+([\d\.]+)", log)
    best_growth = float(best_m.group(1)) if best_m else None

    return {
        "trials": trials,
        "fix_thresholds": fix_thresholds,
        "n_trials": int(trials_m.group(1)) if trials_m else None,
        "seed": int(seed_m.group(1)) if seed_m else None,
        "run_prefix": None,  # log-only fallback can't recover this
        "best_growth": best_growth,
        "source": "log",
    }


def params_to_hp(
    params: dict, conc_default: float = 0.0, gap12_default: float = 0.0,
) -> tuple[dict, float, int, float, float]:
    """Extract HP dict, learning_rate, n_estimators, top3_conc_threshold,
    gap12_min_threshold from Optuna params.

    `conc_default` / `gap12_default` are used when the threshold was fixed via
    --fix-thresholds during tune (so it's absent from params). Caller passes
    fix_thresholds["top3_conc"] / fix_thresholds["gap12"] in that case.
    """
    hp = {
        "num_leaves": params["num_leaves"],
        "max_depth": params["max_depth"],
        "min_child_samples": params["min_child_samples"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
    }
    lr = params["learning_rate"]
    n_est = params["n_estimators"]
    conc = params.get("top3_conc_threshold", conc_default)
    gap12 = params.get("gap12_min_threshold", gap12_default)
    return hp, lr, n_est, conc, gap12


def train_one(df, prefix, trial_num, trial_info, end_date, val_months, log_path,
              gap23_th, ev_th, conc_th_default=0.0, gap12_th_default=0.0):
    """Train a single trial and save to models/<prefix>_<trial>/ranking/.

    Uses avg_best_iter from user_attrs (if available) instead of params'
    n_estimators upper bound. Disables early stopping because we already
    know the effective iteration count from WF-CV.
    """
    params = trial_info["params"]
    user_attrs = trial_info.get("user_attrs", {})
    hp, lr, n_est_upper, conc_th, gap12_th = params_to_hp(
        params, conc_default=conc_th_default, gap12_default=gap12_th_default,
    )

    # Effective iter count: prefer avg_best_iter from WF-CV if available
    avg_best_iter = user_attrs.get("avg_best_iter")
    if avg_best_iter is not None and avg_best_iter > 0:
        effective_n_est = int(avg_best_iter)
        src = f"avg_best_iter from WF-CV (upper {n_est_upper})"
    else:
        effective_n_est = n_est_upper
        src = f"n_est (no avg_best_iter in trial attrs — using upper bound)"

    train_df = df[df["race_date"] < end_date]
    val_start = pd.Timestamp(end_date) - pd.DateOffset(months=val_months)
    val_mask = train_df["race_date"] >= str(val_start.date())

    X = train_df[FEATURES].copy()
    y = train_df["finish_position"]
    meta = train_df[["race_id", "racer_id", "race_date", "boat_number"]].copy()

    n_train = int((~val_mask).sum() // FIELD_SIZE)
    n_val = int(val_mask.sum() // FIELD_SIZE)

    model_name = f"{prefix}_{trial_num}"
    output_dir = MODELS_DIR / model_name / "ranking"
    print(
        f"[{model_name}] Training (n_train={n_train}, n_val={n_val}, "
        f"lr={lr:.4f}, n_est={effective_n_est}, conc={conc_th:.3f}) {src}",
        flush=True,
    )
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            X[~val_mask], y[~val_mask], meta[~val_mask],
            X[val_mask], y[val_mask], meta[val_mask],
            n_estimators=effective_n_est, learning_rate=lr,
            relevance_scheme="podium", extra_params=hp,
            early_stopping_rounds=None,  # Match WF-CV effective iterations
        )
    print(f"[{model_name}] Done in {time.time()-t0:.0f}s", flush=True)

    feature_means = {c: float(X[c].astype("float64").mean()) for c in FEATURES}
    all_hp = dict(hp)
    all_hp["n_estimators"] = effective_n_est
    all_hp["n_estimators_upper"] = n_est_upper
    all_hp["learning_rate"] = lr
    all_hp["relevance_scheme"] = "podium"

    save_model(model, str(output_dir))
    save_model_meta(
        str(output_dir),
        feature_columns=FEATURES,
        hyperparameters=all_hp,
        training={
            "n_train": n_train,
            "n_val": n_val,
            "end_date": end_date,
            "val_months": val_months,
            "dev_prefix": prefix,
            "tune_log": str(log_path),
            "trial_number": trial_num,
            "avg_best_iter": avg_best_iter,
        },
        feature_means=feature_means,
    )

    # Add strategy section
    meta_path = output_dir / "model_meta.json"
    with open(meta_path) as f:
        m = json.load(f)
    m["strategy"] = {
        "type": "P2",
        "bet_pattern": "1-(2,3)-(2,3) adaptive",
        "gap23_threshold": gap23_th,
        "top3_conc_threshold": conc_th,
        "gap12_min_threshold": gap12_th,
        "ev_threshold": ev_th,
        "unit_divisor": 200,
        "bet_cap": 30000,
        "ev_basis": "3連単 odds per ticket",
        "features": "non_odds_21",
        "source": model_name,
    }
    with open(meta_path, "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    return model_name


def cmd_list() -> None:
    print(f"Next prefix: {peek_prefix()}")
    print(f"\nCandidate models (models/[a-z]{{2}}_<trial>/):")
    found = False
    for p in sorted(MODELS_DIR.glob("*/ranking/model_meta.json")):
        name = p.parent.parent.name
        if not is_candidate_dir(name):
            continue
        found = True
        with open(p) as f:
            m = json.load(f)
        tr = m.get("training", {})
        st = m.get("strategy", {})
        prefix = tr.get("dev_prefix", "?")
        trial = tr.get("trial_number", "?")
        log = tr.get("tune_log", "?")
        end_date = tr.get("end_date", "?")
        conc = st.get("top3_conc_threshold", "?")
        print(f"  [{prefix.upper()}_{trial}] end_date={end_date} conc={conc} log={log}")
    if not found:
        print("  (none)")


def cmd_train(args) -> None:
    log_path = Path(args.tune_log)

    # Parse log (prefers trials.json for user_attrs access)
    print(f"Parsing {log_path}...")
    log_info = parse_tune_log(log_path)
    print(
        f"  source={log_info['source']}, {len(log_info['trials'])} trials, "
        f"best growth={log_info['best_growth']}"
    )
    if log_info["source"] == "log":
        print(
            "  WARN: Using log text fallback (no trials.json). best_iter tracking "
            "unavailable — will fall back to n_estimators upper bound."
        )

    # Validate trials BEFORE resolving prefix. This prevents consuming a fresh
    # prefix from the registry when the train will fail anyway (e.g., wrong
    # tune-log path that fell back to log text and lost the trial table).
    trial_nums = [int(x) for x in args.trials.split(",")]
    missing = [t for t in trial_nums if t not in log_info["trials"]]
    if missing:
        print(f"ERROR: Trials not found in log: {missing}")
        sys.exit(1)

    # Resolve prefix priority: explicit --prefix > tune's run_prefix > new
    # prefix from registry. The tune itself reserves a prefix at start time
    # (since v0.26+) so multiple dev models from the same tune share it
    # automatically.
    consumed_new = False
    if args.prefix:
        prefix = args.prefix
    elif log_info.get("run_prefix"):
        prefix = log_info["run_prefix"]
    else:
        prefix = registry_next_prefix()
        consumed_new = True

    # Get fix_thresholds from log. top3_conc fallback is used by params_to_hp
    # because trials sampled with --fix-thresholds top3_conc=X don't carry the
    # value in their params dict.
    fix_th = log_info["fix_thresholds"]
    gap23_th = fix_th.get("gap23", 0.13)
    ev_th = fix_th.get("ev", 0.0)
    conc_th_default = fix_th.get("top3_conc", 0.0)
    gap12_th_default = fix_th.get("gap12", 0.0)

    # Load features once
    print("Loading features...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path, end_date=args.end_date)

    # Train each trial
    saved = []
    for tn in trial_nums:
        trial_info = log_info["trials"][tn]
        name = train_one(df, prefix, tn, trial_info, args.end_date, args.val_months,
                         log_path, gap23_th, ev_th,
                         conc_th_default=conc_th_default,
                         gap12_th_default=gap12_th_default)
        saved.append(tn)

    # The counter is already advanced if registry_next_prefix() was called above.
    print(f"\nSaved [{prefix.upper()}]: trials {saved}")
    print(f"Next prefix: {peek_prefix()}")
    if not consumed_new:
        if args.prefix:
            print("  (counter not advanced — --prefix override)")
        else:
            print(f"  (counter not advanced — inherited '{prefix}' from tune log)")


def main():
    parser = argparse.ArgumentParser(description="Train dev candidate models from Optuna logs")
    parser.add_argument("--list", action="store_true", help="List dev models and registry")
    parser.add_argument("--tune-log", help="Path to Optuna tune log (required for training)")
    parser.add_argument("--trials", help="Comma-separated trial numbers to save")
    parser.add_argument("--prefix", default=None,
                        help="Override prefix (default: auto from registry)")
    parser.add_argument("--end-date", default="2026-01-01",
                        help="Training data end (exclusive). Default: 2026-01-01 for OOS eval")
    parser.add_argument("--val-months", type=int, default=2)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    if not args.tune_log or not args.trials:
        parser.print_help()
        sys.exit(1)

    cmd_train(args)


if __name__ == "__main__":
    main()
