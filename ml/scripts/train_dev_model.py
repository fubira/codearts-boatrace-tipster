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
from scripts.tune_p2 import FEATURES

FIELD_SIZE = 6
MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "registry.json"


def is_candidate_dir(name: str) -> bool:
    """True if name matches dev candidate pattern (e.g., 'aa_294')."""
    return bool(re.fullmatch(r"[a-z]{2}_\d+", name))


def next_prefix(current: str) -> str:
    """aa → ab → ... → az → ba → ... → zz."""
    a, b = current[0], current[1]
    if b < "z":
        return a + chr(ord(b) + 1)
    if a < "z":
        return chr(ord(a) + 1) + "a"
    raise ValueError("Ran out of prefixes (zz reached)")


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"next_prefix": "aa", "runs": {}}


def save_registry(reg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def parse_tune_log(log_path: Path) -> dict:
    """Parse Optuna log. Returns {trial_num: params_dict} and run info."""
    if not log_path.exists():
        print(f"ERROR: Log not found: {log_path}")
        sys.exit(1)

    with open(log_path) as f:
        log = f.read()

    # Command line (trials, seed, relevance, fix-thresholds)
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

    # Trials
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
            trials[num] = {"growth": val, "params": params}
        except json.JSONDecodeError:
            continue

    # Best trial
    best_m = re.search(r"Best Growth:\s+([\d\.]+)", log)
    best_growth = float(best_m.group(1)) if best_m else None
    best_trial_m = re.search(r"Best trial metrics:.*?#\s*(\d+):", log, re.DOTALL)

    return {
        "trials": trials,
        "cmd": cmd,
        "n_trials": int(trials_m.group(1)) if trials_m else None,
        "seed": int(seed_m.group(1)) if seed_m else None,
        "fix_thresholds": fix_thresholds,
        "best_growth": best_growth,
    }


def params_to_hp(params: dict) -> tuple[dict, float, int, float]:
    """Extract HP dict, learning_rate, n_estimators, top3_conc_threshold from Optuna params."""
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
    conc = params.get("top3_conc_threshold", 0.0)
    return hp, lr, n_est, conc


def train_one(df, prefix, trial_num, params, end_date, val_months, log_path,
              gap23_th, ev_th):
    """Train a single trial and save to models/dev/<prefix>_<trial>/ranking/."""
    hp, lr, n_est, conc_th = params_to_hp(params)

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
    print(f"[{model_name}] Training (n_train={n_train}, n_val={n_val}, lr={lr:.4f}, n_est={n_est}, conc={conc_th:.3f})...",
          flush=True)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = train_model(
            X[~val_mask], y[~val_mask], meta[~val_mask],
            X[val_mask], y[val_mask], meta[val_mask],
            n_estimators=n_est, learning_rate=lr,
            relevance_scheme="podium", extra_params=hp,
            early_stopping_rounds=None,  # Full n_est (matches Optuna trial)
        )
    print(f"[{model_name}] Done in {time.time()-t0:.0f}s", flush=True)

    feature_means = {c: float(X[c].astype("float64").mean()) for c in FEATURES}
    all_hp = dict(hp)
    all_hp["n_estimators"] = n_est
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
        "ev_threshold": ev_th,
        "unit_divisor": 150,
        "bet_cap": 30000,
        "ev_basis": "3連単 odds per ticket",
        "features": "non_odds_21",
        "source": model_name,
    }
    with open(meta_path, "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    return model_name


def cmd_list() -> None:
    reg = load_registry()
    print(f"Next prefix: {reg['next_prefix']}")
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
    reg = load_registry()
    prefix = args.prefix or reg["next_prefix"]
    log_path = Path(args.tune_log)

    # Parse log
    print(f"Parsing {log_path}...")
    log_info = parse_tune_log(log_path)
    print(f"  {len(log_info['trials'])} trials, best growth={log_info['best_growth']}")

    # Select trials
    trial_nums = [int(x) for x in args.trials.split(",")]
    missing = [t for t in trial_nums if t not in log_info["trials"]]
    if missing:
        print(f"ERROR: Trials not found in log: {missing}")
        sys.exit(1)

    # Get fix_thresholds from log
    fix_th = log_info["fix_thresholds"]
    gap23_th = fix_th.get("gap23", 0.13)
    ev_th = fix_th.get("ev", 0.0)

    # Load features once
    print("Loading features...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_features_df(args.db_path, end_date=args.end_date)

    # Train each trial
    saved = []
    for tn in trial_nums:
        params = log_info["trials"][tn]["params"]
        name = train_one(df, prefix, tn, params, args.end_date, args.val_months,
                         log_path, gap23_th, ev_th)
        saved.append(tn)

    # Advance next_prefix only when a NEW prefix was consumed
    if prefix == reg["next_prefix"]:
        reg["next_prefix"] = next_prefix(prefix)
        save_registry(reg)

    print(f"\nSaved [{prefix.upper()}]: trials {saved}")
    print(f"Next prefix: {reg['next_prefix']}")


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
