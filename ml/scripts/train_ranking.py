"""Train and save a LambdaRank ranking model for P2 strategy.

Uses Non-odds 21 features. Hyperparams can be loaded from model_meta.json
or specified via CLI.

Usage:
    # Train with params from tune_result:
    uv run --directory ml python -m scripts.train_ranking --save --model-meta models/tune_result

    # Train with explicit params:
    uv run --directory ml python -m scripts.train_ranking --save --relevance podium --n-estimators 1016
"""

import argparse
import time

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURES
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import (
    DEFAULT_PARAMS,
    load_training_params,
    save_model,
    save_model_meta,
)
from boatrace_tipster_ml.training import train_p2_ranker

FIELD_SIZE = 6
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_VAL_MONTHS = 2


def main():
    parser = argparse.ArgumentParser(description="Train LambdaRank ranking model (P2)")
    parser.add_argument("--save", action="store_true", required=True)
    parser.add_argument("--model-dir", default="models/draft/ranking")
    parser.add_argument("--model-meta", default=None,
                        help="Load hyperparams from this model directory's model_meta.json")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--end-date", default=None,
                        help="Training data end date (exclusive, YYYY-MM-DD). Prevents OOS leakage.")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--relevance", default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-child-samples", type=int, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--reg-alpha", type=float, default=None)
    parser.add_argument("--reg-lambda", type=float, default=None)
    args = parser.parse_args()

    # Build params: start from model_meta or defaults
    if args.model_meta:
        params = load_training_params(args.model_meta)
        print(f"Loaded params from {args.model_meta}/model_meta.json")
    else:
        params = {
            "extra_params": {},
            "n_estimators": DEFAULT_PARAMS["n_estimators"],
            "learning_rate": DEFAULT_PARAMS["learning_rate"],
            "relevance_scheme": "linear",
        }

    # CLI overrides
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.relevance is not None:
        params["relevance_scheme"] = args.relevance
    for key in [
        "num_leaves", "max_depth", "min_child_samples", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda",
    ]:
        val = getattr(args, key.replace("-", "_"))
        if val is not None:
            params["extra_params"][key] = val

    print(f"Training params:")
    print(f"  relevance={params['relevance_scheme']}, n_est={params['n_estimators']}, lr={params['learning_rate']:.4f}")
    print(f"  extra={params['extra_params']}")
    print(f"  features: {len(FEATURES)} (P2 non-odds)")

    t0 = time.time()
    print("Building features...")
    df = build_features_df(args.db_path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        return

    end_date = args.end_date or DEFAULT_END_DATE
    print(f"  {len(df)} entries ({len(df) // FIELD_SIZE} races), {len(FEATURES)} features")
    print(f"  end_date={end_date}, val_months={DEFAULT_VAL_MONTHS}")
    print("Training LambdaRank...")

    result = train_p2_ranker(
        df,
        hp=params["extra_params"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        relevance_scheme=params["relevance_scheme"],
        end_date=end_date,
        val_months=DEFAULT_VAL_MONTHS,
    )

    print(f"  Train: {result['n_train']}R, Val: {result['n_val']}R")

    all_hp = dict(params["extra_params"])
    all_hp["n_estimators"] = params["n_estimators"]
    all_hp["learning_rate"] = params["learning_rate"]
    all_hp["relevance_scheme"] = params["relevance_scheme"]

    save_model(result["model"], args.model_dir)
    save_model_meta(
        args.model_dir,
        feature_columns=FEATURES,
        hyperparameters=all_hp,
        training={
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "date_range": result["date_range"],
            "end_date": end_date,
            "val_months": DEFAULT_VAL_MONTHS,
        },
        feature_means=result["feature_means"],
    )
    print(f"Model saved to {args.model_dir}/")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
