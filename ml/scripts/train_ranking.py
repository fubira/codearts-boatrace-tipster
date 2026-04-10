"""Train and save a LambdaRank ranking model for production use.

Usage:
    # Train with params from existing model_meta.json:
    uv run --directory ml python -m scripts.train_ranking --save --model-meta models/trifecta_v1/ranking

    # Train with explicit params:
    uv run --directory ml python -m scripts.train_ranking --save --relevance win_only --num-leaves 92
"""

import argparse
import time

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURE_COLS, prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import (
    DEFAULT_PARAMS,
    load_training_params,
    save_model,
    save_model_meta,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train LambdaRank ranking model")
    parser.add_argument("--save", action="store_true", required=True)
    parser.add_argument("--model-dir", default="models/draft/ranking")
    parser.add_argument("--model-meta", default=None,
                        help="Load hyperparams from this model directory's model_meta.json")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--end-date", default=None,
                        help="Training data end date (exclusive, YYYY-MM-DD). Prevents OOS leakage.")
    # Individual param overrides (applied after --model-meta)
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
    for key, attr in [
        ("num_leaves", "num_leaves"), ("max_depth", "max_depth"),
        ("min_child_samples", "min_child_samples"), ("subsample", "subsample"),
        ("colsample_bytree", "colsample_bytree"),
        ("reg_alpha", "reg_alpha"), ("reg_lambda", "reg_lambda"),
    ]:
        val = getattr(args, attr.replace("-", "_"))
        if val is not None:
            params["extra_params"][key] = val

    print(f"Training params:")
    print(f"  relevance={params['relevance_scheme']}, n_est={params['n_estimators']}, lr={params['learning_rate']:.4f}")
    print(f"  extra={params['extra_params']}")

    t0 = time.time()
    print("Building features...")
    df = build_features_df(args.db_path, end_date=args.end_date)
    X, y, meta = prepare_feature_matrix(df)
    print(f"  {len(X)} entries ({len(X)//6} races), {len(FEATURE_COLS)} features")

    # Val = last ~2 months
    dates = sorted(df["race_date"].unique())
    val_start = dates[max(0, len(dates) - 60)]

    race_dates = meta["race_date"].values
    train_mask = race_dates < val_start
    val_mask = race_dates >= val_start

    X_train, y_train, meta_train = X[train_mask], y[train_mask], meta[train_mask]
    X_val, y_val, meta_val = X[val_mask], y[val_mask], meta[val_mask]

    print(f"  Train: {len(X_train)//6}R, Val: {len(X_val)//6}R")
    print("Training LambdaRank...")

    model, metrics = train_model(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        extra_params=params["extra_params"],
        relevance_scheme=params["relevance_scheme"],
        early_stopping_rounds=50,
    )

    # Feature means for NaN fallback
    feature_means = {c: float(X[c].astype("float64").mean()) for c in FEATURE_COLS}

    # Save complete hyperparameters
    all_hp = dict(params["extra_params"])
    all_hp["n_estimators"] = params["n_estimators"]
    all_hp["learning_rate"] = params["learning_rate"]
    all_hp["relevance_scheme"] = params["relevance_scheme"]

    save_model(model, args.model_dir)
    save_model_meta(
        args.model_dir,
        feature_columns=FEATURE_COLS,
        hyperparameters=all_hp,
        training={
            "n_train": len(X_train) // 6,
            "n_val": len(X_val) // 6,
            "date_range": f"{dates[0]} ~ {dates[-1]}",
        },
        feature_means=feature_means,
    )
    print(f"Model saved to {args.model_dir}/")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
