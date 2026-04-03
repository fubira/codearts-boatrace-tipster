"""Train and save a LambdaRank ranking model for production use.

Usage:
    uv run --directory ml python -m scripts.train_ranking --save
"""

import argparse
import time

from boatrace_tipster_ml.db import DEFAULT_DB_PATH
from boatrace_tipster_ml.feature_config import FEATURE_COLS, prepare_feature_matrix
from boatrace_tipster_ml.features import build_features_df
from boatrace_tipster_ml.model import save_model, save_model_meta, train_model


def main():
    parser = argparse.ArgumentParser(description="Train LambdaRank ranking model")
    parser.add_argument("--save", action="store_true", required=True)
    parser.add_argument("--model-dir", default="models/ranking")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    t0 = time.time()
    print("Building features...")
    df = build_features_df(DEFAULT_DB_PATH)
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
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=50,
    )

    # Feature means for NaN fallback
    feature_means = {c: float(X[c].astype("float64").mean()) for c in FEATURE_COLS}

    save_model(model, args.model_dir)
    save_model_meta(
        args.model_dir,
        feature_columns=FEATURE_COLS,
        hyperparameters={"n_estimators": args.n_estimators, "learning_rate": args.learning_rate},
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
