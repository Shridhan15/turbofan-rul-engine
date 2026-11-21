import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, resolve_path
from src.data.loader import (
    INDEX_COLS,
    add_rul_target,
    load_cmapss_file,
    split_by_engine,
)
from src.models.metrics import evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame, drop_columns: list[str]) -> list[str]:
    """Return active feature columns, excluding IDs, target, and dropped sensors."""
    excluded = set(INDEX_COLS + ["RUL"] + drop_columns)
    return [col for col in df.columns if col not in excluded]


def build_pipeline(model_type: str, params: dict[str, Any]) -> Pipeline:
    """Build a scikit-learn Pipeline for the requested baseline model."""
    if model_type == "ridge":
        alpha = params.get("alpha", 1.0)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    if model_type == "lightgbm":
        return Pipeline([
            ("model", LGBMRegressor(**params, verbosity=-1)),
        ])

    raise ValueError(f"Unsupported model_type: '{model_type}'. Use 'ridge' or 'lightgbm'.")


def train_and_evaluate(model_override: str | None = None) -> dict[str, float]:
    """Run end-to-end baseline training, validation, and MLflow logging."""
    cfg = load_config()

    # 1. Load data, compute clipped RUL target, and split by engine
    raw_df = load_cmapss_file(resolve_path(cfg["paths"]["raw_train"]))
    df = add_rul_target(raw_df, clip_max=cfg["data"]["rul_clip_max"])
    train_df, val_df = split_by_engine(
        df,
        val_size=cfg["data"]["val_size"],
        random_seed=cfg["data"]["random_seed"],
    )

    # 2. Select the 14 active sensors from config.yaml
    feature_cols = get_feature_columns(df, cfg["features"]["drop_columns"])
    X_train, y_train = train_df[feature_cols], train_df["RUL"]
    X_val, y_val = val_df[feature_cols], val_df["RUL"]
    logger.info("Training with %d features: %s", len(feature_cols), feature_cols)

    # 3. Pick model type & parameters
    model_type = model_override or cfg["model"]["type"]
    model_params = cfg["model"]["params"] if model_type == "lightgbm" else {"alpha": 1.0}
    pipeline = build_pipeline(model_type, model_params)

    # 4. Configure local SQLite MLflow tracking anchored to PROJECT_ROOT
    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    run_name = f"baseline-{model_type}-raw-sensors"
    with mlflow.start_run(run_name=run_name):
        # Train
        pipeline.fit(X_train, y_train)

        # Predict & enforce physical lower bound (RUL >= 0)
        train_preds = np.clip(pipeline.predict(X_train), 0.0, None)
        val_preds = np.clip(pipeline.predict(X_val), 0.0, None)

        train_metrics = evaluate_predictions(y_train, train_preds)
        val_metrics = evaluate_predictions(y_val, val_preds)

        # Prefix metrics so train vs. val are crystal clear in MLflow
        logged_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        # Log params, metrics, config file, and trained model
        mlflow.log_params({
            "model_type": model_type,
            "n_features": len(feature_cols),
            "rul_clip_max": cfg["data"]["rul_clip_max"],
            "val_size": cfg["data"]["val_size"],
            **model_params,
        })
        mlflow.log_metrics(logged_metrics)
        mlflow.log_artifact(str(DEFAULT_CONFIG_PATH))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Also save a local copy in artifacts/ for quick inspection
        artifacts_dir: Path = resolve_path(cfg["paths"]["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / f"{run_name}.joblib"
        joblib.dump(pipeline, model_path)

        logger.info("Run '%s' complete.", run_name)
        logger.info(
            "Train RMSE: %.2f | Val RMSE: %.2f | Val NASA Mean: %.2f | Overpred Rate: %.1f%%",
            train_metrics["rmse"],
            val_metrics["rmse"],
            val_metrics["nasa_score_mean"],
            val_metrics["overprediction_rate"] * 100,
        )
        return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline RUL prediction model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["ridge", "lightgbm"],
        default=None,
        help="Override model.type from configs/config.yaml",
    )
    args = parser.parse_args()
    train_and_evaluate(model_override=args.model)