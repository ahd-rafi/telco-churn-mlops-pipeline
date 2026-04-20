#!/usr/bin/env python3
"""
Telco Churn Training Pipeline
------------------------------
Orchestrates the full ML workflow: load → validate → preprocess → engineer → train → evaluate → log.
All experiment metadata is tracked via MLflow for reproducibility.
"""

import os
import sys
import time
import json
import argparse

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Allow imports from the project root so `src.*` packages resolve correctly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
ARTIFACTS_SUBDIR = "artifacts"


def _setup_mlflow(project_root: str, tracking_uri: str | None, experiment_name: str):
    """Initialise MLflow tracking with a file-backed store."""
    uri = tracking_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def _save_feature_metadata(feature_cols: list, target: str, project_root: str):
    """Persist the feature schema so the serving layer uses the exact same columns."""
    artifacts_dir = os.path.join(project_root, ARTIFACTS_SUBDIR)
    os.makedirs(artifacts_dir, exist_ok=True)

    # JSON list for quick inspection
    with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as fh:
        json.dump(feature_cols, fh)

    # MLflow text artifact (one column per line)
    mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

    # Pickle bundle used at serving time
    bundle = {"feature_columns": feature_cols, "target": target}
    pkl_path = os.path.join(artifacts_dir, "preprocessing.pkl")
    joblib.dump(bundle, pkl_path)
    mlflow.log_artifact(pkl_path)

    print(f"  → Saved {len(feature_cols)} feature columns for serving consistency")


def main(args):
    """End-to-end churn-model training pipeline."""

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ── MLflow ────────────────────────────────────────────────────────────
    _setup_mlflow(project_root, args.mlflow_uri, args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("classification_threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # ── 1. Load ──────────────────────────────────────────────────────
        print("\n[1/7] Loading data …")
        raw_df = load_data(args.input)
        print(f"  → {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

        # ── 2. Validate ──────────────────────────────────────────────────
        print("[2/7] Running data-quality checks (Great Expectations) …")
        is_valid, failures = validate_telco_data(raw_df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failures, indent=2),
                            artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed – {failures}")
        print("  → All expectations passed ✓")

        # ── 3. Preprocess ────────────────────────────────────────────────
        print("[3/7] Preprocessing …")
        clean_df = preprocess_data(raw_df)

        processed_path = os.path.join(project_root, "data", "processed",
                                      "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        clean_df.to_csv(processed_path, index=False)
        print(f"  → Cleaned dataset saved ({clean_df.shape[0]:,} rows)")

        # ── 4. Feature engineering ───────────────────────────────────────
        print("[4/7] Engineering features …")
        target_col = args.target
        if target_col not in clean_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        encoded_df = build_features(clean_df, target_col=target_col)

        # XGBoost needs int, not bool
        for col in encoded_df.select_dtypes(include=["bool"]).columns:
            encoded_df[col] = encoded_df[col].astype(int)
        print(f"  → {encoded_df.shape[1]} features after encoding")

        # Save feature metadata for serving
        feature_cols = list(encoded_df.drop(columns=[target_col]).columns)
        _save_feature_metadata(feature_cols, target_col, project_root)

        # ── 5. Split ─────────────────────────────────────────────────────
        print("[5/7] Splitting into train / test …")
        X = encoded_df.drop(columns=[target_col])
        y = encoded_df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        print(f"  → Train {X_train.shape[0]:,} | Test {X_test.shape[0]:,}")

        # Class-imbalance weight
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        imbalance_weight = neg / pos
        print(f"  → Imbalance ratio: {imbalance_weight:.2f}")

        # ── 6. Train ────────────────────────────────────────────────────
        print("[6/7] Training XGBoost classifier …")
        churn_model = XGBClassifier(
            n_estimators=301,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            scale_pos_weight=imbalance_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )

        t0 = time.time()
        churn_model.fit(X_train, y_train)
        train_secs = time.time() - t0
        mlflow.log_metric("train_time_sec", round(train_secs, 2))
        print(f"  → Trained in {train_secs:.1f}s")

        # ── 7. Evaluate ─────────────────────────────────────────────────
        print("[7/7] Evaluating on hold-out set …")
        t1 = time.time()
        probas = churn_model.predict_proba(X_test)[:, 1]
        y_pred = (probas >= args.threshold).astype(int)
        infer_secs = time.time() - t1
        mlflow.log_metric("inference_time_sec", round(infer_secs, 4))

        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, probas)

        for name, val in [("precision", prec), ("recall", rec),
                          ("f1", f1), ("roc_auc", auc)]:
            mlflow.log_metric(name, round(val, 4))

        print(f"\n  Precision : {prec:.3f}")
        print(f"  Recall    : {rec:.3f}")
        print(f"  F1        : {f1:.3f}")
        print(f"  ROC-AUC   : {auc:.3f}")

        # Log model
        mlflow.sklearn.log_model(churn_model, artifact_path="model")
        print("\n  Model logged to MLflow ✓")

        print(f"\n  Throughput: {len(X_test) / infer_secs:,.0f} samples/s")
        print("\n" + classification_report(y_test, y_pred, digits=3))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a churn prediction model (XGBoost + MLflow)"
    )
    parser.add_argument("--input", required=True,
                        help="Path to raw CSV, e.g. data/raw/Telco-Customer-Churn.csv")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", default="Telco Churn")
    parser.add_argument("--mlflow_uri", default=None,
                        help="Override MLflow tracking URI (default: file-based)")
    main(parser.parse_args())
