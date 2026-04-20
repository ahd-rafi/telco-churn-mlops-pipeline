"""
Inference module – serves churn predictions in production.

Loads the trained XGBoost model (via MLflow) and applies the *exact same*
feature transformations that were used during training so there is zero
train / serve skew.
"""

import os
import glob

import pandas as pd
import mlflow

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_DIR = "/app/model"          # Path inside the Docker image

_model = None                     # Will be set at import time

try:
    _model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"[inference] Model loaded from {MODEL_DIR}")
except Exception as _primary_err:
    # Fallback: scan local mlruns for the latest model (development only)
    _local_paths = glob.glob("./mlruns/*/*/artifacts/model")
    if _local_paths:
        _latest = max(_local_paths, key=os.path.getmtime)
        _model = mlflow.pyfunc.load_model(_latest)
        MODEL_DIR = _latest
        print(f"[inference] Dev-fallback model loaded from {_latest}")
    else:
        raise RuntimeError(
            f"Could not load model from {MODEL_DIR} ({_primary_err}). "
            "No local mlruns fallback found either."
        )

# ---------------------------------------------------------------------------
# Feature schema (must match training output exactly)
# ---------------------------------------------------------------------------
_feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
with open(_feature_file) as _fh:
    FEATURE_COLS = [line.strip() for line in _fh if line.strip()]
print(f"[inference] {len(FEATURE_COLS)} feature columns loaded")

# ---------------------------------------------------------------------------
# Constants – identical to training
# ---------------------------------------------------------------------------
BINARY_MAP = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known numeric columns and fill NaNs with 0."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _encode_binaries(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic 0/1 mapping for binary categorical features."""
    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip()
                .map(mapping).astype("Int64").fillna(0).astype(int)
            )
    return df


def _one_hot_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode any leftover object columns (drop_first to match training)."""
    obj_cols = list(df.select_dtypes(include=["object"]).columns)
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    # XGBoost expects int, not bool
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df


def _align_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex columns to the training schema; missing cols filled with 0."""
    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Full serving-time feature pipeline (mirrors training exactly)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = _coerce_numerics(df)
    df = _encode_binaries(df)
    df = _one_hot_remaining(df)
    df = _align_to_schema(df)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict(input_dict: dict) -> str:
    """
    Accept raw customer data (dict) and return a human-readable churn label.

    Returns
    -------
    str
        ``"Likely to churn"`` or ``"Not likely to churn"``
    """
    row_df = pd.DataFrame([input_dict])
    features = _transform(row_df)

    raw_pred = _model.predict(features)

    # Normalise output (numpy → list → scalar)
    if hasattr(raw_pred, "tolist"):
        raw_pred = raw_pred.tolist()
    result = raw_pred[0] if isinstance(raw_pred, (list, tuple)) and len(raw_pred) == 1 else raw_pred

    return "Likely to churn" if result == 1 else "Not likely to churn"
