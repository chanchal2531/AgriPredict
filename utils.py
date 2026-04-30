"""
utils.py — Smart Crop Prediction & Risk Analysis Backend
=========================================================
Centralised preprocessing, feature-engineering, encoding helpers,
and risk-scoring utilities shared by train_model.py and app.py.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RAW_PATH      = os.path.join(DATA_DIR, "crop_production_raw.xlsx")
CLEANED_PATH  = os.path.join(DATA_DIR, "crop_data.csv")        # fast CSV copy
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")

# ── Column names (single source of truth) ────────────────────────────────────
COL_STATE = "State_Name"
COL_CROP       = "Crop"
COL_YEAR       = "Crop_Year"
COL_AREA       = "Area"
COL_PRODUCTION = "Production"
COL_YIELD      = "Yield"
COL_RISK       = "Risk_Category"

# Risk thresholds (yield-based percentile boundaries)
RISK_LOW_THRESH  = 25   # percentile
RISK_HIGH_THRESH = 75   # percentile

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Excel dataset and strip column whitespace."""
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    logger.info(f"Raw data shape: {df.shape}")
    return df


def load_cleaned_data(path: str = CLEANED_PATH) -> pd.DataFrame:
    logger.info(f"Loading cleaned data from: {path}")
    
    df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')

    # FORCE RENAME ALL COLUMNS (POSITION BASED)
    df.columns = [
        "State_Name",
        "District_Name",
        "Crop_Year",
        "Season",
        "Crop",
        "Area",
        "Production"
    ]

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def summarise_missing(df: pd.DataFrame) -> None:
    """Print a missing-value summary for every column."""
    total   = df.isnull().sum()
    percent = (total / len(df) * 100).round(2)
    summary = pd.DataFrame({"Missing Count": total, "Missing %": percent})
    summary = summary[summary["Missing Count"] > 0]
    if summary.empty:
        logger.info("No missing values detected.")
    else:
        logger.info(f"Missing value summary:\n{summary}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numerical columns → mean
      - Categorical/object columns → mode (most frequent)
    """
    df = df.copy()
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            fill_val = df[col].mean()
            df[col].fillna(fill_val, inplace=True)
            logger.info(f"  Filled '{col}' (numeric) with mean={fill_val:.4f}")
        else:
            fill_val = df[col].mode()[0]
            df[col].fillna(fill_val, inplace=True)
            logger.info(f"  Filled '{col}' (categorical) with mode='{fill_val}'")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Strip column names
      2. Summarise and handle missing values
      3. Keep only the columns required by the models
      4. Strip string whitespace in Crop / State_Name
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    logger.info("── Missing value summary (before imputation) ──")
    summarise_missing(df)

    df = handle_missing_values(df)

    logger.info("── Missing value summary (after imputation) ──")
    summarise_missing(df)

    # Normalise string fields
    df[COL_CROP]  = df[COL_CROP].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()

    # Ensure numeric types
    df[COL_AREA]       = pd.to_numeric(df[COL_AREA],       errors="coerce").fillna(0)
    df[COL_PRODUCTION] = pd.to_numeric(df[COL_PRODUCTION], errors="coerce").fillna(0)
    df[COL_YEAR]       = pd.to_numeric(df[COL_YEAR],       errors="coerce").fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features:
      - Yield = Production / Area  (safe: 0-area rows → 0)
      - Risk_Category based on yield percentile bins
    """
    df = df.copy()

    # ── Yield (safe division) ────────────────────────────────────────────────
    df[COL_YIELD] = np.where(
        df[COL_AREA] > 0,
        df[COL_PRODUCTION] / df[COL_AREA],
        0.0,
    )
    # Replace inf / -inf that may arise from edge cases
    df[COL_YIELD] = df[COL_YIELD].replace([np.inf, -np.inf], 0.0)
    df[COL_YIELD] = df[COL_YIELD].fillna(0.0)
    logger.info(f"Yield stats → min={df[COL_YIELD].min():.2f}, "
                f"mean={df[COL_YIELD].mean():.2f}, max={df[COL_YIELD].max():.2f}")

    # ── Risk Category ────────────────────────────────────────────────────────
    low_thresh  = np.percentile(df[COL_YIELD], RISK_LOW_THRESH)
    high_thresh = np.percentile(df[COL_YIELD], RISK_HIGH_THRESH)
    logger.info(f"Risk thresholds → Low<{low_thresh:.3f}, "
                f"Medium={low_thresh:.3f}–{high_thresh:.3f}, High>{high_thresh:.3f}")

    df[COL_RISK] = pd.cut(
        df[COL_YIELD],
        bins=[-np.inf, low_thresh, high_thresh, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    logger.info(f"Risk distribution:\n{df[COL_RISK].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def build_encoders(df: pd.DataFrame) -> dict:
    """
    Fit LabelEncoders for Crop and State_Name.
    Returns a dict:  {"crop": LabelEncoder, "state": LabelEncoder}
    """
    encoders = {}
    for key, col in [("crop", COL_CROP), ("state", COL_STATE)]:
        le = LabelEncoder()
        le.fit(df[col].astype(str).str.strip().unique())
        encoders[key] = le
        logger.info(f"Encoder '{key}': {len(le.classes_)} unique classes")
    return encoders


def save_encoders(encoders: dict, path: str = ENCODERS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"Encoders saved → {path}")


def load_encoders(path: str = ENCODERS_PATH) -> dict:
    with open(path, "rb") as f:
        encoders = pickle.load(f)
    logger.info(f"Encoders loaded from {path}")
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Transform Crop and State_Name columns using pre-fitted encoders.
    Unknown labels are replaced with -1 (handled gracefully in the API).
    """
    df = df.copy()

    def safe_transform(le: LabelEncoder, series: pd.Series) -> pd.Series:
        classes_set = set(le.classes_)
        return series.apply(
            lambda v: le.transform([v])[0] if v in classes_set else -1
        )

    df[COL_CROP]  = safe_transform(encoders["crop"],  df[COL_CROP].astype(str).str.strip())
    df[COL_STATE] = safe_transform(encoders["state"], df[COL_STATE].astype(str).str.strip())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, filename: str) -> None:
    path = os.path.join(MODELS_DIR, filename)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path}")


def load_model(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def encode_single(value: str, encoder: LabelEncoder, label: str) -> int:
    """
    Encode a single string value.
    Raises ValueError with a descriptive message if the value is unknown.
    """
    value = str(value).strip()
    if value not in set(encoder.classes_):
        known = sorted(encoder.classes_)[:10]
        raise ValueError(
            f"Unknown {label}: '{value}'. "
            f"Sample known values: {known}... "
            f"(total {len(encoder.classes_)} known)"
        )
    return int(encoder.transform([value])[0])


def compute_confidence(clf_model, feature_row: np.ndarray) -> float:
    """
    Return the maximum class probability from the classifier as a
    confidence score (0–100 %).
    """
    proba = clf_model.predict_proba(feature_row)   # shape (1, n_classes)
    return float(np.max(proba) * 100)
