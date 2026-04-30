"""
train_model.py — Smart Crop Prediction & Risk Analysis Backend
==============================================================
End-to-end training pipeline:
  1. Load & preprocess data
  2. Feature engineering (Yield + Risk_Category)
  3. Encode categoricals, save encoders
  4. Train RandomForestRegressor  → predict Production
  5. Train RandomForestClassifier → predict Risk_Category
  6. Evaluate both models
  7. Persist models to models/
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, classification_report,
)

# ── Local imports ─────────────────────────────────────────────────────────────
from utils import (
    load_cleaned_data,
    preprocess,
    engineer_features,
    build_encoders,
    save_encoders,
    apply_encoders,
    save_model,
    logger,
    COL_CROP, COL_STATE, COL_YEAR, COL_AREA,
    COL_PRODUCTION, COL_YIELD, COL_RISK,
    MODELS_DIR,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Feature / target column lists ─────────────────────────────────────────────
REGRESSION_FEATURES      = [COL_CROP, COL_STATE, COL_AREA, COL_YEAR]
REGRESSION_TARGET        = COL_PRODUCTION
CLASSIFICATION_FEATURES  = [COL_CROP, COL_STATE, COL_AREA, COL_YEAR]
CLASSIFICATION_TARGET    = COL_RISK


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare() -> pd.DataFrame:
    logger.info("═" * 60)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("═" * 60)

    df = load_cleaned_data()
    df = preprocess(df)
    df = engineer_features(df)

    # Drop rows where Area = 0 (cannot compute meaningful yield/risk)
    before = len(df)
    df = df[df[COL_AREA] > 0].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with Area=0. Remaining: {len(df)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Encode & split
# ─────────────────────────────────────────────────────────────────────────────
def encode_and_split(df: pd.DataFrame):
    logger.info("═" * 60)
    logger.info("STEP 2: Encoding categoricals and splitting data")
    logger.info("═" * 60)

    # Build and persist encoders
    encoders = build_encoders(df)
    save_encoders(encoders)

    # Apply encoders (in-place on a copy)
    df_enc = apply_encoders(df, encoders)

    # ── Regression split ──────────────────────────────────────────────────────
    X_reg = df_enc[REGRESSION_FEATURES].values
    y_reg = df_enc[REGRESSION_TARGET].values

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info(f"Regression  → train={len(X_reg_train):,}  test={len(X_reg_test):,}")

    # ── Classification split ──────────────────────────────────────────────────
    X_clf = df_enc[CLASSIFICATION_FEATURES].values
    y_clf = df_enc[CLASSIFICATION_TARGET].values

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info(f"Classification → train={len(X_clf_train):,}  test={len(X_clf_test):,}")

    return (
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
        X_clf_train, X_clf_test, y_clf_train, y_clf_test,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train Regression Model
# ─────────────────────────────────────────────────────────────────────────────
def train_regression(X_train, X_test, y_train, y_test) -> RandomForestRegressor:
    logger.info("═" * 60)
    logger.info("STEP 3: Training RandomForestRegressor (Production prediction)")
    logger.info("═" * 60)

    reg_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    reg_model.fit(X_train, y_train)
    logger.info("Regression model training complete.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = reg_model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info("── Regression Evaluation ──")
    logger.info(f"  R² Score             : {r2:.4f}")
    logger.info(f"  Mean Absolute Error  : {mae:,.2f}")

    print("\n" + "═" * 50)
    print("  REGRESSION MODEL EVALUATION")
    print("═" * 50)
    print(f"  R² Score            : {r2:.4f}")
    print(f"  Mean Absolute Error : {mae:,.2f}")
    print("═" * 50 + "\n")

    save_model(reg_model, "reg_model.pkl")
    return reg_model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train Classification Model
# ─────────────────────────────────────────────────────────────────────────────
def train_classification(X_train, X_test, y_train, y_test) -> RandomForestClassifier:
    logger.info("═" * 60)
    logger.info("STEP 4: Training RandomForestClassifier (Risk prediction)")
    logger.info("═" * 60)

    clf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",      # handles class imbalance
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf_model.fit(X_train, y_train)
    logger.info("Classification model training complete.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = clf_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
    report = classification_report(y_test, y_pred, labels=["Low", "Medium", "High"])

    logger.info(f"  Accuracy: {acc:.4f}")

    print("\n" + "═" * 50)
    print("  CLASSIFICATION MODEL EVALUATION")
    print("═" * 50)
    print(f"  Accuracy           : {acc:.4f}")
    print(f"\n  Confusion Matrix (Low / Medium / High):\n{cm}")
    print(f"\n  Classification Report:\n{report}")
    print("═" * 50 + "\n")

    save_model(clf_model, "clf_model.pkl")
    return clf_model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load & prepare
    df = load_and_prepare()

    # 2. Encode & split
    (
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
        X_clf_train, X_clf_test, y_clf_train, y_clf_test,
    ) = encode_and_split(df)

    # 3. Regression
    reg_model = train_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test)

    # 4. Classification
    clf_model = train_classification(X_clf_train, X_clf_test, y_clf_train, y_clf_test)

    logger.info("═" * 60)
    logger.info("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    logger.info(f"  Saved to: {MODELS_DIR}/")
    logger.info("    • reg_model.pkl")
    logger.info("    • clf_model.pkl")
    logger.info("    • encoders.pkl")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
