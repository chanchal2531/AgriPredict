"""
app.py — Smart Crop Prediction & Risk Analysis Backend
======================================================
Flask REST API exposing:
  GET  /           → health check
  POST /predict    → full prediction pipeline

Expected JSON body for /predict:
  {
    "crop":  "Rice",
    "state": "Maharashtra",
    "area":  100,
    "year":  2015
  }

Response:
  {
    "predicted_production": float,
    "predicted_yield":      float,
    "risk_category":        "Low" | "Medium" | "High",
    "confidence_score":     float,          ← 0–100 %
    "model_details": {
      "regression_model":     "RandomForestRegressor",
      "classification_model": "RandomForestClassifier"
    }
  }
"""

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Local imports ─────────────────────────────────────────────────────────────
from utils import (
    load_encoders,
    load_model,
    encode_single,
    compute_confidence,
    logger,
    COL_CROP, COL_STATE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Initialise Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow browser frontend on any origin

# ── Load models & encoders once at startup ────────────────────────────────────
logger.info("Loading models and encoders …")
encoders  = load_encoders()                     # {"crop": LabelEncoder, "state": LabelEncoder}
reg_model = load_model("reg_model.pkl")         # RandomForestRegressor
clf_model = load_model("clf_model.pkl")         # RandomForestClassifier
logger.info("Models and encoders loaded successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: validate & parse request payload
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = {"crop", "state", "area", "year"}


def _parse_and_validate(payload: dict) -> tuple:
    """
    Validate the incoming JSON payload.
    Returns (crop, state, area, year) on success.
    Raises ValueError with a descriptive message on failure.
    """
    # ── Check required fields ─────────────────────────────────────────────────
    missing = REQUIRED_FIELDS - set(payload.keys())
    if missing:
        raise ValueError(f"Missing required field(s): {sorted(missing)}")

    crop  = payload["crop"]
    state = payload["state"]
    area  = payload["area"]
    year  = payload["year"]

    # ── Type checks ───────────────────────────────────────────────────────────
    if not isinstance(crop, str) or not crop.strip():
        raise ValueError("'crop' must be a non-empty string.")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("'state' must be a non-empty string.")
    try:
        area = float(area)
    except (TypeError, ValueError):
        raise ValueError("'area' must be a numeric value.")
    try:
        year = int(year)
    except (TypeError, ValueError):
        raise ValueError("'year' must be an integer.")

    if area <= 0:
        raise ValueError("'area' must be greater than 0.")
    if year < 1900 or year > 2100:
        raise ValueError("'year' must be a valid 4-digit year between 1900 and 2100.")

    return crop.strip(), state.strip(), area, year


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "message": "Backend is running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Full crop prediction pipeline.

    Steps:
      1. Parse & validate JSON input
      2. Encode crop & state using saved LabelEncoders
      3. Run RandomForestRegressor → predicted_production
      4. Compute predicted_yield = predicted_production / area
      5. Run RandomForestClassifier → risk_category
      6. Compute confidence_score from classifier probabilities
      7. Return structured JSON response
    """

    # ── 1. Parse request body ─────────────────────────────────────────────────
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON with "
                                 "Content-Type: application/json"}), 400

    try:
        crop, state, area, year = _parse_and_validate(payload)
    except ValueError as exc:
        logger.warning(f"Validation error: {exc}")
        return jsonify({"error": str(exc)}), 422

    # ── 2. Encode categoricals ─────────────────────────────────────────────────
    try:
        crop_enc  = encode_single(crop,  encoders["crop"],  "crop")
        state_enc = encode_single(state, encoders["state"], "state")
    except ValueError as exc:
        logger.warning(f"Encoding error: {exc}")
        return jsonify({"error": str(exc)}), 422

    # ── 3. Build feature vector [Crop, State_Name, Area, Crop_Year] ───────────
    features = np.array([[crop_enc, state_enc, area, year]], dtype=np.float64)

    # ── 4. Predict production ──────────────────────────────────────────────────
    predicted_production = float(reg_model.predict(features)[0])
    predicted_production = max(predicted_production, 0.0)   # clamp negatives

    # ── 5. Compute yield ───────────────────────────────────────────────────────
    predicted_yield = predicted_production / area   # area > 0 guaranteed by validation

    # ── 6. Predict risk category ───────────────────────────────────────────────
    risk_category = str(clf_model.predict(features)[0])

    # ── 7. Confidence score (max class probability) ────────────────────────────
    confidence_score = compute_confidence(clf_model, features)

    # ── 8. Build & return response ─────────────────────────────────────────────
    response = {
        "predicted_production": round(predicted_production, 4),
        "predicted_yield":      round(predicted_yield,      4),
        "risk_category":        risk_category,
        "confidence_score":     round(confidence_score,     2),
        "model_details": {
            "regression_model":     "RandomForestRegressor",
            "classification_model": "RandomForestClassifier",
        },
    }
    logger.info(
        f"Prediction — crop={crop}, state={state}, area={area}, year={year} "
        f"→ production={predicted_production:.2f}, yield={predicted_yield:.4f}, "
        f"risk={risk_category}, confidence={confidence_score:.1f}%"
    )
    return jsonify(response), 200


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found. Available: GET /, POST /predict"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "HTTP method not allowed on this endpoint."}), 405


@app.errorhandler(500)
def internal_error(exc):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error. Check server logs."}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=False in production; use gunicorn or similar WSGI server
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
