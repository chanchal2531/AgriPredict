# 🌾 Smart Crop Prediction & Risk Analysis Backend System

A production-quality machine learning backend that predicts crop production, yield, and risk levels using Random Forest models, exposed via a Flask REST API.

---

## 📁 Project Structure

```
smart_crop_backend/
│
├── data/
│   ├── crop_data.csv               ← Pre-cleaned dataset (CSV, fast-load)
│   ├── crop_production_raw.xlsx    ← Original raw Excel dataset
│   └── crop_production_cleaned.xlsx ← Cleaned Excel (source)
│
├── models/
│   ├── reg_model.pkl               ← Trained RandomForestRegressor
│   ├── clf_model.pkl               ← Trained RandomForestClassifier
│   └── encoders.pkl                ← Fitted LabelEncoders (crop + state)
│
├── utils.py                        ← Shared preprocessing & helper functions
├── train_model.py                  ← End-to-end ML training pipeline
├── app.py                          ← Flask REST API
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Engineer features (Yield, Risk Category)
- Train `RandomForestRegressor` for production prediction
- Train `RandomForestClassifier` for risk classification
- Save all models and encoders to `models/`

### 3. Start the API
```bash
python app.py
```
API runs at `http://localhost:5000`

For production deployment:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 📡 API Endpoints

### `GET /`
Health check.

**Response:**
```json
{"status": "ok", "message": "Backend is running"}
```

---

### `POST /predict`
Predict crop production, yield, and risk level.

**Request Body:**
```json
{
  "crop":  "Rice",
  "state": "Maharashtra",
  "area":  100,
  "year":  2015
}
```

**Response:**
```json
{
  "predicted_production": 320.3246,
  "predicted_yield":      3.2032,
  "risk_category":        "Medium",
  "confidence_score":     36.86,
  "model_details": {
    "regression_model":     "RandomForestRegressor",
    "classification_model": "RandomForestClassifier"
  }
}
```

**Error Responses:**
| Code | Meaning |
|------|---------|
| 400  | Invalid JSON body |
| 422  | Validation error (missing fields, unknown crop/state, bad types) |
| 500  | Internal server error |

---

## 🧠 ML Pipeline

### Feature Engineering
- **Yield** = `Production / Area` (division-by-zero safe)
- **Risk Category**: percentile-based binning
  - `Low`    → yield < 25th percentile
  - `Medium` → 25th – 75th percentile
  - `High`   → yield > 75th percentile

### Models
| Task           | Model                    | Features                              | Target     |
|----------------|--------------------------|---------------------------------------|------------|
| Regression     | RandomForestRegressor    | Crop, State_Name, Area, Crop_Year     | Production |
| Classification | RandomForestClassifier   | Crop, State_Name, Area, Crop_Year     | Risk_Category |

### Encoding
- `Crop` and `State_Name` → `LabelEncoder` (saved to `models/encoders.pkl`)

---

## 📊 Dataset
- **Source**: Indian crop production data (1997–2015)
- **Records**: 246,091
- **Features used**: State_Name, Crop_Year, Crop, Area, Production
- **Known crops**: 124 | **Known states**: 33

---

## 🛡️ Error Handling
- Missing JSON fields → 422 with field name
- Unknown crop or state → 422 with list of valid options
- Invalid numeric types → 422 with field description
- Area ≤ 0 → 422 validation error
- Year out of range → 422 validation error
- All 500 errors logged server-side

---

## 🔧 Configuration (utils.py)
Key constants you can adjust:
```python
RISK_LOW_THRESH  = 25   # percentile for Low/Medium boundary
RISK_HIGH_THRESH = 75   # percentile for Medium/High boundary
RANDOM_SEED      = 42   # for reproducibility
```

Model hyperparameters in `train_model.py`:
```python
n_estimators=150, max_depth=20,
min_samples_split=5, min_samples_leaf=2
```
