# InsureAI — Intelligent Insurance Bundle Recommendation System

> **Hackathon 2026 · Data Overflow**  
> AI-powered ML product for insurance brokers. Get real-time bundle predictions for customer profiles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         InsureAI                                │
├────────────────────────┬────────────────────────────────────────┤
│     Frontend (HTML)    │         Backend (FastAPI)              │
│                        │                                        │
│  Customer Form Input   │  POST /predict                         │
│  ─────────────────►    │   └─ Pydantic validation               │
│                        │   └─ PreprocessorService               │
│  Results Display       │       └─ Feature engineering           │
│  ◄─────────────────    │       └─ OHE / Target encoding         │
│                        │   └─ InferenceService                  │
│  Confidence Bars       │       └─ XGBoost.predict_proba()       │
│  Top-3 Bundles         │       └─ LRU cache (FIFO 512)          │
│  Key Factors           │   └─ PredictionResponse (JSON)         │
│                        │                                        │
│                        │  GET /health                           │
│                        │   └─ Model readiness check             │
└────────────────────────┴────────────────────────────────────────┘
                                    │
                             model.joblib
                      (XGBoost artifact, ~15 MB)
                      feature_names │ label_encoder
                      region_map    │ medians
                      cat_cols      │ num_classes (10)
```

---

## Repository Structure

```
Website/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py        # GET /health
│   │   │   └── predict.py       # POST /predict
│   │   └── router.py            # Route aggregator
│   ├── core/
│   │   ├── config.py            # Pydantic Settings
│   │   └── logging_config.py    # Structured logging
│   ├── models/
│   │   └── loader.py            # Model singleton loader
│   ├── schemas/
│   │   └── customer.py          # Request/response schemas
│   ├── services/
│   │   ├── inference.py         # Inference service + cache
│   │   └── preprocessor.py      # Feature engineering
│   ├── main.py                  # FastAPI app entry point
│   └── requirements.txt
├── frontend/
│   ├── index.html               # SPA with customer form + results
│   ├── style.css                # Dark glassmorphism theme
│   └── app.js                   # API client + result rendering
├── ml/
│   └── retrain.py               # Retraining script skeleton
├── tests/
│   └── test_api.py              # pytest integration tests
├── scripts/
│   ├── run_api.bat              # Windows startup
│   └── run_api.sh               # Linux/macOS startup
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI
├── Dockerfile                   # Multi-stage production image
├── docker-compose.yml           # API + frontend services
├── .env.example                 # Environment variables template
└── README.md
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- `model.joblib` in the root `Data Overflow/` directory

### 1. Install dependencies

```bash
cd "Data Overflow/Website"
pip install -r backend/requirements.txt
```

### 2. Start the API

```bash
# Windows
scripts\run_api.bat

# Linux / macOS
bash scripts/run_api.sh

# Or directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the frontend

Open `frontend/index.html` in your browser (double-click or use Live Server).

The frontend automatically connects to `http://localhost:8000`.

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0",
  "num_classes": 10,
  "api_version": "1.0.0"
}
```

---

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "estimated_annual_income": 75000,
    "adult_dependents": 1,
    "child_dependents": 2,
    "infant_dependents": 0,
    "previous_policy_duration_months": 24,
    "grace_period_extensions": 1,
    "years_without_claims": 3,
    "policy_amendments_count": 2,
    "vehicles_on_policy": 1,
    "custom_riders_requested": 1,
    "days_since_quote": 10,
    "policy_start_month": "March",
    "policy_start_year": 2024,
    "policy_start_week": 12,
    "broker_agency_type": "National_Corporate",
    "acquisition_channel": "Direct_Website",
    "payment_schedule": "Monthly",
    "employment_status": "Employed_FullTime",
    "region_code": "US-CA",
    "deductible_tier": "Tier_2_Mid_Ded"
  }'
```

**Response:**

```json
{
  "predicted_bundle": 2,
  "predicted_bundle_name": "Standard Plan",
  "confidence": 0.7834,
  "top_3": [
    {"bundle_id": 2, "bundle_name": "Standard Plan", "confidence": 0.7834},
    {"bundle_id": 4, "bundle_name": "Premium Plan",  "confidence": 0.1021},
    {"bundle_id": 3, "bundle_name": "Enhanced Plan", "confidence": 0.0612}
  ],
  "key_factors": [
    "Large family (3 dependents) suggests comprehensive coverage need",
    "Broker-assisted purchase — agent-recommended bundle weighted more"
  ],
  "model_version": "1.0.0"
}
```

**Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Bundle Labels

| ID | Name               |
|----|--------------------|
| 0  | Basic Plan         |
| 1  | Essential Plan     |
| 2  | Standard Plan      |
| 3  | Enhanced Plan      |
| 4  | Premium Plan       |
| 5  | Elite Plan         |
| 6  | Family Plan        |
| 7  | Comprehensive Plan |
| 8  | Enterprise Plan    |
| 9  | Ultimate Plan      |

---

## Running Tests

```bash
cd "Data Overflow/Website"
pytest tests/ -v
```

Tests cover: health check, valid prediction, missing optional fields, validation errors, caching.

---

## Docker Deployment

```bash
cd "Data Overflow/Website"

# Build and start
docker-compose up --build

# API available at: http://localhost:8000
# Frontend at:      http://localhost:3000
```

Or build just the API:

```bash
docker build -t insureai-api .
docker run -p 8000:8000 -v "$(pwd)/../model.joblib:/app/model.joblib:ro" insureai-api
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable         | Default          | Description                     |
|------------------|------------------|---------------------------------|
| `MODEL_PATH`     | `../model.joblib`| Path to trained model artifact  |
| `API_HOST`       | `0.0.0.0`        | Bind address                    |
| `API_PORT`       | `8000`           | Port                            |
| `LOG_LEVEL`      | `INFO`           | Logging level                   |
| `CACHE_ENABLED`  | `true`           | Enable prediction caching       |
| `CACHE_MAX_SIZE` | `512`            | Max cached predictions          |

---

## Retraining

```bash
python ml/retrain.py --train path/to/train.csv --output model.joblib --seed 42
```

The script performs the full pipeline: feature engineering → imputation → OHE → region target encoding → XGBoost training with early stopping → artifact save.

---

## MLOps Notes

| Feature                | Implementation                                   |
|------------------------|--------------------------------------------------|
| **Config-driven**      | Pydantic `BaseSettings` + `.env` file            |
| **Model versioning**   | Artifact dict with version field                 |
| **In-memory caching**  | FIFO cache on prediction hash (512 entries)      |
| **Inference logging**  | Latency + bundle + confidence logged per request |
| **Retraining script**  | `ml/retrain.py` with argparse                    |
| **CI pipeline**        | GitHub Actions: lint (ruff) + pytest             |
| **Dockerized**         | Multi-stage image, docker-compose                |

---

## Future Work

- [ ] Model registry integration (MLflow)
- [ ] SHAP-based feature importance in responses
- [ ] A/B testing support (multi-model serving)
- [ ] Redis-backed distributed cache
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] Async inference for batch requests
- [ ] Frontend authentication for brokers
