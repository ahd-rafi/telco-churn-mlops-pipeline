# Telco Churn MLOps Pipeline

> End-to-end machine learning pipeline for predicting telecom customer churn — from data preparation and model training to a production REST API deployed with Docker and CI/CD.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | XGBoost (high-recall, class-imbalance aware) |
| Experiment Tracking | MLflow (parameters, metrics, model artifacts) |
| API Server | FastAPI (`POST /predict`) |
| Web UI | Gradio (mounted at `/ui`) |
| Data Validation | Great Expectations |
| Container | Docker (`python:3.11-slim`) |
| CI/CD | GitHub Actions → Docker Hub |
| Cloud Hosting | AWS ECS Fargate + Application Load Balancer |

---

## Project Structure

```
telco-churn-mlops-pipeline/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned dataset
├── notebooks/
│   └── EDA.ipynb               # Exploratory data analysis
├── scripts/
│   ├── run_pipeline.py         # Full training pipeline
│   ├── prepare_processed_data.py
│   ├── test_pipeline_phase1_data_features.py
│   ├── test_pipeline_phase2_modeling.py
│   └── test_fastapi.py
├── src/
│   ├── app/main.py             # FastAPI + Gradio serving app
│   ├── data/                   # Data loading & preprocessing
│   ├── features/               # Feature engineering
│   ├── serving/                # Inference logic & model artifacts
│   └── utils/                  # Data validation
├── dockerfile
├── requirements.txt
└── README.md
```

---

## What This Project Does

1. **Trains an XGBoost model** on the Telco Customer Churn dataset with optimized hyperparameters and class-imbalance handling
2. **Tracks experiments** with MLflow — every run logs metrics (precision, recall, F1, ROC-AUC), parameters, and serialized model artifacts
3. **Validates data quality** with Great Expectations before training
4. **Serves predictions** via a FastAPI REST API and a Gradio web interface
5. **Containerizes everything** in Docker for consistent deployment
6. **Automates builds** with GitHub Actions — every push to `main` builds and pushes to Docker Hub
7. **Deploys to AWS** using ECS Fargate behind an Application Load Balancer

---

## Quick Start

### Pull and run from Docker Hub

```bash
docker pull rafi44/telco-churn-mlops-pipeline:latest
docker run -p 8000:8000 rafi44/telco-churn-mlops-pipeline:latest
```

### Build locally

```bash
docker build -t telco-churn-mlops-pipeline -f dockerfile .
docker run -p 8000:8000 telco-churn-mlops-pipeline
```

### Access the app

| URL | Description |
|---|---|
| `http://localhost:8000/` | Health check → `{"status": "ok"}` |
| `http://localhost:8000/predict` | Prediction API (POST) |
| `http://localhost:8000/ui` | Gradio web interface |

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1, "MonthlyCharges": 85.0, "TotalCharges": 85.0
  }'
```

---

## Training Pipeline

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

**Model config:** `n_estimators=301`, `learning_rate=0.034`, `max_depth=7`, dynamic `scale_pos_weight`

**Tracked metrics:** precision, recall, F1, ROC-AUC, training time, prediction time, data quality pass

---

## CI/CD Pipeline

```
Push to main → GitHub Actions → Docker Build → Push to rafi44/telco-churn-mlops-pipeline:latest
```

The workflow (`.github/workflows/ci.yml`) automatically:
1. Checks out the code
2. Logs into Docker Hub using repository secrets
3. Builds and pushes the Docker image

**Required GitHub Secrets:** `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (set in repo Settings → Secrets → Actions)

---

## Deployment Architecture

```
Internet → ALB (port 80) → ECS Fargate Task (port 8000) → FastAPI + Gradio
```

| Security Group | Rule |
|---|---|
| ALB | Inbound port 80 from `0.0.0.0/0` |
| ECS Task | Inbound port 8000 from ALB SG only |

---

## Challenges Solved

| Problem | Root Cause | Fix |
|---|---|---|
| Unhealthy ALB targets | Missing health-check endpoint | Added `GET /` returning `{"status": "ok"}` |
| `ModuleNotFoundError: serving` | `PYTHONPATH` missing `src/` | Set `PYTHONPATH=/app/src` in Dockerfile |
| ALB DNS timeout | Security group misconfiguration | ALB SG → port 80 open; Task SG → port 8000 from ALB SG |
| ECS not picking up new image | Stale task definition | Force new deployment after image push |
| Gradio "No runs found" | MLflow experiment name mismatch | Standardized experiment name across training and serving |

---

## Docker Hub

```bash
docker pull rafi44/telco-churn-mlops-pipeline:latest
```

[View on Docker Hub →](https://hub.docker.com/r/rafi44/telco-churn-mlops-pipeline)