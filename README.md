# Hybrid Network Intrusion Detection System (IDS)

## Goal
Build an end-to-end ML-based Intrusion Detection System that:
- Detects known attacks using supervised learning (XGBoost)
- Detects unknown attacks using anomaly detection (Autoencoder)
- Works in a real-time streaming pipeline using Kafka + FastAPI

## High-Level Architecture

Traffic -> Producer -> Kafka -> Consumer -> FastAPI -> ML Model -> Output -> (Elasticsearch -> Kibana)

## Team Roles

### ML Engineer
Responsible for:
- Data preprocessing
- Training XGBoost (full dataset)
- Training Autoencoder (only benign data)
- Hybrid prediction logic
- Saving model artifacts

### Backend Engineer
Responsible for:
- FastAPI inference API
- Kafka producer and consumer
- Integration with ML model
- Optional dashboard (Elasticsearch + Kibana)
- Docker deployment

## ML Design

### 1. XGBoost (Supervised)
- Train on full dataset
- Labels:
  - 0 -> BENIGN
  - 1 -> ATTACK

### 2. Autoencoder (Unsupervised)
- Train only on BENIGN data
- Learns normal traffic patterns
- Detects anomalies using reconstruction error

### 3. Hybrid Logic

```python
final_prediction = (xgb_pred == 1) or (ae_error > threshold)
```

## Dataset

Primary dataset:
- CIC-IDS2017

## Project Structure

```text
ml-ids/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── xgb_trainer.py
│   │   └── autoencoder.py
│   ├── inference/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   └── model_loader.py
│   │   └── kafka_consumer.py
│   ├── producer/
│   │   └── simulate_traffic.py
│   ├── utils/
│   └── mlops/
├── docker/
├── deployments/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment Setup (uv)

Create environment:

```powershell
uv venv
```

Activate:

Windows:

```powershell
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```powershell
uv add pandas numpy scikit-learn xgboost torch fastapi uvicorn kafka-python mlflow
```

Sync environment:

```powershell
uv sync
```

## Git Workflow

Start working:

```powershell
git pull
uv sync
```

After changes:

```powershell
git add .
git commit -m "message"
git pull
git push
```

## Backend API (FastAPI)

Endpoint:
- `POST /predict`

Request:

```json
{
  "features": [0.12, 34.5, 1]
}
```

Response:

```json
{
  "attack": 1,
  "confidence": 0.92
}
```

## Development Phases

### Phase 1 - ML (Model Ready)
- Preprocessing
- Train XGBoost
- Train Autoencoder
- Save artifacts:
  - `scaler.joblib`
  - `xgb_model.joblib`
  - `autoencoder.pt`
  - `threshold.npy`

### Phase 2 - Backend (No Model Required)
- Build FastAPI with dummy response
- Fix API contract
- Test endpoint

### Phase 3 - Kafka Integration
- Producer sends traffic
- Consumer reads from Kafka
- Consumer calls FastAPI

### Phase 4 - Replace Dummy with Real Model
- Load saved artifacts
- Call model inside API

### Phase 5 - Monitoring (Optional)
- Store results in Elasticsearch
- Visualize in Kibana

## Important Rules

- Autoencoder must be trained only on benign data
- Do not change API contract after defining it
- Always run `git pull -> uv sync`
- Do not push `.venv` to GitHub

## Common Mistakes

- Training autoencoder on full dataset
- Skipping `uv sync`
- Not pulling before push
- No API contract


## End Goal

A working system where:
- Kafka -> FastAPI -> ML Model -> Prediction -> Dashboard
- XGBoost detects known attacks
- Autoencoder detects anomalies
- Hybrid logic improves detection

## Backend Quick Start (Implemented)

Run API (dummy inference):

```powershell
uv run uvicorn src.inference.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run Kafka producer:

```powershell
uv run python src\producer\simulate_traffic.py
```

Run Kafka consumer:

```powershell
uv run python src\inference\kafka_consumer.py
```

## Docker and Kubernetes Starters (Implemented)

Added files:
- `docker/docker-compose.yml`
- `docker/k8s/namespace.yaml`
- `docker/k8s/kustomization.yaml`
- `deployments/fastapi-deployment.yaml`
- `deployments/mlflow-deployment.yaml`
- `deployments/kafka-deployment.yaml`

Run with Docker Compose:

```powershell
docker compose -f docker\docker-compose.yml up -d
```

Start only monitoring stack:

```powershell
docker compose -f docker\docker-compose.yml up -d elasticsearch kibana
```

Apply Kubernetes manifests:

```powershell
kubectl apply -k docker\k8s
```

For Kubernetes FastAPI deployment, replace:
- `your-dockerhub-user/ids-fastapi:latest`
with your built/pushed app image.

## Elasticsearch + Kibana Monitoring

Enable Elasticsearch indexing in consumer:

```powershell
$env:ENABLE_ELASTICSEARCH="true"
$env:ELASTICSEARCH_URL="http://127.0.0.1:9200"
$env:ELASTICSEARCH_INDEX_PREFIX="ids-predictions"
uv run python src\inference\kafka_consumer.py
```

This writes one enriched document per prediction into:
- `ids-predictions-*`

Bootstrap Kibana data view + dashboard:

```powershell
uv run python src\inference\setup_kibana.py
```

Then open Kibana:
- `http://127.0.0.1:5601`

Dashboard created:
- **Hybrid IDS Monitoring Dashboard**
  - attack count over time
  - attack vs benign split
  - confidence distribution
  - top suspicious sources
