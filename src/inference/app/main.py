from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List

from src.inference.app.model_loader import HybridModelService, PredictionOutput
from src.mlops.tasks import check_data_drift, trigger_retraining_pipeline

API_TITLE = "Hybrid IDS Inference & MLOps API"
API_VERSION = "1.0.0"
ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACT_DIR", "artifacts")
DRIFT_BATCH_SIZE = int(os.getenv("DRIFT_BATCH_SIZE", "500"))

app = FastAPI(title=API_TITLE, version=API_VERSION)
model_service = HybridModelService(artifacts_dir=ARTIFACTS_DIR)

inference_buffer: List[List[float]] = []

class PredictRequest(BaseModel):
    features: List[float] = Field(..., example=[0.1] * 20, description="A list of 20 feature values.")

class PredictResponse(BaseModel):
    attack: int = Field(..., example=1, description="Prediction result: 1 for Attack, 0 for Benign.")
    confidence: float = Field(..., example=0.98, description="Model confidence score.")

class RetrainResponse(BaseModel):
    message: str
    task_id: str

class StatusResponse(BaseModel):
    model_status: str
    artifacts_loaded: List[str]


@app.get("/health", summary="Health Check", tags=["Monitoring"])
def health_check() -> dict:
    """Returns a 200 OK status if the API is running."""
    return {"status": "ok"}

@app.get("/model-status", response_model=StatusResponse, summary="Get Model Status", tags=["Monitoring"])
def get_model_status() -> StatusResponse:
    """Provides the status of the loaded model and its artifacts."""
    loaded = [name for name, artifact in model_service.__dict__.items() if artifact is not None and name != "dummy_mode"]
    status = "dummy_mode" if model_service.dummy_mode else "production_ready"
    return StatusResponse(model_status=status, artifacts_loaded=loaded)

@app.post("/predict", response_model=PredictResponse, summary="Run Hybrid Prediction", tags=["Inference"])
def predict(payload: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    """
    Performs inference using the hybrid model and queues data for drift detection.
    """
    global inference_buffer
    try:
        result: PredictionOutput = model_service.predict(payload.features)
        
        inference_buffer.append(payload.features)
        
        if len(inference_buffer) >= DRIFT_BATCH_SIZE:
            background_tasks.add_task(check_data_drift.delay, inference_buffer.copy())
            inference_buffer.clear()

        return PredictResponse(attack=result.attack, confidence=result.confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}") from e

@app.post("/retrain", response_model=RetrainResponse, summary="Trigger Model Retraining", tags=["MLOps"])
def trigger_retraining() -> RetrainResponse:
    """
    Triggers the full model retraining pipeline as a background task.
    This API returns immediately with a task ID.
    """
    try:
        task = trigger_retraining_pipeline.delay()
        return RetrainResponse(
            message="Model retraining pipeline triggered successfully.",
            task_id=task.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining task: {str(e)}")
