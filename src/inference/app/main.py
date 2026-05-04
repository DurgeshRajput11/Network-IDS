from __future__ import annotations

import os
import json
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List

from src.inference.app.model_loader import HybridModelService, PredictionOutput
from src.mlops.tasks import check_data_drift, trigger_retraining_pipeline, start_traffic_simulation, stop_traffic_simulation

API_TITLE = "IDS Demo System API"
API_VERSION = "2.0.0"
ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACT_DIR", "artifacts")
DRIFT_BATCH_SIZE = int(os.getenv("DRIFT_BATCH_SIZE", "500"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

app = FastAPI(title=API_TITLE, version=API_VERSION)
model_service = HybridModelService(artifacts_dir=ARTIFACTS_DIR)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

inference_buffer: List[List[float]] = []

class PredictRequest(BaseModel):
    features: List[float] = Field(..., example=[0.1] * 20)

class PredictResponse(BaseModel):
    attack: int
    confidence: float

class TaskResponse(BaseModel):
    message: str
    task_id: str

class StatusResponse(BaseModel):
    model_status: str
    simulation_status: str
    total_predictions: int
    last_drift_check: dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


@app.get("/health", summary="Health Check", tags=["Monitoring"])
def health_check():
    return {"status": "ok"}

@app.get("/status", response_model=StatusResponse, summary="Get System Status", tags=["Monitoring"])
def get_system_status():
    """Provides a snapshot of the system's current state."""
    drift_result = redis_client.get("drift_status")
    return {
        "model_status": "dummy_mode" if model_service.dummy_mode else "production_ready",
        "simulation_status": redis_client.get("simulation_status") or "stopped",
        "total_predictions": int(redis_client.get("prediction_count") or 0),
        "last_drift_check": json.loads(drift_result) if drift_result else {"status": "not_run"},
    }

@app.post("/predict", response_model=PredictResponse, summary="Run Hybrid Prediction", tags=["Inference"])
async def predict(payload: PredictRequest, background_tasks: BackgroundTasks):
    """Performs inference and broadcasts the result via WebSocket."""
    global inference_buffer
    try:
        result = model_service.predict(payload.features)
        
        await manager.broadcast(json.dumps(result.__dict__))
        
        redis_client.incr("prediction_count")

        inference_buffer.append(payload.features)
        if len(inference_buffer) >= DRIFT_BATCH_SIZE:
            background_tasks.add_task(check_data_drift.delay, inference_buffer.copy())
            inference_buffer.clear()

        return PredictResponse(attack=result.attack, confidence=result.confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/simulation/start", response_model=TaskResponse, summary="Start Traffic Simulation", tags=["Demo Control"])
def start_simulation():
    """Triggers the Kafka traffic simulation as a background task."""
    task = start_traffic_simulation.delay()
    redis_client.set("simulation_status", "running")
    return TaskResponse(message="Traffic simulation started.", task_id=task.id)

@app.post("/simulation/stop", response_model=TaskResponse, summary="Stop Traffic Simulation", tags=["Demo Control"])
def stop_simulation():
    """Stops the Kafka traffic simulation."""
    task = stop_traffic_simulation.delay()
    redis_client.set("simulation_status", "stopped")
    return TaskResponse(message="Traffic simulation stopping.", task_id=task.id)

@app.post("/retrain", response_model=TaskResponse, summary="Trigger Model Retraining", tags=["MLOps"])
def trigger_retraining():
    """Triggers the full model retraining pipeline as a background task."""
    task = trigger_retraining_pipeline.delay()
    return TaskResponse(message="Model retraining pipeline triggered.", task_id=task.id)

@app.websocket("/ws/predictions")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to stream real-time predictions."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)