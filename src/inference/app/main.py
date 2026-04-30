from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model_loader import HybridModelService


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    attack: int
    confidence: float


app = FastAPI(title="Hybrid IDS Inference API", version="0.1.0")
model_service = HybridModelService(artifacts_dir=os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "dummy" if model_service.dummy_mode else "hybrid"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = model_service.predict(payload.features)
        return PredictResponse(attack=result.attack, confidence=result.confidence)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
