from __future__ import annotations

import os
from typing import Any

try:
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover
    Celery = None  # type: ignore[assignment]

from src.inference.app.model_loader import HybridModelService
from src.utils.feature_utils import to_float_features, validate_feature_vector


def create_celery_app() -> Any:
    if Celery is None:
        raise RuntimeError("Celery is not installed. Install with: uv add celery")
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    return Celery("ids_tasks", broker=broker_url, backend=backend_url)


celery_app = create_celery_app()
model_service = HybridModelService(artifacts_dir=os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))


@celery_app.task(name="ids.predict")
def predict_task(features: list[float], expected_size: int | None = None) -> dict[str, float | int]:
    normalized = to_float_features(features)
    validate_feature_vector(normalized, expected_size=expected_size)
    prediction = model_service.predict(normalized)
    return {"attack": prediction.attack, "confidence": prediction.confidence}

