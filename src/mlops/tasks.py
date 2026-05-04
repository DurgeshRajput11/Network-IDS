from __future__ import annotations

import os
import subprocess
from typing import Any, List
from celery import Celery

from src.mlops.drift_detector import DriftDetector

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

detector = DriftDetector(artifacts_dir=os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))

@celery_app.task(name="check_data_drift")
def check_data_drift(features_batch: List[List[float]]) -> dict:
    """Background task to run KS-Test on a batch of inference traffic."""
    print(f"Running drift detection on a batch of {len(features_batch)} samples...")
    result = detector.detect_drift(features_batch)
    if result.get("drift_detected"):
        print(f"[ALERT] Data Drift Detected! Ratio: {result.get('drift_ratio', 0):.2f}")
    else:
        print("No significant data drift detected.")
    return result

@celery_app.task(name="trigger_retraining_pipeline")
def trigger_retraining_pipeline() -> dict:
    """
    Executes the entire training pipeline as a shell command.
    This ensures it runs in a clean, isolated process.
    """
    scripts = ["processing.py", "autoencoder.py", "train_xgboost.py"]
    training_dir = "src/training"
    results = {}

    for script in scripts:
        command = f"python {os.path.join(training_dir, script)}"
        print(f"Executing retraining step: {command}")
        
        try:
            process = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            results[script] = {"status": "success", "output": process.stdout}
            print(f"Successfully executed {script}.")
        except subprocess.CalledProcessError as e:
            error_message = f"Failed to execute {script}. Error: {e.stderr}"
            print(error_message)
            results[script] = {"status": "failed", "error": e.stderr}
            return {"status": "FAILED", "details": results}

    print("Retraining pipeline completed successfully.")
    return {"status": "SUCCESS", "details": results}
