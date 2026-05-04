from __future__ import annotations

import os
import json
import redis
import subprocess
from celery import Celery

from src.mlops.drift_detector import DriftDetector

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

detector = DriftDetector(artifacts_dir=os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))

@celery_app.task(name="check_data_drift")
def check_data_drift(features_batch: list[list[float]]) -> dict:
    """Runs KS-Test and stores the result in Redis for monitoring."""
    print(f"Running drift detection on a batch of {len(features_batch)} samples...")
    result = detector.detect_drift(features_batch)
    redis_client.set("drift_status", json.dumps(result))
    return result

@celery_app.task(name="trigger_retraining_pipeline")
def trigger_retraining_pipeline() -> dict:
    """Executes the entire training pipeline."""
    redis_client.set("prediction_count", "0")
    redis_client.set("drift_status", json.dumps({"status": "retraining_in_progress"}))
    
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
    return {"status": "SUCCESS"}


SIMULATION_PID_KEY = "simulation_pid"

@celery_app.task(name="start_traffic_simulation")
def start_traffic_simulation():
    """Starts both the traffic producer and consumer."""
    producer_cmd = "python src/simulation/simulate_traffic.py"
    producer_proc = subprocess.Popen(producer_cmd, shell=True)
    redis_client.set(f"{SIMULATION_PID_KEY}_producer", producer_proc.pid)
    
    consumer_cmd = "python src/simulation/kafka_consumer.py"
    consumer_proc = subprocess.Popen(consumer_cmd, shell=True)
    redis_client.set(f"{SIMULATION_PID_KEY}_consumer", consumer_proc.pid)
    
    return {"producer_pid": producer_proc.pid, "consumer_pid": consumer_proc.pid}

@celery_app.task(name="stop_traffic_simulation")
def stop_traffic_simulation():
    """Stops the simulation processes using their stored PIDs."""
    for proc_type in ["producer", "consumer"]:
        pid = redis_client.get(f"{SIMULATION_PID_KEY}_{proc_type}")
        if pid:
            try:
                os.kill(int(pid), 9) 
                print(f"Stopped {proc_type} process with PID {pid}")
            except (OSError, ValueError) as e:
                print(f"Could not stop {proc_type} process with PID {pid}: {e}")
    return {"status": "stopped"}
