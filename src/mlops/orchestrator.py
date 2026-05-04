import time
import json
import requests

API_BASE_URL = "http://fastapi:8000" 
DRIFT_THRESHOLD = 0.2
CHECK_INTERVAL_SECONDS = 60 

def main():
    """
    The main loop for the MLOps orchestrator.
    Periodically checks for data drift and triggers retraining if necessary.
    """
    print("--- MLOps Orchestrator Started ---")
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/status")
            response.raise_for_status()
            status = response.json()
            
            print(f"Checking status... Current prediction count: {status.get('total_predictions')}")

            drift_info = status.get("last_drift_check", {})
            drift_ratio = drift_info.get("drift_ratio", 0.0)

            if drift_info.get("drift_detected") or drift_ratio > DRIFT_THRESHOLD:
                print(f"[ALERT] Drift detected! Ratio: {drift_ratio:.2f}. Triggering retraining...")
                
                retrain_response = requests.post(f"{API_BASE_URL}/retrain")
                retrain_response.raise_for_status()
                print(f"Retraining triggered successfully: {retrain_response.json()}")
                
                print("Waiting for retraining to complete...")
                time.sleep(600) 
            else:
                print("No significant drift detected.")

        except requests.exceptions.RequestException as e:
            print(f"Could not connect to API: {e}")
        
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()