from __future__ import annotations

import os
import numpy as np
import mlflow
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self, artifacts_dir: str = "artifacts", p_value_threshold: float = 0.05):
        self.artifacts_dir = artifacts_dir
        self.p_value_threshold = p_value_threshold
        self.baseline_features = self._load_baseline()

    def _load_baseline(self) -> np.ndarray | None:
        path = os.path.join(self.artifacts_dir, "data", "X_train.npy")
        if os.path.exists(path):
            baseline = np.load(path)
            if baseline.shape[0] > 10000:
                indices = np.random.choice(baseline.shape[0], 10000, replace=False)
                return baseline[indices]
            return baseline
        print(f"Warning: Baseline data not found at {path}")
        return None

    def detect_drift(self, current_data: list[list[float]]) -> dict[str, float | bool]:
        """Runs KS-Test feature-by-feature against baseline."""
        if self.baseline_features is None or not current_data:
            return {"drift_detected": False, "drift_ratio": 0.0, "drifted_features": 0}

        current_arr = np.array(current_data)
        num_features = current_arr.shape[1]
        drifted_features = 0

        for i in range(num_features):
            baseline_col = self.baseline_features[:, i]
            current_col = current_arr[:, i]
            
            stat, p_value = ks_2samp(baseline_col, current_col)
            
            if p_value < self.p_value_threshold:
                drifted_features += 1

        drift_ratio = drifted_features / num_features
        is_drifted = drift_ratio > 0.2

        try:
            mlflow.set_experiment("Network-IDS")
            with mlflow.start_run(run_name="data_drift_check"):
                mlflow.log_metric("drift_ratio", drift_ratio)
                mlflow.log_metric("drifted_features_count", drifted_features)
                mlflow.log_param("is_drifted", is_drifted)
        except Exception as e:
            print(f"Failed to log to MLflow: {e}")

        return {
            "drift_detected": is_drifted,
            "drift_ratio": drift_ratio,
            "drifted_features": drifted_features
        }

if __name__ == "__main__":
    detector = DriftDetector("../../artifacts")
    dummy_data = [list(np.random.rand(20)) for _ in range(100)]
    result = detector.detect_drift(dummy_data)
    print(f"Drift Result: {result}")