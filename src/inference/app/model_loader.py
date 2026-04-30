from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import joblib
except Exception:
    joblib = None

try:
    import torch
except Exception:
    torch = None

try:
    from xgboost import XGBClassifier
except Exception: 
    XGBClassifier = None

@dataclass
class PredictionOutput:
    attack: int
    confidence: float

class HybridModelService:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        self.scaler: Any | None = None
        self.xgb_model: Any | None = None
        self.autoencoder: Any | None = None
        self.threshold: float | None = None
        self.benign_idx: int = 0
        self.dummy_mode = True
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        scaler_path = os.path.join(self.artifacts_dir, "scaler.joblib")
        xgb_path = os.path.join(self.artifacts_dir, "xgboost_model.json")
        ae_path = os.path.join(self.artifacts_dir, "autoencoder.pt")
        thresh_path = os.path.join(self.artifacts_dir, "ae_threshold.json")
        labels_path = os.path.join(self.artifacts_dir, "label_map.json")

        if joblib and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if XGBClassifier and os.path.exists(xgb_path):
            self.xgb_model = XGBClassifier()
            self.xgb_model.load_model(xgb_path)

        if torch and os.path.exists(ae_path):
            self.autoencoder = torch.jit.load(ae_path, map_location="cpu")
            self.autoencoder.eval()

        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                self.threshold = float(json.load(f).get("tau", 0.0))

        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                label_map = json.load(f)
                self.benign_idx = label_map.get("Benign", 0)

        self.dummy_mode = self.xgb_model is None or self.scaler is None

    def predict(self, features: list[float]) -> PredictionOutput:
        sample = np.asarray(features, dtype=np.float32).reshape(1, -1)

        if self.dummy_mode:
            base_score = float(np.clip(np.mean(np.abs(sample)), 0.0, 1.0))
            attack = 1 if base_score >= 0.55 else 0
            return PredictionOutput(attack=attack, confidence=round(max(base_score, 1.0 - base_score), 4))

        x_input = self.scaler.transform(sample)

        ae_error = 0.0
        is_extreme_anomaly = False
        if self.autoencoder is not None and self.threshold is not None:
            with torch.no_grad():
                x_tensor = torch.tensor(x_input, dtype=torch.float32)
                reconstruction = self.autoencoder(x_tensor)
                ae_error = float(((x_tensor - reconstruction) ** 2).mean().item())
                is_extreme_anomaly = ae_error > self.threshold

        xgb_pred = int(self.xgb_model.predict(x_input)[0])
        xgb_is_attack = (xgb_pred != self.benign_idx)
        probas = self.xgb_model.predict_proba(x_input)[0]
        xgb_conf = float(probas[xgb_pred])

        if is_extreme_anomaly and not xgb_is_attack:
            return PredictionOutput(attack=1, confidence=min(ae_error / self.threshold, 1.0))
        
        return PredictionOutput(attack=1 if xgb_is_attack else 0, confidence=xgb_conf)