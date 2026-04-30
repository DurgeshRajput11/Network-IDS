from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover - optional at runtime
    joblib = None

try:
    import torch
except Exception:  # pragma: no cover - optional at runtime
    torch = None


@dataclass
class PredictionOutput:
    attack: int
    confidence: float


class HybridModelService:
    """Loads model artifacts if available and falls back to deterministic dummy mode."""

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        self.scaler: Any | None = None
        self.xgb_model: Any | None = None
        self.autoencoder: Any | None = None
        self.threshold: float | None = None
        self.dummy_mode = True
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        scaler_path = self._first_existing("scaler.joblib", "scaler.pkl")
        xgb_path = self._first_existing("xgb_model.joblib", "xgb_model.pkl")
        threshold_path = self._first_existing("threshold.npy", "autoencoder_threshold.npy")
        ae_path = self._first_existing("autoencoder.pt")

        if joblib and scaler_path:
            self.scaler = joblib.load(scaler_path)
        if joblib and xgb_path:
            self.xgb_model = joblib.load(xgb_path)
        if threshold_path:
            self.threshold = float(np.load(threshold_path))
        if ae_path and torch:
            self.autoencoder = torch.load(ae_path, map_location="cpu")
            if hasattr(self.autoencoder, "eval"):
                self.autoencoder.eval()

        self.dummy_mode = self.xgb_model is None

    def _first_existing(self, *names: str) -> str | None:
        for name in names:
            path = os.path.join(self.artifacts_dir, name)
            if os.path.exists(path):
                return path
        return None

    def predict(self, features: list[float]) -> PredictionOutput:
        sample = np.asarray(features, dtype=np.float32).reshape(1, -1)

        if self.dummy_mode:
            # Stable dummy behavior for API contract and integration testing.
            base_score = float(np.clip(np.mean(np.abs(sample)), 0.0, 1.0))
            attack = 1 if base_score >= 0.55 else 0
            confidence = round(max(base_score, 1.0 - base_score), 4)
            return PredictionOutput(attack=attack, confidence=confidence)

        x_input = self.scaler.transform(sample) if self.scaler else sample
        xgb_pred = int(self.xgb_model.predict(x_input)[0])
        xgb_conf = float(xgb_pred)
        if hasattr(self.xgb_model, "predict_proba"):
            xgb_conf = float(self.xgb_model.predict_proba(x_input)[0][1])

        ae_error = 0.0
        if self.autoencoder is not None and torch is not None:
            with torch.no_grad():
                x_tensor = torch.tensor(x_input, dtype=torch.float32)
                reconstruction = self.autoencoder(x_tensor)
                ae_error = float(((x_tensor - reconstruction) ** 2).mean().item())

        ae_alert = bool(self.threshold is not None and ae_error > self.threshold)
        attack = int(xgb_pred == 1 or ae_alert)

        if self.threshold and self.threshold > 0:
            ae_conf = min(ae_error / self.threshold, 1.0)
            confidence = float(max(xgb_conf, ae_conf))
        else:
            confidence = xgb_conf

        return PredictionOutput(attack=attack, confidence=round(confidence, 4))
