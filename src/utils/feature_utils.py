from __future__ import annotations

from typing import Iterable


def to_float_features(values: Iterable[float]) -> list[float]:
    """Normalize incoming feature values into a strict float list."""
    return [float(v) for v in values]


def validate_feature_vector(features: list[float], expected_size: int | None = None) -> None:
    if not features:
        raise ValueError("features must not be empty")
    if expected_size is not None and len(features) != expected_size:
        raise ValueError(f"expected {expected_size} features, got {len(features)}")

