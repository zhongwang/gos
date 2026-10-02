"""Production CPU decision weights and calibrated logit threshold."""
import json
from pathlib import Path

import numpy as np

from .features import MULTISCALE_SUMMARY_NAMES


class DecisionHead:
    def __init__(self, path: str | Path):
        saved = json.loads(Path(path).read_text())
        if saved.get("schema_version") != "gos-decision-head/1":
            raise ValueError("unsupported decision head schema")
        if tuple(saved["cpu_feature_names"]) != MULTISCALE_SUMMARY_NAMES:
            raise ValueError("CPU feature order does not match the decision head")
        # Only the used slices are shipped; preserve their original precision.
        self.coef = self._vector(saved["cpu_coef"])
        self.mean = self._vector(saved["cpu_mean"])
        self.scale = self._vector(saved["cpu_scale"])
        self.intercept = float(saved["intercept"])
        availability = saved["availability"]
        self.availability_coef = float(availability["coef"])
        self.availability_mean = float(availability["mean"])
        self.availability_scale = float(availability["scale"])
        self.calibration = saved["calibration"]
        self.threshold_logit = float(self.calibration["threshold_logit"])
        scalars = [self.intercept, self.availability_coef, self.availability_mean,
                   self.availability_scale, self.threshold_logit]
        if not np.isfinite(scalars).all() or self.availability_scale <= 0 or np.any(self.scale <= 0):
            raise ValueError("decision weights must be finite with positive scales")

    @staticmethod
    def _vector(values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (47,) or not np.isfinite(values).all():
            raise ValueError("expected 47 finite decision weights")
        return values

    def base_logits(self, cpu_features: np.ndarray) -> np.ndarray:
        if cpu_features.ndim != 2 or cpu_features.shape[1] != 47:
            raise ValueError("expected a (records, 47) CPU feature matrix")
        if not np.isfinite(cpu_features).all():
            raise ValueError("CPU features must be finite")
        availability = ((1.0 - self.availability_mean) / self.availability_scale) * self.availability_coef
        cpu = ((cpu_features.astype(np.float64) - self.mean) / self.scale) @ self.coef
        return self.intercept + availability + cpu


def sigmoid(logit: float) -> float:
    """Stable sigmoid, including large negative logits."""
    if logit >= 0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exp_logit = np.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))
