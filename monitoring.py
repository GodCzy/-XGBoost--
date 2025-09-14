"""Monitoring and feedback control utilities."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


class ProcessMonitor:
    """Monitor emissions, detect anomalies and trigger optimization."""

    def __init__(
        self,
        threshold: float,
        optimizer: Callable,
        objective: Callable[[List[float]], float],
        bounds: List[Tuple[float, float]],
        window: int = 20,
        adapt_rate: float = 0.1,
        alert_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.threshold = threshold
        self.optimizer = optimizer
        self.objective = objective
        self.bounds = bounds
        self.window = window
        self.adapt_rate = adapt_rate
        self.alert_fn = alert_fn or (lambda msg: print(msg))
        self.residuals: List[float] = []

    # ------------------------------------------------------------------
    def _detect_anomaly(self, actual: float, predicted: float) -> bool:
        residual = actual - predicted
        self.residuals.append(residual)
        if len(self.residuals) < self.window:
            return False
        recent = np.array(self.residuals[-self.window :])
        mean = recent.mean()
        std = recent.std() or 1.0
        return abs(residual - mean) > 3 * std

    def _update_threshold(self, value: float) -> None:
        self.threshold = (
            1 - self.adapt_rate
        ) * self.threshold + self.adapt_rate * value

    # ------------------------------------------------------------------
    def step(self, actual_emission: float, predicted_emission: float):
        """Process a new emission reading and run optimization when needed."""

        anomaly = self._detect_anomaly(actual_emission, predicted_emission)
        if anomaly:
            self.alert_fn(
                f"Anomaly detected: actual={actual_emission:.2f}, "
                f"predicted={predicted_emission:.2f}"
            )

        if actual_emission > self.threshold or anomaly:
            params, val = self.optimizer(self.objective, self.bounds)
            self._update_threshold(val)
            return params, val

        self._update_threshold(actual_emission)
        return None, actual_emission
