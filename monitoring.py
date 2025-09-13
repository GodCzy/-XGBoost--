"""Monitoring and feedback control utilities."""
from __future__ import annotations

from typing import Callable, List, Tuple


class ProcessMonitor:
    """Monitor emissions and trigger optimization when needed."""

    def __init__(self, threshold: float, optimizer: Callable,
                 objective: Callable[[List[float]], float], bounds: List[Tuple[float, float]]):
        self.threshold = threshold
        self.optimizer = optimizer
        self.objective = objective
        self.bounds = bounds

    def adjust(self, current_emission: float):
        """Run optimizer if emission exceeds threshold."""
        if current_emission > self.threshold:
            params, val = self.optimizer(self.objective, self.bounds)
            return params, val
        return None, current_emission
