"""Monitoring and feedback control utilities."""

from __future__ import annotations

from typing import Callable, List, Tuple
import logging


logger = logging.getLogger(__name__)


class ProcessMonitor:
    """Monitor emissions and trigger optimization when needed."""

    def __init__(
        self,
        threshold: float,
        optimizer: Callable,
        objective: Callable[[List[float]], float],
        bounds: List[Tuple[float, float]],
    ):
        self.threshold = threshold
        self.optimizer = optimizer
        self.objective = objective
        self.bounds = bounds

    def adjust(self, current_emission: float):
        """Run optimizer if emission exceeds threshold."""
        if current_emission > self.threshold:
            logger.info(
                "Emission %.2f exceeds threshold %.2f",
                current_emission,
                self.threshold,
            )
            try:
                params, val = self.optimizer(self.objective, self.bounds)
                logger.info("Optimization finished with value %.3f", val)
                return params, val
            except Exception:  # pragma: no cover - optimizer errors
                logger.exception("Optimization failed")
                raise
        logger.debug(
            "Emission %.2f within threshold %.2f",
            current_emission,
            self.threshold,
        )
        return None, current_emission
