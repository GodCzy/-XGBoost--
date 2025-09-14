"""Application configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass
import os
import logging


@dataclass
class Config:
    """Configuration loaded from environment variables."""

    threshold: float = float(os.getenv("EMISSION_THRESHOLD", 50))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    models_dir: str = os.getenv("MODELS_DIR", "models")

    def logging_level(self) -> int:
        return getattr(logging, self.log_level.upper(), logging.INFO)
