"""Experiment tracking utilities for optimization algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ArrayLike = Sequence[float]


class ExperimentManager:
    """Run optimizers, log results, and create basic plots."""

    def __init__(self, log_file: str = "experiments.csv"):
        self.log_file = Path(log_file)

    def run(
        self,
        name: str,
        optimizer: Callable,
        objective: Callable[[ArrayLike], float],
        bounds: Sequence[Tuple[float, float]],
        **kwargs,
    ):
        result = optimizer(objective, bounds, return_history=True, **kwargs)
        best_params, best_val, history = result
        record = {
            "algorithm": name,
            "best_params": best_params.tolist(),
            "best_val": float(best_val),
            "iterations": len(history),
        }
        self._log(record)
        self.plot_history(name, history)
        self.plot_search_space(name, history)
        return best_params, best_val

    def _log(self, record: dict):
        df = pd.DataFrame([record])
        if self.log_file.exists():
            df.to_csv(self.log_file, mode="a", header=False, index=False)
        else:
            df.to_csv(self.log_file, index=False)

    def plot_history(self, name: str, history):
        vals = [h["value"] for h in history]
        plt.figure()
        plt.plot(vals, marker="o")
        plt.title(f"{name} performance")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.tight_layout()
        plt.savefig(f"{name}_performance.png")
        plt.close()

    def plot_search_space(self, name: str, history):
        params = [h["params"] for h in history]
        arr = pd.DataFrame(params)
        plt.figure()
        plt.scatter(arr.iloc[:, 0], arr.iloc[:, 1], c=range(len(arr)), cmap="viridis")
        plt.title(f"{name} search space")
        plt.xlabel("param0")
        plt.ylabel("param1")
        plt.tight_layout()
        plt.savefig(f"{name}_space.png")
        plt.close()

    def compare(self):
        if self.log_file.exists():
            return pd.read_csv(self.log_file)
        return pd.DataFrame()
