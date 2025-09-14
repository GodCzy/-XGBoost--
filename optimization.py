"""Optimization algorithms for process tuning."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


ArrayLike = Sequence[float]


def pso(
    objective: Callable[[ArrayLike], float],
    bounds: Sequence[Tuple[float, float]],
    num_particles: int = 15,
    iterations: int = 30,
    inertia: float = 0.5,
    cognitive: float = 1.5,
    social: float = 1.5,
    return_history: bool = False,
):
    """Minimize objective function using a basic PSO algorithm."""
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    pos = np.random.uniform(lb, ub, (num_particles, dim))
    vel = np.zeros_like(pos)
    personal_best_pos = pos.copy()
    personal_best_val = np.array([objective(p) for p in pos])
    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx]
    history = []

    for _ in range(iterations):
        r1, r2 = np.random.rand(2)
        vel = (
            inertia * vel
            + cognitive * r1 * (personal_best_pos - pos)
            + social * r2 * (global_best_pos - pos)
        )
        pos = pos + vel
        pos = np.clip(pos, lb, ub)
        values = np.array([objective(p) for p in pos])
        better = values < personal_best_val
        personal_best_pos[better] = pos[better]
        personal_best_val[better] = values[better]
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx]
        if return_history:
            history.append(
                {
                    "params": global_best_pos.copy().tolist(),
                    "value": float(personal_best_val[global_best_idx]),
                }
            )

    if return_history:
        return global_best_pos, float(personal_best_val[global_best_idx]), history
    return global_best_pos, float(personal_best_val[global_best_idx])


def bayesian_optimization(
    objective: Callable[[ArrayLike], float],
    bounds: Sequence[Tuple[float, float]],
    n_init: int = 5,
    iterations: int = 25,
    random_state: int | None = None,
    return_history: bool = False,
):
    """Simple Gaussian-process Bayesian optimization."""
    rng = np.random.default_rng(random_state)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    X = rng.uniform(lb, ub, (n_init, dim))
    y = np.array([objective(x) for x in X])
    history = [{"params": X[i].tolist(), "value": float(y[i])} for i in range(len(X))]
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=random_state)

    for _ in range(iterations):
        gp.fit(X, y)
        candidates = rng.uniform(lb, ub, (100, dim))
        mu, sigma = gp.predict(candidates, return_std=True)
        best = y.min()
        improvement = best - mu
        with np.errstate(divide="warn"):
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        x_next = candidates[np.argmax(ei)]
        y_next = objective(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        history.append({"params": x_next.tolist(), "value": float(y_next)})

    idx = np.argmin(y)
    best_x = X[idx]
    best_y = y[idx]
    if return_history:
        return best_x, float(best_y), history
    return best_x, float(best_y)


def genetic_algorithm(
    objective: Callable[[ArrayLike], float],
    bounds: Sequence[Tuple[float, float]],
    population_size: int = 20,
    generations: int = 30,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.5,
    random_state: int | None = None,
    return_history: bool = False,
):
    """Minimize objective using a basic genetic algorithm."""
    rng = np.random.default_rng(random_state)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    pop = rng.uniform(lb, ub, (population_size, dim))
    fitness = np.array([objective(ind) for ind in pop])
    history = [
        {"params": pop[i].tolist(), "value": float(fitness[i])}
        for i in range(population_size)
    ]

    for _ in range(generations):
        parents = []
        for _ in range(population_size):
            idx1, idx2 = rng.integers(0, population_size, 2)
            winner = pop[idx1] if fitness[idx1] < fitness[idx2] else pop[idx2]
            parents.append(winner)
        parents = np.array(parents)

        offspring = parents.copy()
        for i in range(0, population_size, 2):
            if rng.random() < crossover_rate and i + 1 < population_size:
                alpha = rng.random()
                offspring[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                offspring[i + 1] = alpha * parents[i + 1] + (1 - alpha) * parents[i]

        mutation = rng.normal(0, 1, offspring.shape) * (ub - lb) * 0.1
        mask = rng.random(offspring.shape) < mutation_rate
        offspring = np.clip(offspring + mask * mutation, lb, ub)

        pop = offspring
        fitness = np.array([objective(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        history.append(
            {
                "params": pop[best_idx].tolist(),
                "value": float(fitness[best_idx]),
            }
        )

    best_idx = np.argmin(fitness)
    best_params = pop[best_idx]
    best_val = fitness[best_idx]
    if return_history:
        return best_params, float(best_val), history
    return best_params, float(best_val)
