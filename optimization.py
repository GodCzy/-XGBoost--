"""Simple Particle Swarm Optimization implementation for process tuning."""
from __future__ import annotations

import numpy as np


def pso(objective, bounds, num_particles=15, iterations=30,
        inertia=0.5, cognitive=1.5, social=1.5):
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

    for _ in range(iterations):
        r1, r2 = np.random.rand(2)
        vel = (inertia * vel +
               cognitive * r1 * (personal_best_pos - pos) +
               social * r2 * (global_best_pos - pos))
        pos = pos + vel
        pos = np.clip(pos, lb, ub)
        values = np.array([objective(p) for p in pos])
        better = values < personal_best_val
        personal_best_pos[better] = pos[better]
        personal_best_val[better] = values[better]
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx]
    return global_best_pos, personal_best_val[global_best_idx]
