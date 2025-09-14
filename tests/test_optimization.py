import numpy as np

from optimization import pso


def sphere(x):
    return np.sum(x**2)


def test_pso_finds_minimum_near_zero():
    bounds = [(-1, 1), (-1, 1)]
    pos, val = pso(sphere, bounds, num_particles=10, iterations=10)
    assert len(pos) == 2
    assert val >= 0
