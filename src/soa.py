from __future__ import annotations

import numpy as np


def update_particles_soa(
    particles: dict,
    steps: int,
    dt: float,
    ax: float,
    ay: float,
    damping: float,
) -> dict:
    """Update particles using a Structure of Arrays representation.

    Arrays are stored contiguously and updated with NumPy vectorized operations,
    which reduces Python-loop overhead and improves data locality.
    """
    x = particles["x"]
    y = particles["y"]
    vx = particles["vx"]
    vy = particles["vy"]
    mass = particles["mass"]

    for _ in range(steps):
        vx[:] = (vx + ax * dt / mass) * damping
        vy[:] = (vy + ay * dt / mass) * damping
        x[:] = x + vx * dt
        y[:] = y + vy * dt

    return particles
