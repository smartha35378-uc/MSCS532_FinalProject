from __future__ import annotations

from typing import List, Dict


def update_particles_aos(
    particles: List[Dict[str, float]],
    steps: int,
    dt: float,
    ax: float,
    ay: float,
    damping: float,
) -> List[Dict[str, float]]:
    """Update particles using an Array of Structures representation.

    Each particle is a Python dictionary, which is easy to understand
    but less efficient for large numerical workloads because of object
    indirection and poor memory locality.
    """
    for _ in range(steps):
        for p in particles:
            # Update velocity first, then apply it to position.
            p["vx"] = (p["vx"] + ax * dt / p["mass"]) * damping
            p["vy"] = (p["vy"] + ay * dt / p["mass"]) * damping
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
    return particles
