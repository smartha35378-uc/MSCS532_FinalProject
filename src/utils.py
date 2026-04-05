from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    sizes: Tuple[int, ...] = (10_000, 50_000, 100_000, 250_000)
    steps: int = 20
    repeats: int = 5
    seed: int = 42
    dt: float = 0.01
    ax: float = 0.001
    ay: float = -0.002
    damping: float = 0.999


def generate_particle_data(n: int, seed: int = 42) -> Tuple[List[Dict[str, float]], dict]:
    """Generate equivalent datasets for AoS and SoA implementations."""
    rng = np.random.default_rng(seed)

    # Build one shared set of values for both layouts.
    x = rng.random(n, dtype=np.float64)
    y = rng.random(n, dtype=np.float64)
    vx = rng.normal(0.0, 1.0, n).astype(np.float64)
    vy = rng.normal(0.0, 1.0, n).astype(np.float64)
    mass = rng.uniform(0.5, 5.0, n).astype(np.float64)

    aos = [
        {
            "x": float(x[i]),
            "y": float(y[i]),
            "vx": float(vx[i]),
            "vy": float(vy[i]),
            "mass": float(mass[i]),
        }
        for i in range(n)
    ]

    soa = {
        # Keep each field in its own contiguous array.
        "x": x.copy(),
        "y": y.copy(),
        "vx": vx.copy(),
        "vy": vy.copy(),
        "mass": mass.copy(),
    }
    return aos, soa


def relative_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    return np.allclose(a, b, rtol=tol, atol=tol)


def summarize_speedup(aos_seconds: float, soa_seconds: float) -> float:
    if soa_seconds <= 0:
        raise ValueError("SoA time must be positive.")
    return aos_seconds / soa_seconds
