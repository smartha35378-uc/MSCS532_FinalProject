from __future__ import annotations

import csv
import time
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Dict, List

import numpy as np

from aos import update_particles_aos
from soa import update_particles_soa
from utils import BenchmarkConfig, generate_particle_data, relative_close, summarize_speedup


def _clone_soa(soa: dict) -> dict:
    # Copy arrays so each benchmark run starts from the same state.
    return {k: v.copy() for k, v in soa.items()}


def _aos_to_arrays(aos: list) -> dict:
    # Convert AoS output into arrays for an easy field-by-field comparison.
    return {
        "x": np.array([p["x"] for p in aos], dtype=np.float64),
        "y": np.array([p["y"] for p in aos], dtype=np.float64),
        "vx": np.array([p["vx"] for p in aos], dtype=np.float64),
        "vy": np.array([p["vy"] for p in aos], dtype=np.float64),
        "mass": np.array([p["mass"] for p in aos], dtype=np.float64),
    }


def validate_equivalence(aos_result: list, soa_result: dict, tol: float = 1e-9) -> bool:
    aos_arrays = _aos_to_arrays(aos_result)
    return all(relative_close(aos_arrays[key], soa_result[key], tol=tol) for key in aos_arrays)


def benchmark_one_size(n: int, config: BenchmarkConfig) -> Dict[str, float]:
    aos_base, soa_base = generate_particle_data(n=n, seed=config.seed)

    aos_times: List[float] = []
    soa_times: List[float] = []

    for _ in range(config.repeats):
        # Time AoS on a fresh copy of the same input data.
        aos_particles = deepcopy(aos_base)
        start = time.perf_counter()
        aos_result = update_particles_aos(
            particles=aos_particles,
            steps=config.steps,
            dt=config.dt,
            ax=config.ax,
            ay=config.ay,
            damping=config.damping,
        )
        aos_times.append(time.perf_counter() - start)

        # Time SoA on an equivalent fresh copy.
        soa_particles = _clone_soa(soa_base)
        start = time.perf_counter()
        soa_result = update_particles_soa(
            particles=soa_particles,
            steps=config.steps,
            dt=config.dt,
            ax=config.ax,
            ay=config.ay,
            damping=config.damping,
        )
        soa_times.append(time.perf_counter() - start)

    # Validate correctness on one equivalent run
    aos_particles = deepcopy(aos_base)
    soa_particles = _clone_soa(soa_base)
    aos_result = update_particles_aos(
        particles=aos_particles,
        steps=config.steps,
        dt=config.dt,
        ax=config.ax,
        ay=config.ay,
        damping=config.damping,
    )
    soa_result = update_particles_soa(
        particles=soa_particles,
        steps=config.steps,
        dt=config.dt,
        ax=config.ax,
        ay=config.ay,
        damping=config.damping,
    )
    equivalent = validate_equivalence(aos_result, soa_result)

    # Use the median to reduce the effect of noisy runs.
    aos_median = median(aos_times)
    soa_median = median(soa_times)
    speedup = summarize_speedup(aos_median, soa_median)

    return {
        "n": n,
        "aos_median_seconds": aos_median,
        "soa_median_seconds": soa_median,
        "speedup_x": speedup,
        "equivalent": float(equivalent),
    }


def run_benchmarks(config: BenchmarkConfig, output_csv: Path) -> List[Dict[str, float]]:
    rows = [benchmark_one_size(n, config) for n in config.sizes]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        # Save results so they can be reused in the report.
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "aos_median_seconds", "soa_median_seconds", "speedup_x", "equivalent"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows
