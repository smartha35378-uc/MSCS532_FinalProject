from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
# Allow the tests to import modules directly from src/.
sys.path.insert(0, str(PROJECT_SRC))

from aos import update_particles_aos
from soa import update_particles_soa
from benchmark import validate_equivalence
from utils import generate_particle_data


class TestParticleOptimization(unittest.TestCase):
    def test_aos_and_soa_are_equivalent(self) -> None:
        aos, soa = generate_particle_data(n=1000, seed=7)

        # Both versions should produce the same final values.
        aos_result = update_particles_aos(
            particles=deepcopy(aos),
            steps=10,
            dt=0.01,
            ax=0.001,
            ay=-0.002,
            damping=0.999,
        )
        soa_result = update_particles_soa(
            particles={k: v.copy() for k, v in soa.items()},
            steps=10,
            dt=0.01,
            ax=0.001,
            ay=-0.002,
            damping=0.999,
        )

        self.assertTrue(validate_equivalence(aos_result, soa_result))

    def test_state_changes_after_updates(self) -> None:
        aos, soa = generate_particle_data(n=100, seed=3)
        original_x = soa["x"].copy()

        # A few steps should change the particle positions.
        update_particles_soa(
            particles=soa,
            steps=5,
            dt=0.01,
            ax=0.001,
            ay=-0.002,
            damping=0.999,
        )

        self.assertFalse(np.allclose(original_x, soa["x"]))


if __name__ == "__main__":
    unittest.main()
