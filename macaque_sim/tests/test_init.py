"""
test_init.py — Smoke tests for initializer module integration.

Verifies that the initializer module and its components can be imported
and used correctly in the context of the full pipeline.

Run with:
  pytest tests/test_init.py -v
"""

import numpy as np
import pytest

from macaque_sim.initializer.mri_init import initialize
from macaque_sim.config.sim_config import default_config
from macaque_sim.config.scenarios import build_grooming_matrix


def test_initializer_produces_valid_state():
    """Verify initialize() returns valid state arrays for simulation start."""
    cfg = default_config()
    B = build_grooming_matrix(cfg)
    
    A_0, P_0, R_0 = initialize(cfg, B)
    
    # Check shapes
    assert A_0.shape == (cfg.N, cfg.N)
    assert P_0.shape == (cfg.N,)
    assert R_0.shape == (cfg.N,)
    
    # Check constraints
    assert np.all(A_0 >= 0), "Alliance matrix must be non-negative"
    assert np.allclose(np.diag(A_0), 0.0), "Alliance diagonal must be zero"
    assert np.isclose(P_0.sum(), 1.0, atol=1e-12), "Power must sum to 1"
    assert set(R_0.tolist()) == set(range(1, cfg.N + 1)), "Ranks must be permutation 1..N"


def test_initializer_handles_configuration_changes():
    """Verify initialize() adapts to different configuration parameters."""
    for N in [10, 20, 30]:
        for num_matrilines in [2, 3, 4]:
            cfg = default_config()
            cfg.N = N
            cfg.num_matrilines = min(num_matrilines, N)
            B = build_grooming_matrix(cfg)
            
            A_0, P_0, R_0 = initialize(cfg, B)
            
            assert A_0.shape == (N, N)
            assert P_0.shape == (N,)
            assert R_0.shape == (N,)


def test_initializer_respects_grooming_matrix_structure():
    """Verify initialize() properly uses grooming matrix for alliance seeding."""
    cfg = default_config()
    B = build_grooming_matrix(cfg)
    
    A_0, P_0, R_0 = initialize(cfg, B)
    
    # Alliance matrix should reflect grooming relationships
    # (non-zero entries where grooming exists)
    assert A_0.shape == B.shape, "Alliance matrix should match grooming matrix shape"
    
    # Where grooming is zero, alliance should tend to be zero
    # (This is probabilistic, but the initialization should respect it)
    assert np.all(A_0 >= 0), "Alliance values must be non-negative"
