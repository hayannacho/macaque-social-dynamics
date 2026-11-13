"""
test_grooming_matrix.py — Unit tests for grooming matrix generation in scenarios.py.

Covers:
  - uniform_grooming(), upward_biased_grooming(), coalition_grooming()
    return matrices with correct shape, zero diagonal, and non-negative values.
  - build_grooming_matrix() dispatches correctly from config.grooming_strategy.
  - Different strategies produce measurably different matrices.
  - Uniform grooming may be symmetric, while upward_biased and coalition
    should produce directed (non-symmetric) matrices.

Run with:
  pytest tests/test_grooming_matrix.py -v
"""

from types import SimpleNamespace

import numpy as np
import pytest

from macaque_sim.config.scenarios import (
    uniform_grooming,
    upward_biased_grooming,
    coalition_grooming,
    build_grooming_matrix,
)


def assert_basic_matrix_properties(B: np.ndarray, N: int) -> None:
    assert isinstance(B, np.ndarray)
    assert B.shape == (N, N)
    assert np.all(B >= 0)
    assert np.allclose(np.diag(B), 0.0)


def test_uniform_grooming_basic_properties():
    N = 6
    B = uniform_grooming(N)

    assert_basic_matrix_properties(B, N)

    # all off-diagonal entries should be equal in uniform grooming
    off_diag = B[~np.eye(N, dtype=bool)]
    assert np.allclose(off_diag, off_diag[0])


def test_upward_biased_grooming_basic_properties():
    N = 5
    ranks = np.array([1, 2, 3, 4, 5])  # monkey 0 highest, monkey 4 lowest
    B = upward_biased_grooming(N, ranks)

    assert_basic_matrix_properties(B, N)

    # monkey 3 (rank 4) should groom monkey 0 (rank 1) more than monkey 4 (rank 5)
    assert B[3, 0] > B[3, 4]

    # directed / asymmetric check: these should differ
    assert B[4, 0] != B[0, 4]


def test_coalition_grooming_basic_properties():
    N = 6
    ranks = np.array([1, 2, 3, 4, 5, 6])
    clique = [3, 4, 5]
    B = coalition_grooming(N, ranks, clique)

    assert_basic_matrix_properties(B, N)

    # clique members should groom each other more than background
    assert B[3, 4] > B[3, 1]
    assert B[4, 5] > B[4, 0]

    # still no self-grooming
    assert B[3, 3] == 0.0


def test_build_grooming_matrix_uniform():
    cfg = SimpleNamespace(N=5, grooming_strategy="uniform")
    B = build_grooming_matrix(cfg)

    assert_basic_matrix_properties(B, 5)


def test_build_grooming_matrix_upward_biased():
    cfg = SimpleNamespace(N=5, grooming_strategy="upward_biased")
    ranks = np.array([1, 2, 3, 4, 5])

    B = build_grooming_matrix(cfg, ranks=ranks)

    assert_basic_matrix_properties(B, 5)

    assert B[3, 0] > B[3, 4]


def test_build_grooming_matrix_coalition():
    cfg = SimpleNamespace(N=6, grooming_strategy="coalition")
    ranks = np.array([1, 2, 3, 4, 5, 6])
    clique = [3, 4, 5]

    B = build_grooming_matrix(cfg, ranks=ranks, clique=clique)

    assert_basic_matrix_properties(B, 6)
    assert B[3, 4] > B[3, 1]


def test_build_grooming_matrix_unknown_strategy_raises():
    cfg = SimpleNamespace(N=5, grooming_strategy="banana")

    with pytest.raises(ValueError):
        build_grooming_matrix(cfg)


def test_different_strategies_produce_different_matrices():
    N = 6
    ranks = np.array([1, 2, 3, 4, 5, 6])
    clique = [3, 4, 5]

    B_uniform = uniform_grooming(N)
    B_upward = upward_biased_grooming(N, ranks)
    B_coalition = coalition_grooming(N, ranks, clique)

    assert not np.allclose(B_uniform, B_upward)
    assert not np.allclose(B_uniform, B_coalition)
    assert not np.allclose(B_upward, B_coalition)


def test_uniform_is_symmetric_but_upward_and_coalition_are_directed():
    N = 6
    ranks = np.array([1, 2, 3, 4, 5, 6])
    clique = [3, 4, 5]

    B_uniform = uniform_grooming(N)
    B_upward = upward_biased_grooming(N, ranks)
    B_coalition = coalition_grooming(N, ranks, clique)

    # uniform baseline may be symmetric except for zero diagonal
    assert np.allclose(B_uniform, B_uniform.T)

    # these should be non-symmetric / directed in typical use
    assert not np.allclose(B_upward, B_upward.T)
    assert not np.allclose(B_coalition, B_coalition.T)