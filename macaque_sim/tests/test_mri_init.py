"""
test_mri_init.py — Unit tests for initializer/mri_init.py.

Covers:
  - build_matrilines() returns the requested number of matrilines
    and includes all monkeys exactly once.
  - assign_initial_ranks() returns a valid permutation of ranks 1..N.
  - youngest-ascendancy is implemented within each matriline
    (later-listed members receive better ranks).
  - seed_power_vector() returns a non-negative vector summing to 1.0.
  - higher-ranked monkeys receive larger initial power.
  - seed_alliance_matrix() returns a non-negative matrix with zero diagonal.
  - initialize() returns correctly shaped outputs (A_0, P_0, R_0).
  - Changing N or num_matrilines produces correctly shaped outputs.
  - validation checks raise errors for invalid inputs.

Run with:
  pytest tests/test_mri_init.py -v
"""

import numpy as np
import pytest

from macaque_sim.initializer.mri_init import (
    Matriline,
    build_matrilines,
    assign_initial_ranks,
    seed_power_vector,
    seed_alliance_matrix,
    initialize,
)
from macaque_sim.config.sim_config import default_config
from macaque_sim.config.scenarios import build_grooming_matrix


def test_build_matrilines_returns_expected_number_of_groups():
    N = 12
    num_matrilines = 4

    matrilines = build_matrilines(N, num_matrilines)

    assert isinstance(matrilines, list)
    assert len(matrilines) == num_matrilines
    assert all(isinstance(m, Matriline) for m in matrilines)

    all_members = [member for m in matrilines for member in m.members]
    assert sorted(all_members) == list(range(N))

    # Matriline ranks should be 1..num_matrilines
    assert [m.rank for m in matrilines] == list(range(1, num_matrilines + 1))


def test_build_matrilines_raises_when_num_matrilines_is_zero():
    with pytest.raises(ValueError):
        build_matrilines(10, 0)


def test_build_matrilines_raises_when_num_matrilines_exceeds_N():
    with pytest.raises(ValueError):
        build_matrilines(5, 6)


def test_assign_initial_ranks_returns_permutation():
    N = 10
    matrilines = build_matrilines(N, 3)

    R_0 = assign_initial_ranks(matrilines, N)

    assert isinstance(R_0, np.ndarray)
    assert R_0.shape == (N,)
    assert sorted(R_0.tolist()) == list(range(1, N + 1))


def test_youngest_ascendancy_within_matriline():
    matrilines = [
        Matriline(members=[0, 2, 4], rank=1),
        Matriline(members=[1, 3, 5], rank=2),
    ]

    R_0 = assign_initial_ranks(matrilines, 6)

    # Later-listed members are treated as younger, so they get better (smaller) ranks
    assert R_0[4] < R_0[2] < R_0[0]
    assert R_0[5] < R_0[3] < R_0[1]

    # Higher-ranking matriline should receive better ranks overall
    assert max(R_0[[0, 2, 4]]) < min(R_0[[1, 3, 5]])


def test_seed_power_vector_sums_to_one():
    R_0 = np.array([1, 2, 3, 4, 5])

    P_0 = seed_power_vector(R_0)

    assert isinstance(P_0, np.ndarray)
    assert P_0.shape == (5,)
    assert np.all(P_0 >= 0)
    assert np.isclose(P_0.sum(), 1.0, atol=1e-12)


def test_seed_power_vector_gives_more_power_to_higher_rank():
    R_0 = np.array([1, 2, 3, 4, 5])

    P_0 = seed_power_vector(R_0)

    # Higher-ranked monkeys should receive larger initial power
    assert P_0[0] > P_0[1] > P_0[2] > P_0[3] > P_0[4]


def test_seed_alliance_matrix_is_nonnegative_with_zero_diagonal():
    B = np.array([
        [0.0, 0.1, 0.2],
        [0.3, 0.0, 0.1],
        [0.2, 0.4, 0.0],
    ])

    A_0 = seed_alliance_matrix(B, scale=0.05)

    assert isinstance(A_0, np.ndarray)
    assert A_0.shape == (3, 3)
    assert np.all(A_0 >= 0)
    assert np.allclose(np.diag(A_0), 0.0)


def test_seed_alliance_matrix_raises_for_negative_entries():
    B = np.array([
        [0.0, -0.1],
        [0.2, 0.0],
    ])

    with pytest.raises(ValueError):
        seed_alliance_matrix(B, scale=0.05)


def test_seed_alliance_matrix_raises_for_negative_scale():
    B = np.array([
        [0.0, 0.1],
        [0.2, 0.0],
    ])

    with pytest.raises(ValueError):
        seed_alliance_matrix(B, scale=-0.01)


def test_initialize_returns_correct_shapes_and_constraints():
    cfg = default_config()
    B = build_grooming_matrix(cfg)

    A_0, P_0, R_0 = initialize(cfg, B)

    assert isinstance(A_0, np.ndarray)
    assert isinstance(P_0, np.ndarray)
    assert isinstance(R_0, np.ndarray)

    assert A_0.shape == (cfg.N, cfg.N)
    assert P_0.shape == (cfg.N,)
    assert R_0.shape == (cfg.N,)

    assert np.all(A_0 >= 0)
    assert np.allclose(np.diag(A_0), 0.0)
    assert np.isclose(P_0.sum(), 1.0, atol=1e-12)
    assert sorted(R_0.tolist()) == list(range(1, cfg.N + 1))


def test_initialize_changes_shape_when_N_changes():
    cfg = default_config()
    cfg.N = 40
    cfg.num_matrilines = 5

    B = build_grooming_matrix(cfg)
    A_0, P_0, R_0 = initialize(cfg, B)

    assert A_0.shape == (40, 40)
    assert P_0.shape == (40,)
    assert R_0.shape == (40,)


def test_build_matrilines_respects_num_matrilines_change():
    N = 15
    matrilines_3 = build_matrilines(N, 3)
    matrilines_5 = build_matrilines(N, 5)

    assert len(matrilines_3) == 3
    assert len(matrilines_5) == 5

    all_members_3 = sorted([member for m in matrilines_3 for member in m.members])
    all_members_5 = sorted([member for m in matrilines_5 for member in m.members])

    assert all_members_3 == list(range(N))
    assert all_members_5 == list(range(N))


def test_initialize_raises_for_wrong_B_shape():
    cfg = default_config()
    wrong_B = np.zeros((cfg.N - 1, cfg.N - 1))

    with pytest.raises(ValueError):
        initialize(cfg, wrong_B)