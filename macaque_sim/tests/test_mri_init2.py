#Contact Aaryan for Questions
import numpy as np
import pytest

from macaque_sim.config.sim_config import default_config
from macaque_sim.initializer.mri_init import (
    Matriline,
    build_matrilines,
    assign_initial_ranks,
    seed_power_vector,
    seed_alliance_matrix,
    initialize,
)


def make_config(**overrides):
    cfg = default_config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


# ----------------------------
# build_matrilines()
# ----------------------------

def test_build_matrilines_balanced_complete_and_ranked():
    matrilines = build_matrilines(N=10, num_matrilines=3)

    assert len(matrilines) == 3
    assert [m.rank for m in matrilines] == [1, 2, 3]
    assert [m.members for m in matrilines] == [
        [0, 3, 6, 9],
        [1, 4, 7],
        [2, 5, 8],
    ]

    members = [member for mat in matrilines for member in mat.members]
    sizes = [len(mat.members) for mat in matrilines]

    assert sorted(members) == list(range(10))
    assert len(members) == len(set(members))
    assert max(sizes) - min(sizes) <= 1


def test_build_matrilines_single_matriline():
    matrilines = build_matrilines(N=5, num_matrilines=1)

    assert len(matrilines) == 1
    assert matrilines[0].rank == 1
    assert matrilines[0].members == [0, 1, 2, 3, 4]


def test_build_matrilines_each_monkey_own_matriline():
    matrilines = build_matrilines(N=4, num_matrilines=4)

    assert len(matrilines) == 4
    assert [m.rank for m in matrilines] == [1, 2, 3, 4]
    assert [m.members for m in matrilines] == [[0], [1], [2], [3]]


@pytest.mark.parametrize(
    "N,num_matrilines,msg",
    [
        (5, 0, "must be >= 1"),
        (5, -1, "must be >= 1"),
        (5, 6, "cannot exceed N"),
    ],
)
def test_build_matrilines_rejects_invalid_num_matrilines(N, num_matrilines, msg):
    with pytest.raises(ValueError, match=msg):
        build_matrilines(N=N, num_matrilines=num_matrilines)


# ----------------------------
# assign_initial_ranks()
# ----------------------------

def test_assign_initial_ranks_exact_order_and_rank_accuracy():
    matrilines = [
        Matriline(members=[0, 3, 6], rank=1),
        Matriline(members=[1, 4, 7], rank=2),
        Matriline(members=[2, 5, 8], rank=3),
    ]

    R_0 = assign_initial_ranks(matrilines, N=9)

    expected = np.array([3, 6, 9, 2, 5, 8, 1, 4, 7])
    assert np.array_equal(R_0, expected)

    # Highest-ranking matriline's youngest member gets rank 1.
    assert R_0[6] == 1

    # R_0 is a permutation of [1..N].
    assert np.array_equal(np.sort(R_0), np.arange(1, 10))


def test_assign_initial_ranks_sorts_by_matriline_rank_not_input_order():
    matrilines = [
        Matriline(members=[2, 5, 8], rank=3),
        Matriline(members=[0, 3, 6], rank=1),
        Matriline(members=[1, 4, 7], rank=2),
    ]

    R_0 = assign_initial_ranks(matrilines, N=9)

    expected = np.array([3, 6, 9, 2, 5, 8, 1, 4, 7])
    assert np.array_equal(R_0, expected)
    assert R_0[6] == 1


def test_assign_initial_ranks_single_matriline_uses_later_listed_as_younger():
    matrilines = [Matriline(members=[10, 11, 12, 13], rank=1)]
    R_0 = assign_initial_ranks(matrilines, N=14)

    assert R_0[13] == 1
    assert R_0[12] == 2
    assert R_0[11] == 3
    assert R_0[10] == 4
    assert np.array_equal(np.sort(R_0[R_0 > 0]), np.arange(1, 5))


# ----------------------------
# seed_power_vector()
# ----------------------------

def test_seed_power_vector_exact_values_normalized_and_monotone():
    R_0 = np.array([1, 2, 3, 4, 5])
    P_0 = seed_power_vector(R_0)

    expected = np.array([5, 4, 3, 2, 1], dtype=float) / 15.0
    assert np.allclose(P_0, expected, atol=0.0, rtol=0.0)
    assert abs(P_0.sum() - 1.0) <= 1e-12
    assert np.all(P_0[:-1] > P_0[1:])


def test_seed_power_vector_handles_unsorted_rank_vector_correctly():
    R_0 = np.array([3, 1, 4, 2])
    P_0 = seed_power_vector(R_0)

    expected_scores = np.array([2, 4, 1, 3], dtype=float)
    expected = expected_scores / expected_scores.sum()

    assert np.allclose(P_0, expected, atol=0.0, rtol=0.0)
    assert abs(P_0.sum() - 1.0) <= 1e-12
    assert P_0[1] > P_0[3] > P_0[0] > P_0[2]

# ----------------------------
# seed_alliance_matrix()
# ----------------------------

def test_seed_alliance_matrix_exact_values_nonnegative_zero_diagonal():
    B = np.array([
        [1.0, 2.0, 0.5],
        [3.0, 4.0, 1.5],
        [0.2, 0.8, 5.0],
    ])

    A_0 = seed_alliance_matrix(B, scale=0.05)

    expected = np.array([
        [0.0, 0.10, 0.025],
        [0.15, 0.0, 0.075],
        [0.01, 0.04, 0.0],
    ])

    assert A_0.shape == (3, 3)
    assert np.allclose(A_0, expected, atol=1e-15, rtol=0.0)
    assert np.all(A_0 >= 0.0)
    assert np.allclose(np.diag(A_0), 0.0, atol=0.0, rtol=0.0)


def test_seed_alliance_matrix_scale_zero_gives_zero_matrix():
    B = np.array([
        [7.0, 2.0],
        [3.0, 9.0],
    ])

    A_0 = seed_alliance_matrix(B, scale=0.0)

    assert A_0.shape == (2, 2)
    assert np.array_equal(A_0, np.zeros((2, 2)))
    assert np.all(A_0 >= 0.0)
    assert np.allclose(np.diag(A_0), 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "B,scale,msg",
    [
        (np.ones((2, 3)), 0.05, "B must be a square matrix"),
        (np.ones(3), 0.05, "B must be a square matrix"),
        (np.array([[0.0, -1.0], [1.0, 0.0]]), 0.05, "B cannot contain negative entries"),
        (np.ones((3, 3)), -0.1, "scale must be non-negative"),
    ],
)
def test_seed_alliance_matrix_rejects_invalid_inputs(B, scale, msg):
    with pytest.raises(ValueError, match=msg):
        seed_alliance_matrix(B, scale=scale)


# ----------------------------
# initialize()
# ----------------------------

def test_initialize_exact_outputs_for_valid_case():
    cfg = make_config(N=30, num_matrilines=2, alliance_seed_scale=0.05)
    B = np.arange(1, 30 * 30 + 1, dtype=float).reshape(30, 30)

    A_0, P_0, R_0 = initialize(cfg, B)

    expected_R_0 = assign_initial_ranks(build_matrilines(cfg.N, cfg.num_matrilines), cfg.N)
    expected_P_0 = seed_power_vector(expected_R_0)
    expected_A_0 = B.astype(float) * 0.05
    np.fill_diagonal(expected_A_0, 0.0)

    assert np.array_equal(R_0, expected_R_0)
    assert np.allclose(P_0, expected_P_0, atol=1e-15, rtol=0.0)
    assert np.allclose(A_0, expected_A_0, atol=1e-15, rtol=0.0)

    assert np.array_equal(np.sort(R_0), np.arange(1, cfg.N + 1))
    assert abs(P_0.sum() - 1.0) <= 1e-12
    assert np.all(A_0 >= 0.0)
    assert np.allclose(np.diag(A_0), 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "N,num_matrilines",
    [
    (30, 1),
    (30, 4),
    (30, 30),
    (40, 5),
    (50, 10),
    (50, 50),
    ],
)
def test_initialize_dynamic_sizing_and_validity(N, num_matrilines):
    cfg = make_config(N=N, num_matrilines=num_matrilines, alliance_seed_scale=0.05)
    B = np.ones((N, N))

    A_0, P_0, R_0 = initialize(cfg, B)

    assert R_0.shape == (N,)
    assert P_0.shape == (N,)
    assert A_0.shape == (N, N)

    assert np.array_equal(np.sort(R_0), np.arange(1, N + 1))
    assert abs(P_0.sum() - 1.0) <= 1e-12
    assert np.all(A_0 >= 0.0)
    assert np.allclose(np.diag(A_0), 0.0, atol=0.0, rtol=0.0)

    if N > 1:
        alpha = np.where(R_0 == 1)[0][0]
        assert 0 <= alpha < N


def test_initialize_rank_1_goes_to_highest_matriline_youngest_member():
    cfg = make_config(N=30, num_matrilines=3, alliance_seed_scale=0.05)
    B = np.ones((30, 30))

    A_0, P_0, R_0 = initialize(cfg, B)

    assert R_0[27] == 1

    assert np.array_equal(np.sort(R_0), np.arange(1, 31))
    assert abs(P_0.sum() - 1.0) <= 1e-12
    assert np.all(A_0 >= 0.0)
    assert np.allclose(np.diag(A_0), 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "B,msg",
    [
        (np.ones((29, 29)), "B must have shape"),
        (np.ones((30, 29)), "B must have shape"),
    ],
)
def test_initialize_rejects_bad_B_shape(B, msg):
    cfg = make_config(N=30, num_matrilines=4, alliance_seed_scale=0.05)
    with pytest.raises(ValueError, match=msg):
        initialize(cfg, B)


def test_initialize_rejects_negative_B_via_alliance_seeding():
    cfg = make_config(N=30, num_matrilines=2, alliance_seed_scale=0.05)
    B = np.ones((30, 30))
    B[1, 2] = -2.0

    with pytest.raises(ValueError, match="B cannot contain negative entries"):
        initialize(cfg, B)


# ----------------------------
# Final invariant-focused tests
# ----------------------------

def test_initialize_invariants_hold_for_nontrivial_case():
    cfg = make_config(N=30, num_matrilines=4, alliance_seed_scale=0.03125)
    rng = np.random.default_rng(12345)
    B = rng.random((cfg.N, cfg.N))

    A_0, P_0, R_0 = initialize(cfg, B)

    assert np.array_equal(np.sort(R_0), np.arange(1, cfg.N + 1))
    assert abs(P_0.sum() - 1.0) <= 1e-12
    assert np.all(A_0 >= 0.0)
    assert np.allclose(np.diag(A_0), 0.0, atol=0.0, rtol=0.0)

    alpha_idx = np.argmin(R_0)
    omega_idx = np.argmax(R_0)
    assert P_0[alpha_idx] > P_0[omega_idx]


def test_alliance_matrix_off_diagonal_matches_scaled_B():
    cfg = make_config(N=30, num_matrilines=3, alliance_seed_scale=0.125)
    B = np.arange(30 * 30, dtype=float).reshape(30, 30)

    A_0, _, _ = initialize(cfg, B)

    for i in range(cfg.N):
        for j in range(cfg.N):
            if i == j:
                assert A_0[i, j] == 0.0
            else:
                assert A_0[i, j] == B[i, j] * cfg.alliance_seed_scale