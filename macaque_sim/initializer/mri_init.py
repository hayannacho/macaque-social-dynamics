# mri_init.py — Matrilineal Rank Inheritance initializer
# Assignee: Dev C + Dev D (S1-T3)
# See: project_management/tasks/sprint_1/S1-T3_mri_initializer.md

"""
This module builds the simulation state at t = 0:

  - R_0 : initial discrete rank array (1 = alpha)
  - P_0 : initial power vector, normalized to sum to 1.0
  - A_0 : initial alliance matrix, seeded from the grooming matrix B

Biological logic
----------------
The initializer follows a simplified MRI rule:

1. Monkeys are divided into matrilines.
2. Matrilines are ordered by matriline rank (rank 1 matriline is the highest-ranking family line).
3. Within each matriline, younger members outrank older members ("youngest-ascendancy").

Because explicit age data are not available at this stage, a following simple proxy is used:
- later-listed members within a matriline are treated as younger.

This implements the Sprint 1 S1-T3 task:
  - define a Matriline structure
  - assign ranks by matriline order, then youngest-first within matriline
  - seed A_0 from B
  - seed P_0 from ranks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Matriline:
    """
    A maternal family line.

    Attributes
    ----------
        members : list[int]
            Monkey indices belonging to this matriline.
        rank : int
            Matriline rank, where 1 = highest-ranking matriline.
    """
    members: List[int]
    rank: int


def build_matrilines(N: int, num_matrilines: int) -> List[Matriline]:
    """
    Split N monkeys into num_matrilines matrilines.

    Returns
    -------
    list[Matriline]
        A list of Matriline objects.

    Notes
    -----
    - Matrilines with smaller rank values have higher social status.
    - Monkeys are assigned to matrilines in a cyclic (one-by-one) manner
      to keep group sizes approximately balanced.
    - The order of members is preserved and later used as a simple proxy
      for age (later-listed members are treated as younger).
    """
    if num_matrilines < 1:
        raise ValueError("num_matrilines must be >= 1.")
    if num_matrilines > N:
        raise ValueError("num_matrilines cannot exceed N.")

    matrilines = [
        Matriline(members=[], rank=i + 1)
        for i in range(num_matrilines)
    ]

    # Assign monkeys one-by-one across matrilines to keep sizes roughly balanced
    for monkey in range(N):
        idx = monkey % num_matrilines
        matrilines[idx].members.append(monkey)

    return matrilines


def assign_initial_ranks(matrilines: List[Matriline], N: int) -> np.ndarray:
    """
    Assign the initial discrete ranks R_0.

    Rank assignment rule
    --------------------
    1. Higher-ranking matrilines come first.
    2. Within each matriline, younger members come first (youngest-ascendancy).

    Convention
    ----------
    Smaller rank number = higher status.
    Rank 1 = alpha.

    Assumption
    ----------
    Later-listed members inside each matriline are treated as younger,
    so we reverse the member list when assigning ranks.
    """
    R_0 = np.zeros(N, dtype=int)
    current_rank = 1

    # Sort by matriline rank so rank-1 matriline is processed first
    matrilines_sorted = sorted(matrilines, key=lambda m: m.rank)

    for matriline in matrilines_sorted:
        # Reverse member order so later-listed (younger) members receive better ranks first
        for monkey in reversed(matriline.members):
            R_0[monkey] = current_rank
            current_rank += 1

    return R_0


def seed_power_vector(R_0: np.ndarray) -> np.ndarray:
    """
    Seed the initial power vector P_0 from ranks.

    Higher-ranked monkeys receive larger initial power.
    The vector is normalized to sum exactly to 1.0.
    """
    N = len(R_0)

    # rank 1 gets score N, rank N gets score 1
    scores = (N - R_0 + 1).astype(float)
    total = scores.sum()

    if total <= 0:
        raise ValueError("Power scores must sum to a positive value.")

    P_0 = scores / total
    return P_0


def seed_alliance_matrix(B: np.ndarray, R_0: np.ndarray | None = None, scale: float = 0.05) -> np.ndarray:
    """
    Seed the initial alliance matrix A_0 from grooming matrix B.

    Parameters
    ----------
    B : np.ndarray
        Grooming matrix of shape (N, N).
    R_0 : np.ndarray | None
        Initial rank array.
    scale : float
        Small non-negative multiplier controlling the strength of
        embryonic alliance ties at t = 0.

    Returns
    -------
    np.ndarray
        Initial alliance matrix A_0.
    """
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("B must be a square matrix.")
    if scale < 0:
        raise ValueError("scale must be non-negative.")
    if np.any(B < 0):
        raise ValueError("B cannot contain negative entries.")

    A_0 = B.astype(float) * scale
    np.fill_diagonal(A_0, 0.0)

    return A_0


def initialize(config, B: np.ndarray):
    """
    Build the full MRI-based initial state.

    Parameters
    ----------
    config : SimConfig-like object
          - N
          - num_matrilines
          - alliance_seed_scale
    B : np.ndarray
        Grooming matrix of shape (N, N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (A_0, P_0, R_0)
    """
    N = config.N

    if B.shape != (N, N):
        raise ValueError(f"B must have shape ({N}, {N}); got {B.shape}.")

    matrilines = build_matrilines(N, config.num_matrilines)
    R_0 = assign_initial_ranks(matrilines, N)
    P_0 = seed_power_vector(R_0)
    A_0 = seed_alliance_matrix(B, R_0=R_0, scale=config.alliance_seed_scale)

    # Final safety checks
    if sorted(R_0.tolist()) != list(range(1, N + 1)):
        raise ValueError("R_0 must be a permutation of ranks 1..N.")
    if not np.isclose(P_0.sum(), 1.0, atol=1e-12):
        raise ValueError("P_0 must sum to 1.0.")
    if np.any(A_0 < 0):
        raise ValueError("A_0 cannot contain negative entries.")
    if not np.allclose(np.diag(A_0), 0.0):
        raise ValueError("A_0 diagonal must be zero.")

    return A_0, P_0, R_0
