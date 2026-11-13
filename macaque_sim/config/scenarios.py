# scenarios.py — Predefined experiment scenarios + grooming matrix strategies
# Assignee: Dev E (S1-T7), Dev C (S1-T2)
# See: project_management/tasks/sprint_1/S1-T7_scenario_schema.md
# See: project_management/tasks/sprint_1/S1-T2_grooming_matrix.md


"""
scenarios.py — Experiment scenario schema and factory functions.
 
Defines the ExperimentScenario dataclass and the four mandatory experiments:
  1. scenario_noise()          — σ sweep (social noise)
  2. scenario_grooming_bias()  — B_ij strategy comparison
  3. scenario_decay()          — λ sweep (aristocratic decay)
  4. scenario_alpha_removal()  — mid-sim alpha removal perturbation
 
Also provides:
  load_scenario(path)  → ExperimentScenario   (from JSON file)
  save_scenario(scenario, path)               (to JSON file)
 
Consumed by: Monte Carlo runner (experiments/monte_carlo.py) — Dev E.
Depends on:  SimConfig field names (config/sim_config.py) — Dev C.
 
Author: Dev E (Jenny)

AI Usage: Asked AI to generate docstrings and comments for the code. After completing logic, also asked for it to refactor.
Also asked AI to cross check with other files to make sure I wasn't missing a param/function/method/class/etc.

"""
 
from __future__ import annotations
 
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Union

from macaque_sim.config.sim_config import SimConfig
SweepOrValue = Union[float, int, str, tuple]
 
 
# ---------------------------------------------------------------------------
# ExperimentScenario dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class ExperimentScenario:
    """
    A fully self-contained description of one experiment.
 
    Fields
    ------
    name : str
        Short identifier used in file names and log headers (e.g. "noise_sweep").
 
    description : str
        Human-readable explanation of what this experiment tests.
 
    param_overrides : dict[str, SweepOrValue]
        Keys are SimConfig field names.  Values are either:
          - A raw scalar  →  that parameter is fixed at this value for the run.
          - A 3-tuple (start, stop, num_steps)  →  the Monte Carlo runner
            expands this into np.linspace(start, stop, num_steps) and runs
            one batch of MC trials per grid point.
 
    perturbations : list[dict]
        Zero or more mid-simulation events.  Each dict must contain:
          "type"   : str   — perturbation kind (e.g. "alpha_removal")
          "time"   : float — simulation time at which to trigger
          "target" : int   — agent index affected (use -1 for "highest ranked")
          "params" : dict  — any additional keyword args passed to the
                             perturbation injector (experiments/perturbations.py)
 
    num_mc_runs : int
        Number of independent Monte Carlo repetitions per parameter grid point.
        Defaults to 100.
    """
 
    name: str
    description: str
    param_overrides: dict = field(default_factory=dict)
    perturbations: list = field(default_factory=list)
    num_mc_runs: int = 100
 
    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
 
    def is_sweep(self, key: str) -> bool:
        """Return True if param_overrides[key] is a numeric or categorical sweep."""
        val = self.param_overrides.get(key)
        if not isinstance(val, (tuple, list)):
            return False
        if len(val) == 3:
            start, stop, num_steps = val
            return isinstance(start, (int, float)) and isinstance(stop, (int, float)) and isinstance(num_steps, int)
        if len(val) >= 2 and all(isinstance(x, str) for x in val):
            return True
        return False
 
    def sweep_keys(self) -> list[str]:
        """Return all keys in param_overrides that are sweep ranges."""
        return [k for k in self.param_overrides if self.is_sweep(k)]
 
    def fixed_keys(self) -> list[str]:
        """Return all keys in param_overrides that are fixed scalar values."""
        return [k for k in self.param_overrides if not self.is_sweep(k)]
 
    def validate(self, valid_param_keys: set[str] | None = None) -> None:
        """
        Raise ValueError if the scenario is malformed.
 
        Checks:
          - name and description are non-empty strings.
          - Each sweep tuple has exactly 3 elements and num_steps >= 1.
          - Each perturbation dict contains required keys.
          - num_mc_runs >= 1.
          - If valid_param_keys is provided, all param_overrides keys are valid.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ExperimentScenario.name must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("ExperimentScenario.description must be a non-empty string.")
        if not isinstance(self.num_mc_runs, int) or self.num_mc_runs < 1:
            raise ValueError(f"num_mc_runs must be a positive integer; got {self.num_mc_runs!r}.")
 
        for key, val in self.param_overrides.items():
            if valid_param_keys is not None and key not in valid_param_keys:
                raise ValueError(
                    f"Unknown param_overrides key '{key}'. "
                    f"Valid SimConfig fields: {sorted(valid_param_keys)}."
                )
            if isinstance(val, (tuple, list)):
                if len(val) == 3:
                    start, stop, num_steps = val
                    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
                        raise ValueError(
                            f"Numeric sweep for '{key}' must use numeric (start, stop, num_steps)."
                        )
                    if not isinstance(num_steps, int) or num_steps < 1:
                        raise ValueError(
                            f"num_steps for '{key}' must be a positive integer; got {num_steps!r}."
                        )
                elif len(val) >= 2 and all(isinstance(x, str) for x in val):
                    # Categorical sweep (e.g., grooming strategy variants)
                    pass
                else:
                    raise ValueError(
                        f"Sweep for '{key}' must be numeric (start, stop, num_steps) or "
                        f"a categorical string list/tuple with at least 2 values."
                    )
 
        required_perturbation_keys = {"type", "time", "target", "params"}
        for i, p in enumerate(self.perturbations):
            missing = required_perturbation_keys - set(p.keys())
            if missing:
                raise ValueError(
                    f"Perturbation[{i}] is missing required keys: {missing}."
                )


def simconfig_field_names() -> set[str]:
    """Return all field names defined on SimConfig."""
    return {f.name for f in fields(SimConfig)}
 
 
# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------
 
def _scenario_to_dict(scenario: ExperimentScenario) -> dict:
    """
    Convert an ExperimentScenario to a JSON-serializable dict.
    Tuples in param_overrides are stored as lists (JSON has no tuple type).
    """
    d = asdict(scenario)
    # asdict already converts tuples → lists, which is what we want for JSON
    return d
 
 
def _scenario_from_dict(d: dict) -> ExperimentScenario:
    """
    Reconstruct an ExperimentScenario from a plain dict (e.g. from JSON).
    Lists of length 3 in param_overrides are re-interpreted as sweep tuples.
    """
    overrides = {}
    for key, val in d.get("param_overrides", {}).items():
        if isinstance(val, list) and len(val) == 3:
            overrides[key] = tuple(val)   # restore sweep tuple
        else:
            overrides[key] = val
 
    return ExperimentScenario(
        name=d["name"],
        description=d["description"],
        param_overrides=overrides,
        perturbations=d.get("perturbations", []),
        num_mc_runs=d.get("num_mc_runs", 100),
    )
 
 
def save_scenario(scenario: ExperimentScenario, path: str | Path) -> None:
    """
    Serialize an ExperimentScenario to a JSON file.
 
    Parameters
    ----------
    scenario : ExperimentScenario
    path     : str or Path — destination file (will be created or overwritten).
    """
    scenario.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_scenario_to_dict(scenario), f, indent=2)
 
 
def load_scenario(path: str | Path) -> ExperimentScenario:
    """
    Load an ExperimentScenario from a JSON file produced by save_scenario().
 
    Parameters
    ----------
    path : str or Path — source file.
 
    Returns
    -------
    ExperimentScenario
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    scenario = _scenario_from_dict(d)
    scenario.validate(valid_param_keys=simconfig_field_names())
    return scenario
 
 
# ---------------------------------------------------------------------------
# Factory functions — four mandatory experiments
# ---------------------------------------------------------------------------
 
def scenario_noise(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Experiment 1 — Social Noise (σ sweep).
 
    Scientific question: What happens when environmental unpredictability increases?
    We sweep σ from 0.01 (low noise) to 1.0 (high noise) in 11 steps and measure
    how rank stability collapses as stochastic alliance fluctuations grow.
 
    σ controls the diffusion term σ·√A_ij·dW in the CIR SDE.
    """
    return ExperimentScenario(
        name="noise_sweep",
        description=(
            "Sweep the SDE noise parameter σ from 0.01 to 1.0 across 11 grid points. "
            "Measures how increasing environmental unpredictability destabilizes the "
            "dominance hierarchy by amplifying stochastic alliance fluctuations. "
            "Note: σ=0 is excluded because SimConfig.validate() enforces sigma > 0."
        ),
        param_overrides={
            "sigma": (0.01, 1.0, 11),   
        },
        perturbations=[],
        num_mc_runs=num_mc_runs,
    )
 
 
def scenario_grooming_bias(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Experiment 2 — Grooming Bias (B_ij strategy comparison).
 
    Scientific question: What strategy is more effective for low-ranking monkeys —
    grooming the alpha directly ("upward grooming"), or grooming peers to form a
    coalition ("coalition grooming")?
 
    The B_ij matrix encodes directed grooming rates.  This experiment tags the
    strategy type so the Monte Carlo runner can swap in different B_ij matrices
    (defined in perturbations.py or the config layer) and compare overthrow rates
    across strategies.
 
    param_overrides["grooming_strategy"] is a string tag consumed by the
    perturbation injector to select the appropriate B_ij matrix at runtime.
    """
    return ExperimentScenario(
        name="grooming_bias",
        description=(
            "Compare two low-ranker grooming strategies: (1) upward_bias — "
            "subordinates direct grooming toward the alpha; (2) coalition — "
            "subordinates groom each other to accumulate shared power. "
            "Measures overthrow rate and time-to-overthrow under each strategy."
        ),
        param_overrides={
            "grooming_strategy": ("upward_biased", "coalition"),
        },
        perturbations=[],
        num_mc_runs=num_mc_runs,
    )
 
 
def scenario_decay(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Experiment 3 — Aristocratic Decay (λ sweep).
 
    Scientific question: How quickly must alliances be maintained to preserve power?
    We sweep the memory decay parameter λ from 0.05 (slow decay, alliances persist)
    to 1.0 (fast decay, constant grooming required) across 20 grid points.
 
    λ controls the drift term (B_ij − λ·A_ij)·dt in the CIR SDE.
    High λ forces continuous grooming investment just to maintain existing alliances.
    """
    return ExperimentScenario(
        name="decay_sweep",
        description=(
            "Sweep the alliance memory decay parameter λ from 0.05 to 1.0 across "
            "20 grid points. Measures how rapidly alliances must be reinforced to "
            "sustain dominance, and at what decay rate hierarchy stability collapses."
        ),
        param_overrides={
            "lambda_decay": (0.05, 1.0, 20),   # sweep: slow decay → fast decay
        },
        perturbations=[],
        num_mc_runs=num_mc_runs,
    )
 
 
def scenario_phase(
    num_mc_runs: int = 50,
    sigma_steps: int = 7,
    lambda_steps: int = 7,
) -> ExperimentScenario:
    """
    Experiment 5 — Phase Diagram (2D σ × λ sweep).

    Scientific question: How do noise and memory decay jointly control aristocracy
    stability? This is the central figure of the project: one 1D curve can't
    distinguish "noise dominates" from "λ dominates" from "they trade off".

    A full 2D Monte Carlo sweep over (σ, λ) produces a survival-rate heatmap
    where the 50% contour marks the stable ↔ unstable transition. Both axes
    match scenario_noise and scenario_decay ranges so 1D slices of the grid
    agree with the dedicated 1D sweeps.
    """
    return ExperimentScenario(
        name="phase_sweep",
        description=(
            f"2D parameter sweep over σ (social noise, {sigma_steps} steps in "
            f"[0.01, 1.0]) and λ (aristocratic memory decay, {lambda_steps} "
            "steps in [0.05, 1.0]). Each grid cell runs num_mc_runs independent "
            "trials; the resulting alpha-survival-rate heatmap is the phase "
            "diagram delivered by S4-T2."
        ),
        param_overrides={
            "sigma":        (0.01, 1.0, sigma_steps),
            "lambda_decay": (0.05, 1.0, lambda_steps),
        },
        perturbations=[],
        num_mc_runs=num_mc_runs,
    )


def scenario_alpha_removal(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Experiment 4 — Alpha Removal (mid-simulation perturbation).
 
    Scientific question: What happens when the top-ranked individual is suddenly
    removed at the midpoint of the simulation?
 
    At t = T/2, the alpha's row and column in A(t) are zeroed out, simulating
    death, transfer, or removal in a captive breeding context.  The system must
    re-stabilize and we measure time-to-restabilization and which matriline
    ascends to dominance.
 
    Uses target=-1 as a sentinel meaning "agent with rank 1 (current alpha)
    at time of trigger" so the perturbation injector can resolve the actual
    agent index dynamically.
    """
    return ExperimentScenario(
        name="alpha_removal",
        description=(
            "Remove the alpha individual (rank 1) at t = T/2 by zeroing their "
            "row and column in the alliance matrix A(t). Measures time-to-"
            "restabilization and the identity of the successor matriline. "
            "Mimics real primate management scenarios (death, transfer, capture)."
        ),
        param_overrides={},           # no parameter sweep — single fixed config
        perturbations=[
            {
                "type":   "alpha_removal",
                "time":   -1,          # sentinel: resolved to T/2 at runtime
                "target": -1,          # sentinel: resolved to rank-1 agent at trigger
                "params": {
                    "zero_row": True,
                    "zero_col": True,
                    "note": "time=-1 means T/2; target=-1 means current alpha (rank 1)",
                },
            }
        ],
        num_mc_runs=num_mc_runs,
    )


def scenario_volatility_spike_moderate(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Follow-up to scenario_volatility_spike: spike sigma to 0.40 (just past the
    transition) instead of 0.90 (well above it), same half-run duration.

    Why: the original sigma=0.90 spike result is overdetermined -- at that
    sigma the noise sweep reports 4% baseline survival, so half a simulation
    at sigma=0.90 essentially guarantees deposition during the spike. That
    makes it a sufficient-condition demonstration rather than a tight test
    of the deposed-basin claim. sigma=0.40 sits just past the transition
    where dispossession is partial (the noise sweep reports 30% survival at
    sigma=0.41 over a full T=100 window), so this scenario is a dose-
    response point on the spike-severity curve.
    """
    return ExperimentScenario(
        name="volatility_spike_moderate",
        description=(
            "Transient noise shock at moderate severity: sigma jumps from "
            "0.30 to 0.40 (just past the transition) at t = T/4 and returns "
            "to 0.30 at t = 3T/4. Same half-run spike duration as the "
            "scenario_volatility_spike scenario but at a smaller spike size."
        ),
        param_overrides={},
        perturbations=[
            {
                "type":   "volatility_spike",
                "time":   25.0,
                "target": -1,
                "params": {"factor": 4.0 / 3.0, "note": "sigma 0.30 -> 0.40 (just past transition)"},
            },
            {
                "type":   "volatility_spike",
                "time":   75.0,
                "target": -1,
                "params": {"factor": 3.0 / 4.0, "note": "sigma 0.40 -> 0.30 (back to baseline)"},
            },
        ],
        num_mc_runs=num_mc_runs,
    )


def scenario_volatility_spike_short(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Follow-up to scenario_volatility_spike: shorter spike, longer recovery
    window. Spike is sigma=0.40 over a quarter of the run [T/4, T/2], then
    sigma returns to 0.30 for the remaining half [T/2, T].

    Why: the bistability question is whether deposed alphas recover after
    sigma returns to baseline. The original scenario gives the system only
    T/4 = 25 time units of recovery, which at production swap rates is on
    the order of a single swap event -- not enough for any deposed alpha
    to climb back even by random walk. Doubling the recovery window to
    T/2 = 50 time units is on the order of 5 swap events, which is enough
    to detect whether deposed alphas climb back at the random-permutation
    rate (~1/N = 3.3%) or stay at 0%.
    """
    return ExperimentScenario(
        name="volatility_spike_short",
        description=(
            "Short transient shock with long recovery window: sigma jumps "
            "from 0.30 to 0.40 at t = T/4 and returns to 0.30 at t = T/2. "
            "Recovery window is the second half [T/2, T] of the run."
        ),
        param_overrides={},
        perturbations=[
            {
                "type":   "volatility_spike",
                "time":   25.0,
                "target": -1,
                "params": {"factor": 4.0 / 3.0, "note": "sigma 0.30 -> 0.40 (spike start)"},
            },
            {
                "type":   "volatility_spike",
                "time":   50.0,
                "target": -1,
                "params": {"factor": 3.0 / 4.0, "note": "sigma 0.40 -> 0.30 (recovery start, T/2 onward)"},
            },
        ],
        num_mc_runs=num_mc_runs,
    )


def scenario_volatility_spike(num_mc_runs: int = 50) -> ExperimentScenario:
    """
    Experiment 6 — Transient noise shock (bistability probe).

    Scientific question: if we briefly crank social noise sigma well above the
    overthrow threshold and then return it to baseline, does the hierarchy
    snap back, or does it stay in the deposed basin? This is the time-resolved
    version of the hysteresis-loop quasi-static ramp.

    Mechanism: two volatility_spike perturbations. The first fires at t = T/4
    and multiplies sigma by 3 (0.30 -> 0.90, well into the overthrow regime
    per the noise sweep). The second fires at t = 3T/4 with factor 1/3 to
    return sigma to its original 0.30. The 50 time units of high-sigma
    operation (T/4 -> 3T/4) is enough to trigger overthrows in trials where
    the system is poised on the boundary; the question the experiment answers
    is whether those trials recover their original alpha after sigma returns.
    """
    return ExperimentScenario(
        name="volatility_spike",
        description=(
            "Transient noise shock: sigma jumps from 0.30 to 0.90 at t = T/4 "
            "and returns to 0.30 at t = 3T/4. Tests bistability — does the "
            "system stay in the deposed basin after noise returns to baseline?"
        ),
        param_overrides={},           # single-cell scenario, like alpha_removal
        perturbations=[
            {
                "type":   "volatility_spike",
                "time":   25.0,        # T/4 with default T = 100
                "target": -1,          # not used by volatility_spike, but required field
                "params": {"factor": 3.0, "note": "sigma 0.30 -> 0.90 (above overthrow threshold)"},
            },
            {
                "type":   "volatility_spike",
                "time":   75.0,        # 3T/4
                "target": -1,
                "params": {"factor": 1.0 / 3.0, "note": "sigma 0.90 -> 0.30 (back to baseline)"},
            },
        ],
        num_mc_runs=num_mc_runs,
    )


# ---------------------------------------------------------------------------
# Grooming matrix strategies — Dev C (Anna; S1-T2)
# ---------------------------------------------------------------------------

import numpy as np

def uniform_grooming(N: int, rate: float = 0.1) -> np.ndarray:
    """
    Baseline grooming matrix: every monkey grooms every other monkey at the same rate.

    Returns:
        (N, N) ndarray, non-negative matrix with zero diagonal.
    """
    B = np.full((N, N), rate, dtype=float)
    np.fill_diagonal(B, 0.0)
    return B


def upward_biased_grooming(
    N: int,
    ranks: np.ndarray,
    high_rate: float = 0.2,
    low_rate: float = 0.05,
) -> np.ndarray:
    """
    Subordinates preferentially groom higher-ranked monkeys.

    Convention:
    - smaller rank number = higher status (e.g. rank 1 = alpha)

    If monkey j outranks monkey i, then i -> j gets high_rate.
    Otherwise i -> j gets low_rate.
    """
    B = np.full((N, N), low_rate, dtype=float)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if ranks[j] < ranks[i]:
                B[i, j] = high_rate

    np.fill_diagonal(B, 0.0)
    return B


def coalition_grooming(
    N: int,
    ranks: np.ndarray,
    clique: list[int],
    clique_rate: float = 0.25,
    background_rate: float = 0.05,
) -> np.ndarray:
    """
    Construct a coalition grooming matrix.

    Clique members groom one another more than background, and grooming within the clique is directed;
    Lower-ranked clique members groom higher-ranked clique partners more strongly.
    """
    B = np.full((N, N), background_rate, dtype=float)

    for i in clique:
        for j in clique:
            if i == j:
                continue
            
            # Lower-ranked clique members groom higher-ranked clique partners more strongly.
            if ranks[j] < ranks[i]:
                B[i, j] = clique_rate
            else:
                B[i, j] = background_rate + 0.05

    np.fill_diagonal(B, 0.0)
    return B


def build_grooming_matrix(
    config,
    ranks: np.ndarray | None = None,
    clique: list[int] | None = None,
) -> np.ndarray:
    """
    Dispatch grooming matrix construction based on config.grooming_strategy.

    Parameters:
        config : SimConfig-like object
            Must provide at least:
            - N
            - grooming_strategy
        ranks : np.ndarray | None
            Optional rank array, shape (N,), where smaller number means higher rank.
            If None, uses a simple baseline ranking 1..N.
        clique : list[int] | None
            Optional coalition member indices for coalition strategy.

    Returns:
        np.ndarray of shape (N, N)
    """
    N = config.N
    strategy = config.grooming_strategy

    if ranks is None:
        # Fallback baseline ranking:
        # monkey 0 has rank 1, monkey 1 has rank 2, ..., monkey N-1 has rank N
        ranks = np.arange(1, N + 1)

    if strategy == "uniform":
        B = uniform_grooming(N)

    elif strategy == "upward_biased":
        B = upward_biased_grooming(N, ranks)

    elif strategy == "coalition":
        if clique is None:
            # Coalition size: ~20% of N, but at least 3 members
            clique_size = max(3, N // 5)
            # Default clique = lowest-ranked individuals
            clique = list(range(N - clique_size, N))
            
        B = coalition_grooming(N, ranks, clique)

    else:
        raise ValueError(
            f"Unknown grooming_strategy: {strategy}. "
            f"Choose from ['uniform', 'upward_biased', 'coalition']."
        )

    # Safety checks
    if B.shape != (N, N):
        raise ValueError(f"Grooming matrix must have shape ({N}, {N}), got {B.shape}.")
    if np.any(B < 0):
        raise ValueError("Grooming matrix cannot contain negative entries.")

    np.fill_diagonal(B, 0.0)
    return B
