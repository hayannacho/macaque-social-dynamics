# sim_config.py -- Parameter dataclass + loader
# Assignee: Dev C (S1-T1)
# See: project_management/tasks/sprint_1/S1-T1_config_handler.md

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict
import json

@dataclass
class SimConfig:
    """Central configuration object."""

    # Biological / model parameters
    lambda_decay: float     # alliance decay rate (lambda)
    sigma: float            # noise strength
    epsilon: float          # rank swap threshold margin

    # Simulation time controls
    dt_sde: float           # timestep for alliance SDE
    dt_ode: float           # timestep for power ODE
    T: float                # total simulation time
    cooldown_duration: float  # lockout duration in ODE time units after a conflict (NOT macro-step count; CooldownTracker adds this directly to the ODE time t)

    # Population
    N: int                  # number of individuals

    # Scenario / initialization controls
    grooming_strategy: str = "uniform"
    num_matrilines: int = 4
    alliance_seed_scale: float = 0.05
    random_seed: int = 42
    # Multi-rate solver: SDE micro-steps per ODE macro-step (added for Dev A S2-T2).
    micro_steps_per_macro: int = 5
    # Multi-rate solver: how to fold M EM micro-step matrices into the matrix
    # passed to RK4. Default "max" preserves the production behavior; "mean"
    # was added post-review to probe the envelope-bias confound (see §5.4
    # mean-vs-max coupling experiment in final_report.tex).
    coupling_mode: str = "max"
    # Hysteresis margin mode. Default "fixed" uses cfg.epsilon as a constant;
    # "adaptive" sets epsilon = epsilon_factor * running_std(dP) over the last
    # epsilon_window macro-steps (see final_report.tex §5.4 adaptive-epsilon
    # paragraph). Reverts to cfg.epsilon when the rolling buffer has < 2 samples.
    epsilon_mode: str = "fixed"
    epsilon_window: int = 20
    epsilon_factor: float = 2.0

    def validate(self) -> None:
        """Validate config values."""

        # Type checks
        numeric_fields = [
            self.lambda_decay,
            self.sigma,
            self.epsilon,
            self.dt_sde,
            self.dt_ode,
            self.T,
            self.alliance_seed_scale,
        ]
        if not all(isinstance(x, (int, float)) for x in numeric_fields):
            raise TypeError("Numeric parameters must be int or float.")

        if not isinstance(self.N, int):
            raise TypeError("N must be int.")
        if not isinstance(self.cooldown_duration, (int, float)):
            raise TypeError("cooldown_duration must be int or float.")
        if not isinstance(self.num_matrilines, int):
            raise TypeError("num_matrilines must be int.")
        if not isinstance(self.grooming_strategy, str):
            raise TypeError("grooming_strategy must be string.")
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be int.")
        if not isinstance(self.micro_steps_per_macro, int):
            raise TypeError("micro_steps_per_macro must be int.")
        if not isinstance(self.coupling_mode, str):
            raise TypeError("coupling_mode must be str.")
        if not isinstance(self.epsilon_mode, str):
            raise TypeError("epsilon_mode must be str.")
        if not isinstance(self.epsilon_window, int):
            raise TypeError("epsilon_window must be int.")
        if not isinstance(self.epsilon_factor, (int, float)):
            raise TypeError("epsilon_factor must be int or float.")
        if self.epsilon_window < 1:
            raise ValueError("epsilon_window must be >= 1.")
        if self.epsilon_factor <= 0:
            raise ValueError("epsilon_factor must be > 0.")

        # Value checks
        if self.lambda_decay <= 0:
            raise ValueError("lambda_decay must be > 0.")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0.")

        if self.dt_sde <= 0 or self.dt_ode <= 0:
            raise ValueError("dt must be > 0.")
        if self.T <= 0:
            raise ValueError("T must be > 0.")

        if self.N < 1:
            raise ValueError("N must be >= 1.")
        if self.cooldown_duration < 0:
            raise ValueError("cooldown_duration must be >= 0.")
        if self.micro_steps_per_macro < 1:
            raise ValueError("micro_steps_per_macro must be >= 1.")

        if self.num_matrilines < 1:
            raise ValueError("num_matrilines must be >= 1.")
        if self.num_matrilines > self.N:
            raise ValueError("num_matrilines cannot exceed N.")

        # Troop size constraint
        if not (30 <= self.N <= 50):
            raise ValueError("N must be between 30 and 50.")

        # Stability condition
        if self.dt_sde >= (2.0 / self.lambda_decay):
            raise ValueError("dt_sde must satisfy dt_sde < 2 / lambda_decay.")

        # Allowed grooming strategies
        allowed = {"uniform", "upward_biased", "coalition"}
        if self.grooming_strategy not in allowed:
            raise ValueError(
                f"Invalid grooming_strategy: {self.grooming_strategy}. "
                f"Choose from {sorted(allowed)}."
            )

        allowed_coupling = {"max", "mean"}
        if self.coupling_mode not in allowed_coupling:
            raise ValueError(
                f"Invalid coupling_mode: {self.coupling_mode}. "
                f"Choose from {sorted(allowed_coupling)}."
            )

        allowed_epsilon = {"fixed", "adaptive"}
        if self.epsilon_mode not in allowed_epsilon:
            raise ValueError(
                f"Invalid epsilon_mode: {self.epsilon_mode}. "
                f"Choose from {sorted(allowed_epsilon)}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_config() -> SimConfig:
    """Return baseline configuration."""

    # NOTE (Sprint 4, S4-T4): sigma tuned up 0.05 -> 0.30 based on the S4-T1
    # noise_sweep data (100 MC runs per grid cell, 2026-04-23). At the old
    # sigma=0.05 the alpha survived 100% of runs across every other parameter
    # sweep -- lambda_decay and grooming_strategy both showed zero influence
    # on alpha_survival, which meant the baseline demo was boring AND we had
    # no empirical driver for tuning those other axes. The ticket wants
    # "stable but contestable": >70% survival with rare overthrows. Jenny's
    # noise_sweep shows a clean phase transition from 100% survival at
    # sigma <= 0.21 down to 6% at sigma=0.50; the interesting-but-not-chaotic
    # regime sits at sigma ~= 0.30, where 76% of runs preserve the alpha and
    # 24% produce at least one overthrow. Full rationale + per-cell numbers
    # in project_management/reports/sensitivity_report.md Section 3.
    #
    # lambda_decay and grooming_strategy are unchanged: the data gave me no
    # reason to move them, and leaving "uniform" grooming as default means
    # the baseline demo isn't implicitly taking sides in the coalition-vs-
    # upward_biased question the grooming_bias experiment is designed to ask.
    #
    # Solver stability margins at these tuned values (sim.analysis.sensitivity
    # .validate_solver_stability): dt_sde margin 200x, Feller ratio 2.22,
    # multi-rate identity 1.00 -- all three PASS. Locked in by the regression
    # test `test_default_config_passes_solver_stability`.
    cfg = SimConfig(
        lambda_decay=0.10,
        sigma=0.30,
        epsilon=0.02,
        dt_sde=0.10,
        dt_ode=0.50,
        T=100.0,
        cooldown_duration=5,
        N=30,
        grooming_strategy="uniform",
        num_matrilines=4,
        alliance_seed_scale=0.05,
        random_seed=42,
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> SimConfig:
    """
    Load a config file and override only the keys present in that file.

    Supported:
    - .json
    - .yaml / .yml (PyYAML needs to be installed)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML config files."
            ) from e

        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)

    else:
        raise ValueError("Unsupported config format. Use .json, .yaml, or .yml")

    if payload is None:
        payload = {}

    if not isinstance(payload, dict):
        raise TypeError("Config file must be a dictionary of key-value pairs.")

    base = default_config()
    valid_fields = {f.name for f in fields(SimConfig)}

    unknown = set(payload.keys()) - valid_fields
    if unknown:
        raise KeyError(f"Unknown config field(s): {sorted(unknown)}")

    merged = base.to_dict()
    merged.update(payload)

    cfg = SimConfig(**merged)
    cfg.validate()
    return cfg
