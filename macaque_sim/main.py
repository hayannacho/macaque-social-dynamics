# main.py -- Entry point / single-run simulation driver
# Assignee: Dev A + Dev B (S3-T1)
# See: project_management/tasks/sprint_3/S3-T1_pipeline_integration.md

"""
Single-run simulation driver. This is the module `python main.py` and Dev E's
Monte Carlo runner both go through.

NOTE (Sprint 3, S3-T1): the pipeline is one macro-step loop that advances (A, P)
via the multi-rate controller, runs Dev D's discrete-event layer (hysteresis +
cooldown), logs state via SimLogger, and fires any scheduled perturbations via
Dev E's injector.

    per macro-step:
        1. multirate_step(A, P, B, cfg, rng) -> (A, P)
        2. check_overthrows(t, P, R, cfg, cooldown) -> R, swap_events
        3. logger.record(t, R, P, A)
        4. apply_perturbation(p, A, P, R, cfg) for any p that should_trigger

running the visualizations: python -m macaque_sim.main --scenario coalition --plot output/coalition_combined.png
"""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np

from macaque_sim.config.scenarios import build_grooming_matrix
from macaque_sim.config.sim_config import SimConfig, default_config, load_config
from macaque_sim.events.cooldown import CooldownTracker
from macaque_sim.events.epsilon_tracker import EpsilonTracker
from macaque_sim.events.hysteresis import check_overthrows
from macaque_sim.events.logger import SimLogger
from macaque_sim.experiments.perturbations import apply_perturbation, should_trigger
from macaque_sim.initializer.mri_init import initialize
from macaque_sim.solvers.multirate import multirate_step


def _coerce_config(config_like: dict[str, Any] | SimConfig | None) -> SimConfig:
    """
    Normalize whatever the caller passed in into a validated SimConfig.

    NOTE: I'm accepting three shapes on purpose so every entry point (CLI, unit
    tests, Dev E's Monte Carlo runner) can hand me whatever's most convenient on
    their side without a bunch of conversion boilerplate at each call-site.

      - SimConfig      -> trust it, pass through.
      - dict           -> treat as partial overrides on top of default_config().
      - None / missing -> just use default_config().

    gotcha: when merging a dict, I have to re-validate() after the merge. If the
    user passes dt_sde or lambda_decay that violate the stability constraint,
    SimConfig.validate raises -- we want that to blow up HERE, not deep inside
    em_step on the first micro-step when the error message is less useful.
    """
    # Default path: no config supplied -> build the standard one Dev C defined.
    if config_like is None:
        return default_config()

    # Already a SimConfig -> trust it. Dev C's validate() was already called at
    # construction time, so we don't re-validate and risk breaking Dev E's
    # scenario overrides which legitimately mutate fields post-construction.
    if isinstance(config_like, SimConfig):
        return config_like

    # Dict of overrides -> merge into defaults, peel off non-SimConfig keys,
    # re-validate, then re-attach the peeled keys as runtime attributes.
    #
    # gotcha: Dev E's MC runner stuffs `perturbations` (a list, not a SimConfig
    # field) into the merged dict so the scenario's mid-sim shocks flow through
    # to run_simulation. SimConfig(**base) would TypeError on that. Filter the
    # dict down to real fields first, then re-glue the extras on the resulting
    # cfg so run_simulation's `getattr(cfg, "perturbations", ...)` still works.
    if isinstance(config_like, dict):
        base = default_config().to_dict()
        base.update(config_like)
        field_names = {f.name for f in dataclass_fields(SimConfig)}
        extras = {k: base.pop(k) for k in list(base) if k not in field_names}
        cfg = SimConfig(**base)
        cfg.validate()
        for k, v in extras.items():
            setattr(cfg, k, v)
        return cfg

    # Anything else is almost certainly a typo or a stale API. Yell.
    raise TypeError(
        f"config_like must be SimConfig, dict, or None; got {type(config_like).__name__}"
    )


def run_simulation(
    config_like: dict[str, Any] | SimConfig | None = None,
    *,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> SimLogger:
    """
    Run a single simulation trajectory end-to-end and return the populated logger.

    This is the function everything else in the pipeline eventually calls -- the CLI
    entry point wraps it, Dev E's Monte Carlo runner shells into it via the
    run_single_trial alias below, and the eventual visualization code consumes the
    SimLogger it returns.

    Parameters
    ----------
    config_like
        SimConfig, override dict, or None (default_config). See _coerce_config.
    initial_state
        Optional pre-built (A, P, R) triple. Overrides the MRI initializer so
        callers can inject a custom starting state -- e.g., after applying an
        orphan_injection perturbation at t=0. Arrays are copied defensively so
        mutations inside the loop don't leak back to the caller.

    Returns
    -------
    SimLogger
        Populated logger containing time, rank, and power history, plus any
        perturbation events that fired during the run. I also bolt on three
        extra attributes (alpha_survived, num_swaps, final_ranks) at the end
        because Dev E's Monte Carlo extraction code expects them.
    """
    # Normalize the caller's config into a SimConfig we can trust for the whole run.
    cfg = _coerce_config(config_like)

    # --- 1. Build initial state -----------------------------------------------
    # Grooming matrix first -- Dev C's scenarios.build_grooming_matrix dispatches
    # on cfg.grooming_strategy to pick uniform / upward_biased / coalition.
    B = build_grooming_matrix(cfg)

    if initial_state is not None:
        # Caller supplied their own (A, P, R) -- validate shapes against cfg.N
        # then copy so we don't mutate their arrays.
        A_in, P_in, R_in = initial_state
        N = cfg.N
        if A_in.shape != (N, N) or P_in.shape != (N,) or np.asarray(R_in).shape != (N,):
            raise ValueError(
                f"initial_state shapes must match cfg.N={N}; got "
                f"A={A_in.shape}, P={P_in.shape}, R={np.asarray(R_in).shape}."
            )
        A = np.asarray(A_in, dtype=float).copy()
        P = np.asarray(P_in, dtype=float).copy()
        R = np.asarray(R_in).copy()
    else:
        # Default path: Dev C+D's MRI initializer. Same code CP1 validated.
        A, P, R = initialize(cfg, B)

    # Snapshot the initial rank vector so the end-of-run summary can tell who
    # was alpha at t=0 regardless of any swaps that happen later.
    R_initial = R.copy()

    # One rng for the whole run, seeded from cfg.random_seed so identical seeds
    # produce identical trajectories. Every em_step call plumbs this through.
    rng = np.random.default_rng(cfg.random_seed)

    # Dev D's logger pre-allocates its arrays from config.T / config.dt_ode, so
    # instantiating it here also fixes the expected number of logged rows.
    logger = SimLogger(cfg)

    # S3-T1: Dev D's cooldown tracker has to live at pipeline scope -- check_overthrows
    # reads/writes its state every step, so it needs to persist across the whole loop.
    cooldown = CooldownTracker(cfg)

    # Adaptive-epsilon tracker (post-review C1). Only instantiated when
    # cfg.epsilon_mode == "adaptive"; otherwise stays None and the discrete-
    # event layer falls back to the static cfg.epsilon (production behavior).
    epsilon_tracker: EpsilonTracker | None = None
    if getattr(cfg, "epsilon_mode", "fixed") == "adaptive":
        epsilon_tracker = EpsilonTracker(
            window_size=getattr(cfg, "epsilon_window", 20),
            factor=getattr(cfg, "epsilon_factor", 2.0),
        )

    # Perturbations may have been attached to the config by Dev E's scenario
    # system. A plain default_config() has none -- use getattr so we don't crash
    # on that case either.
    perturbations = getattr(cfg, "perturbations", None) or []

    # Row 0 of the logger is always the initial condition at t = 0.
    t = 0.0
    logger.record(t, R, P, A)

    # --- 2. Main macro-step loop ----------------------------------------------
    # The logger pre-allocated `expected_steps` rows. Row 0 was just filled with
    # the initial condition above, so we have `expected_steps - 1` real macro-
    # steps of work to do. Using the logger's count instead of recomputing T/dt_ode
    # keeps us consistent with Dev D's bookkeeping.
    num_steps = logger.expected_steps
    for step in range(1, num_steps):
        # Track both the interval endpoints because Dev E's should_trigger uses
        # a half-open (t_prev, t_now] window to fire each perturbation exactly once.
        t_prev = t
        t = step * cfg.dt_ode

        # 1. SDE + ODE: advance (A, P) by one macro-step. multirate_step handles
        #    the M micro-step EM loop and the single RK4 macro call internally,
        #    so from up here it looks like one atomic step.
        A, P = multirate_step(A, P, B, cfg, rng)

        # Adaptive-epsilon: feed the post-step P into the tracker so the next
        # check_overthrows sees the current rolling-window noise floor. The
        # first call seeds the tracker (no delta yet); subsequent calls update
        # the running buffer of P_new - P_old.
        epsilon_override: float | None = None
        if epsilon_tracker is not None:
            epsilon_tracker.update(P)
            epsilon_override = epsilon_tracker.current_epsilon(fallback=cfg.epsilon)

        # 2. Discrete-event layer: Dev D's hysteresis engine looks for any
        #    subordinate whose power now exceeds a dominant's by epsilon, with
        #    cooldowns blocking ping-pong swaps. Any swap it fires mutates R and
        #    bumps both agents onto cooldown for the next cfg.cooldown_duration.
        #
        #    gotcha: check_overthrows returns a NEW R; rebind it, don't rely on
        #    in-place mutation -- it makes a defensive copy internally.
        R, swap_events = check_overthrows(t, P, R, cfg, cooldown, epsilon_override=epsilon_override)
        for ev in swap_events:
            logger.record_event(t, "overthrow", ev)

        # 3. Fire any perturbations whose trigger time lands in (t_prev, t]
        #    BEFORE logging, so the row at t=trigger already reflects the shock
        #    (e.g. alpha_removal -> P[alpha] is 0 at the trigger's logged index,
        #    not one step later). This is the semantic the report + integration
        #    tests assume: "alpha's power drops to 0 at t = T/2".
        for p in perturbations:
            if should_trigger(p, t_prev, t, cfg.T):
                A, P, R, cfg = apply_perturbation(p, A, P, R, cfg)
                if hasattr(cfg, "B"):
                    B = cfg.B
                logger.record_event(t, "perturbation", dict(p))

        # 4. Log the post-step, post-perturbation state (A too, so the
        #    pre-allocated history is populated).
        logger.record(t, R, P, A)

    # --- 3. Attach Monte-Carlo summary metrics --------------------------------
    #
    # NOTE: Dev E's Monte Carlo runner pulls these three attributes off the
    # returned logger. Historically the runner expected `run_simulation` to
    # decorate the logger with them, so I'm doing that here to keep that
    # contract.
    final_ranks = R.tolist()
    # alpha_survived: did the agent that started as alpha (rank 1) still hold rank 1
    # at t = T? We stashed R_initial right after initialize(); compare by agent index.
    initial_alpha = int(np.argmin(R_initial))
    logger.alpha_survived = bool(R[initial_alpha] == 1) # type: ignore
    # num_swaps: count overthrow events in the logger.
    logger.num_swaps = sum(1 for e in logger.swap_events if e.get("type") == "overthrow") # type: ignore
    # final_ranks: just the rank array at t = T, as a list for JSON-friendliness.
    logger.final_ranks = final_ranks # type: ignore

    return logger


def run_single_trial(config_dict: dict[str, Any]) -> SimLogger:
    """Alias used by Dev E's Monte Carlo runner (experiments/monte_carlo.py)."""
    # Dev E's runner hands us a plain dict of overrides, so we go through
    # run_simulation and let _coerce_config do the dict-merge path.
    return run_simulation(config_dict)


def _summarize(logger: SimLogger) -> str:
    """
    Build a short stdout-friendly summary for `python main.py` runs.

    Why : CP2 gate asks for "prints summary stats" at the end of a default run,
    so we need something printable that is informative without dumping the
    whole logger to screen.
    """
    # Pull out the time axis and power history the logger already tracks.
    time_axis = logger.get_time_axis()
    P_hist = logger.get_power_history()

    return "\n".join([
        f"Simulation complete: {logger.step_idx} logged steps, t_final={time_axis[-1]:.3f}",
        f"Final power vector (top 5 by value): "
        f"{np.sort(P_hist[-1])[::-1][:5].round(4).tolist()}",
        f"Events recorded: {len(logger.get_events())}",
    ])


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single macaque simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a JSON/YAML config file. Defaults to default_config().",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["coalition", "upward_biased", "uniform"],
        help="Grooming strategy to use (overrides config file).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save combined bump+centrality PNG to this path.",
    )
    args = parser.parse_args(argv)

    # Either load the user-specified config or fall back to Dev C's default.
    cfg: SimConfig
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    # Apply scenario override if given
    if args.scenario is not None:
        cfg_dict = cfg.to_dict()
        cfg_dict["grooming_strategy"] = args.scenario
        cfg = SimConfig(**cfg_dict)
        cfg.validate()

    logger = run_simulation(cfg)
    print(_summarize(logger))

    # Generate combined visualization
    if args.plot is not None:
        from macaque_sim.visualization.centrality_ts import plot_combined
        plot_combined(logger, save_path=args.plot)
        print(f"Combined plot saved to: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli())
