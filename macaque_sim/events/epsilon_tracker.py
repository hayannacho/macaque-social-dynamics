# epsilon_tracker.py -- Adaptive hysteresis margin tracker (post-review).
#
# Maintains a rolling buffer of per-agent power-vector deltas across the last
# K macro-steps and exposes a current epsilon value that is `factor` times the
# pooled standard deviation of those deltas.
#
# This was added in response to the post-submission review's "single most
# actionable fix to rescue the grooming-bias gap": calibrate epsilon as
# 2 * running std(dP) over the last K macro-steps, separately per run, so the
# hysteresis margin tracks the actual noise floor of dP_i in the current
# scenario rather than being pinned to a value calibrated on a uniform-grooming
# reference run. See final_report.tex section 5.4 (adaptive-epsilon paragraph).

from __future__ import annotations

from collections import deque

import numpy as np


class EpsilonTracker:
    """
    Rolling-window estimator of the per-agent power-delta noise floor.

    Usage:
        tracker = EpsilonTracker(window_size=20, factor=2.0)
        for step in range(num_steps):
            P = step_dynamics(...)
            tracker.update(P)
            eps = tracker.current_epsilon(fallback=cfg.epsilon)
            ... use eps in the discrete-event check ...

    The tracker stores the last `window_size` macro-step delta vectors
    P_new - P_old. `current_epsilon` returns `factor * std(pooled deltas)`,
    falling back to the supplied default value when fewer than 2 samples are
    available (since a 1-sample std is undefined).
    """

    def __init__(self, window_size: int = 20, factor: float = 2.0) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1.")
        if factor <= 0:
            raise ValueError("factor must be > 0.")
        self.window_size = int(window_size)
        self.factor = float(factor)
        self._deltas: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._last_P: np.ndarray | None = None

    def update(self, P_new: np.ndarray) -> None:
        """Push the latest power vector. The first call seeds the baseline; each
        subsequent call appends `P_new - P_last` to the rolling buffer."""
        P_arr = np.asarray(P_new, dtype=float)
        if self._last_P is not None:
            self._deltas.append(P_arr - self._last_P)
        self._last_P = P_arr.copy()

    def current_epsilon(self, fallback: float) -> float:
        """Return 2 * std(pooled deltas) over the rolling window, or `fallback`
        when the buffer has fewer than 2 entries (insufficient to estimate std).
        """
        if len(self._deltas) < 2:
            return float(fallback)
        pooled = np.concatenate(list(self._deltas))
        # Use ddof=1 (sample std) to be consistent with how the report
        # describes the original epsilon = 2 * std(dP) calibration.
        sigma = float(np.std(pooled, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0.0:
            return float(fallback)
        return self.factor * sigma
