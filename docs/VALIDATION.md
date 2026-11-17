# Local validation — 2026-11-13

## Portfolio visualization — 2026-11-17

The current public source was rerun on Python 3.12.14. The original suite passed **281 tests**, retaining its six Feller-condition warnings. `examples/plot_strategy_results.py` runs all three strategies with seed 42, N=30, and T=10, exports 1,800 agent-state values, and checks finite normalized power at every logged step. Each strategy logs 20 states through t=9.5; event counts are uniform 0, upward-biased 15, and coalition 6. These are single-seed illustrative outputs, not Monte Carlo findings.

The PNG was rendered with Matplotlib 3.11.2 and visually checked. [Exported trajectories](demo/power-trajectories.csv) and [run settings](demo/run-summary.json) support the figure. Original model files are unchanged; the figure generator is a later addition.

Environment: macOS arm64, Python 3.12. Dependencies are recorded in [constraints-tested.txt](../constraints-tested.txt).

| Check | Observed result |
| --- | --- |
| Original team suite: `python -m pytest -q` | **281 passed**, 6 warnings, no skips, 4.28 s |
| `python examples/contribution_demo.py` | All three strategies passed initial-state and finite normalized power-history assertions |
| `python -m macaque_sim.main --config examples/quickstart.json` | Completed; 20 logged steps, final logged time 9.5, 6 events |
| Original source integrity | All 61 retained unchanged files match the source SHA-256 manifest |
| Clean install | README constrained editable install succeeded in a second empty Python 3.12 virtual environment; contribution demo passed |
| Dependency consistency | `pip check`: no broken requirements |

The six warnings come from original tests exercising states that violate the Feller condition. The model reports that non-negativity in those cases relies on clipping. They are retained warnings, not suppressed failures.

The example uses a configured horizon of 10, but the original logger's final recorded state is at 9.5. This edition preserves that original behavior rather than changing the scientific implementation during portfolio preparation.

This test result replaces neither an independent scientific validation nor a replication of the production Monte Carlo results. The original documents contain differing historical counts (277/280 passing); the count above is the fresh observation from this source edition.

The workflow in `.github/workflows/tests.yml` is prepared but has not run on GitHub. Python versions other than 3.12 and the optional image-export/study extras have not been validated here. The SVG preview does not require image-export extras.
