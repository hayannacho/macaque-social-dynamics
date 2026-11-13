# Contribution evidence and team credit

## Anna Cho — Configuration & Biology Lead

The final report explicitly assigns Anna:

- `SimConfig` and the configuration layer;
- the three grooming-matrix strategies;
- the matrilineal-rank-inheritance initializer;
- the biological cross-reference and tuned-defaults justification from primary literature.

The unchanged source excerpt is in [team-contributions.txt](source/team-contributions.txt). It comes from the **Division of Labor** section of `final_report.tex` in the supplied project snapshot. Module headers identify Dev C in `config/sim_config.py` and `initializer/mri_init.py`; `config/scenarios.py` identifies both Dev E and Dev C. These are documentary attributions, not independently recovered Git commit evidence.

### Suggested review path

1. Read [`SimConfig.validate`](../macaque_sim/config/sim_config.py) and the configuration loader.
2. Compare the three grooming constructors and dispatcher in [`scenarios.py`](../macaque_sim/config/scenarios.py). The surrounding experiment schema/factories are shared team work.
3. Follow [`initialize`](../macaque_sim/initializer/mri_init.py) through matrilines, ranks, power, and alliances.
4. Run [the contribution demo](../examples/contribution_demo.py), then the relevant original team tests.
5. Read [the original biological background](source/biological_background.original.md) as historical interpretation, with the later limitations summarized in the README.

## Team ownership boundaries

| Contributor | Documented lead responsibility |
| --- | --- |
| Samantha | Euler–Maruyama SDE and multirate controller; production sweeps and sensitivity analysis |
| Benyuan | RK4 power dynamics, renormalization and divergence detection; integration work |
| Anna | Configuration, grooming matrices, MRI initializer, biological interpretation and default rationale |
| Aaryan | Hysteresis, cooldown, structured logger and multi-swap behavior |
| Jenny | Scenario schema, perturbations, Monte Carlo runner and production dataset |
| Parisa | Visualization, heatmaps, phase diagram, hysteresis and matriline views |

The report describes overlapping work on experiments/integration. This table identifies lead roles rather than exclusive ownership. Original tests remain team artifacts. Do not describe this simulator as a solo project or claim authorship of every line in a shared file.

## Portfolio wording

> Led the configuration and biological initialization layer of a six-person macaque social-dynamics simulation project, implementing validated model parameters, three grooming-matrix strategies, and matrilineal rank initialization. Connected model assumptions to biological literature and documented calibration-dependent limitations.

## Changes made for this edition

The original simulator and tests are unchanged. The contribution-first README, packaging metadata, isolated dependency constraints, small example, preview SVG, and CI workflow were added for portfolio preparation. These additions should not be represented as original coursework deliverables.
