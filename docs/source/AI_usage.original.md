## Tools used

- **Claude (Anthropic)**
- **GitHub Copilot**

## What we used the tools for

**Project management.** Translating our high-level sprint plan into a
concrete multi-sprint task list (the files under
`project_management/tasks/`), tracking sprint progress, and keeping the
checkpoint integration log up to date.

**Implementation support.** Boilerplate scaffolding (CLI argparse setup,
pytest fixtures, plotting style helpers), debugging help when our tests
caught regressions we couldn't immediately localize, and refactoring
passes after each sprint to keep the module structure clean. The team
designed every numerical algorithm before any AI involvement -- the
Euler-Maruyama / RK4 / multirate / hysteresis logic, the SDE
formulation, the eigenvector ODE, the experimental scenarios -- and AI
was used to help us implement them, not design them.

**Writing support.** Editing for clarity in the markdown reports under
`project_management/reports/`, drafting subsection scaffolds we then
revised, and generating docstrings from already-written code. The
literature review and the conceptual model were team-authored from
papers we read directly.

**Documentation.** Turning the per-sprint task notes into the README
section layout and the file-tree summary.

## Files / sections materially affected

- `macaque_sim/` -- AI-assisted refactoring passes after sprint
  milestones; team designed all numerical methods, schemas, and the
  pipeline architecture. Tests in `macaque_sim/tests/` were team-written
  with AI debugging help where needed.
- `macaque_sim/config/scenarios.py` -- two late-added volatility-spike
  follow-up scenarios (`volatility_spike_moderate`, `volatility_spike_short`)
  for the §5.3 dose-response ladder; AI helped with the parameter
  bookkeeping, team chose the spike severities and durations.
- `macaque_sim/visualization/style.py` -- AI helped catch a bug where
  `get_matriline_ids` used contiguous-block assignment but
  `mri_init.build_matrilines` used cyclic; the two now match.
- `generate_figures.py` -- redesigned the baseline figure (rank heatmap +
  matriline-summed power) and the volatility-spike figure (4-bar
  severity ladder) at final-pass; AI helped with plot layout and
  text-positioning details for zero-height bars.
- `project_management/tasks/` -- task breakdown drafts authored with
  AI help, then edited by the team.
- `project_management/reports/` -- markdown report drafts (numerical
  methods, eigenvector centrality, discrete event logic, biological
  background, experimental design, sensitivity, results) used AI for
  initial scaffolds and editing passes; team owns all factual content
  and verified all citations.
- `final_report.tex` -- assembled from the per-section markdown drafts.
  AI helped with the late §5.4 robustness-and-follow-on-studies section
  (seven studies covering swap-event distributions, MRI-vs-random init,
  coalition-size sensitivity, ε sweep, heterogeneous σ); team chose
  which studies to run, ran them, and wrote the interpretive prose.
- `final_demo.ipynb` -- promoted from `checkpoint2_demo.ipynb` to be
  the canonical end-to-end walkthrough; §16-§22 production-MC analysis
  cells were AI-drafted, team verified outputs against `output/results/`
  JSONs.
- `egbib.bib` -- citations were each verified by team members against
  the actual papers (DOIs, page numbers, author lists). Two rounds of
  verification: the first eight entries on 2026-04-26, the remaining
  entries on 2026-04-27.
- `literature_review.tex` -- team-authored from primary sources.

## How we verified correctness

- The full pytest suite (`python -m pytest macaque_sim/tests/`) is part
  of CI and currently passes 277 tests with 1 skipped (278 collected).
  Every numerical module has unit tests we authored.
- For numerical claims (the Feller condition, the Euler stability
  bound, the RK4 stability disk, the multi-rate identity), team members
  derived the math by hand and cross-checked against textbook
  references (Higham 2001, Kloeden \& Platen 1992).
- For biological claims, citations were verified against PDFs of the
  primary papers. Each entry in `egbib.bib` carries a "VERIFIED" line
  noting when the lookup was done.

We take responsibility for the correctness and originality of the
project as submitted.
