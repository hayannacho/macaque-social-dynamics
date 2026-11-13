# Biological Background (S4-T7, Dev C)

**Sprint:** 4
**Author:** Dev C (Anna)
**Status:** Draft, ready for S4-T7 final-report assembly.

This is the biological-background section for the final report. It covers the four pieces S4-T7 asks Dev C for: matrilineal rank inheritance, grooming as biological currency, the emotional-bookkeeping (EMO) framework, and the parameter-justification cross-reference (the latter is also expanded in `sensitivity_report.md` §5–§6 -- this section gives the narrative version).

Voice convention: written in the same first-person teammate-at-whiteboard style as the rest of the project's docs. The intent is that it can be moved into the LaTeX report with minor formatting edits, not rewritten.

---

## 1. Why macaques?

Macaque troops are about as good a real-world testbed for "how do hierarchies stay stable, and what makes them break?" as you can find in primatology. The key observable facts:

1. **Rank is heritable.** Daughters inherit a rank just below their mother in the linear dominance order (Kawamura's matrilineal rank inheritance, established 1958 \cite{kawamura1958matriarchal}). This produces strict, multi-generational lineages -- alpha matriline at the top, omega matriline at the bottom, everyone else in between in a fixed order. A naive observer would predict "matriline rank = forever."
2. **Hierarchies are mostly stable -- but not always.** Multi-decade studies of captive and wild colonies show that the linear order usually holds for years to decades at a time. Dettmer documents the *first* matrilineal overthrow in one captive rhesus troop's 25-year history \cite{dettmer2014reproductive}, and frames the broader literature (Ehardt & Bernstein 1986, Gouzoules 1980, Hambright & Gust 2003 among others, all cited in Dettmer's introduction) as a "handful" of documented cases since the 1960s. When overthrows do happen they are dramatic and costly: in Dettmer's case the deposed matriline suffered 100% infant loss in the year of the coup and didn't reproduce viable offspring again for three years. So the actual pattern is "stable on the order of decades, occasionally and catastrophically restructured."
3. **Grooming is the currency.** Allogrooming (grooming-others) isn't just hygiene -- it's the primary affiliative behavior, and it predicts later coalitionary support in fights \cite{schino2007support, schino2008reciprocation, matheson2000grooming, hodgson2024allogrooming}. If A groomed B yesterday, A is statistically more likely to come to B's defense today.

The central scientific question is: **what makes the rare overthrow happen?** That's what this project models.

## 2. Matrilineal Rank Inheritance (MRI)

Kawamura's 1958 observation \cite{kawamura1958matriarchal} -- daughters slot into the rank just below their mother, creating block-structured matrilines -- has been confirmed across decades of captive macaque colony work \cite{kutsukake2000inheritance, dettmer2014reproductive, oatesobrien2010predictors}. There's a refinement worth knowing: within a matriline, daughters rank in *reverse age order* (the youngest daughter outranks her older sisters), per Chapais 2002 and Silk 2001/2009 (both cited in Dettmer's intro). The mechanism is partly maternal coalitional support during juvenile rank establishment, partly inheritance of the mother's grooming network.

The model encodes this as the initial condition. The MRI initializer (`macaque_sim/initializer/mri_init.py`) builds:

- a strict linear rank order `R[t_0]` with agents grouped into matrilines,
- a power vector `P(t_0)` weighted so the alpha matriline starts with the most centrality,
- an alliance matrix `A(t_0)` seeded so within-matriline ties are stronger than between-matriline ties.

This is the textbook "stable aristocracy" baseline. If the simulator did nothing else (no SDE noise, no overthrows), the linear order would persist forever. The simulator's job is to perturb this baseline and see when it breaks.

## 3. Grooming as biological currency

Allogrooming has measurable physiological benefits to the recipient (parasite removal, stress reduction) and is metabolically costly to the giver \cite{matheson2000grooming, hodgson2024allogrooming, disarbois2024virtual}. Schino's grooming-for-support hypothesis \cite{schino2007support, schino2008reciprocation} treats grooming as an investment whose dividend is later coalitionary aid: if I groom you today, you're more likely to back me up if I get into a fight tomorrow.

Two observable patterns matter for the model:

**Upward-biased grooming.** In stable troops, low-rankers spend disproportionate grooming time on high-rankers. This makes sense from the support hypothesis: grooming a high-ranker buys you protection, even if the high-ranker rarely returns the grooming directly. Schino \cite{schino2007support} documents this pattern across multiple macaque populations.

**Coalition grooming.** When low-rankers preferentially groom *each other* instead of grooming up, they're building a different kind of insurance: collective defense. This pattern is rare in stable troops but shows up in two well-documented contexts:

1. Just after an overthrow, while a new dominant matriline consolidates. Dettmer documents this directly: after the 2009 coup at Poolesville, the attacking Matriline 3 then experienced internal fighting between its largest family units to determine which would assume dominance \cite{dettmer2014reproductive}.
2. When the alpha is absent or weakened (medical removal, injury, age, illness), creating opportunity. Oates-O'Brien's retrospective analysis of 20 captive overthrow events at the California National Primate Research Center identifies absence of the alpha female (typically hospitalized for medical reasons) as the single strongest univariate predictor of overthrows (P=0.031) \cite{oatesobrien2010predictors}. Dettmer's case is consistent: the alpha's eldest daughter was removed from the troop one month before the coup for medical treatment, and the challenging matriline had been growing in population for 1-2 years prior \cite{dettmer2014reproductive}.

(Female-bonded coalitional dynamics are baseline rather than exceptional in some non-cercopithecine primates, but we don't try to model that range here -- this is a macaque-specific framework.)

In the model, both patterns are programmable as `B_ij` (the directed grooming-rate matrix), and the experiments compare what happens to alpha survival under each. The headline result -- 88% survival under upward-biased vs 43% under coalition -- matches the field intuition that coalitional restructuring is the destabilizing mechanism. (Schino's 2007 meta-analysis of 36 studies across 14 primate species \cite{schino2007support} reports a small but significant correlation between grooming and later agonistic support, r=0.157 controlling for kinship -- the empirical foundation for treating grooming as coalitional investment that gets cashed in during fights.)

## 4. Emotional Bookkeeping (EMO) and the lambda parameter

Schino's grooming-for-support framework \cite{schino2007support, schino2008reciprocation} treats individual grooming as a partner-specific investment that can be cashed in later as agonistic support during fights. Schino's 2007 meta-analysis of 36 studies across 14 primate species found a small but significant positive correlation (weighted r=0.157, controlling for maternal kinship) between grooming exchanged and later support \cite{schino2007support}. The natural cognitive extension -- that animals maintain partner-specific affective tallies of past exchanges -- is the "emotional bookkeeping" hypothesis (developed in detail by Aureli & Schaffner 2002 *Behaviour* 139, and Schino & Aureli 2009 *Advances in the Study of Behavior* 39; both are foundational EMO references that should be added to the project bibliography for the final report). The EMO framework actually argues that primate emotional bonds are durable -- they enable long-term partner choice without requiring constant short-term top-up.

For the simulator, two parameters matter:

- **`lambda` -- decay rate of the structural alliance matrix `A_ij`** (not of cognitive memory). In the CIR SDE:
$$
dA_{ij} = (B_{ij} - \lambda A_{ij})\,dt + \sigma \sqrt{A_{ij}}\,dW
$$
The `(B - lambda * A)` term is mean-reversion: alliance mass decays toward equilibrium at rate `lambda` unless reinforced by new grooming `B`. This is structural alliance dynamics, separable from any claim about how primate cognitive bookkeeping works.

- **`sigma` -- stochastic volatility on alliance edges**, biologically representing environmental and demographic unpredictability (food scarcity, illness, group movement, individual variation in grooming bouts).

The CIR (Cox-Ingersoll-Ross) form with the `sqrt(A)` diffusion term is mathematically convenient (it guarantees `A_ij >= 0` for the right Feller condition \cite{feller1951diffusion}) and biologically reasonable (alliance-mass noise scales with alliance strength -- weak ties are quiet, strong ties have more variability).

## 5. Parameter justification: why these defaults are reasonable

The detailed numerical justification is in `sensitivity_report.md` §5–§6. The short version for the report:

- **N=30** is at the small end of macaque troop demographics. Dettmer's NIH troop ran 55-107 individuals \cite{dettmer2014reproductive} and Matheson's field studies of captive cercopithecines run in similar ranges \cite{matheson2000grooming}. We chose N=30 for visual tractability on the bump charts and Monte Carlo wall-clock; doubling to N=60 would be more biologically representative but quadruples the alliance-matrix computation per step. Worth flagging in the discussion as a known limitation.
- **lambda=0.10** corresponds to a ~7-time-unit alliance-matrix half-life. Working convention: 1 time unit ≈ 1 week (chosen so that a 100-time-unit run maps to ~2 simulated years -- the right window for observing rare overthrow events). Under that mapping, the alliance half-life is ~7 weeks. Important caveat: this is the decay rate of the *structural alliance matrix*, not of any cognitive or affective memory (per §4). Schino's grooming-for-support framework \cite{schino2007support, schino2008reciprocation} establishes that grooming functions as repeated partner-specific investment whose support dividends require continued exchange, but doesn't pin down a specific structural-decay timescale. So 7 weeks is in the right qualitative zone but not quantitatively calibrated.
- **sigma=0.30** sits at the inflection point of the noise-vs-survival curve and produces ~76% alpha survival per single 100-time-unit (~ 2-year) run. Dettmer's case study \cite{dettmer2014reproductive} documents the *first* overthrow in their captive troop's 25-year history -- so the empirical "overthrow per troop-year" rate from that one data point is roughly 4% per year, with the broader literature (Ehardt & Bernstein 1986, Hambright & Gust 2003, all cited in Dettmer's intro) suggesting most captive troops go decades without one. Our 76% alpha-survival over a simulated 2-year window puts the alpha at modest-but-real risk, which is roughly the regime Dettmer characterizes as "rare but devastating." The original sigma=0.05 effectively never produced overthrows, which doesn't match Dettmer's framing at all -- a captive colony observed for years should show *some* probability of instability, not perfect stability.
- **epsilon=0.02** is the hysteresis margin -- the power gap that has to be sustained before a swap fires. Calibrated against the noise floor at sigma=0.30 (P_i fluctuations on the order of 0.01) so a momentary noise blip can't trigger a swap. Dettmer's coup itself was a discrete event (March 10, 2009) but it was preceded by 1-2 years of population growth in the challenging matriline, plus the alpha's eldest daughter being removed from the troop one month prior \cite{dettmer2014reproductive}. So the *event* is discrete but the *build-up* is on a months-to-years timescale -- the epsilon margin enforces this in the model by requiring sustained P_sub > P_dom rather than instantaneous threshold crossings.

What I'd flag honestly for the discussion section: the time-unit mapping (1 unit ≈ 1 week) is conventional, chosen for the convenient mapping to "a few simulated years per run." A proper calibration would map `dt_ode` to observed grooming-bout intervals from a published time-budget study. Worth mentioning as a future-work direction so the reader doesn't think we're claiming more empirical grounding than we have.

There's also a useful empirical refinement from Kutsukake's 2000 study of 69 adult females in a Shiga-Heights provisioned Japanese macaque troop \cite{kutsukake2000inheritance}. Kutsukake reports that strict MRI rules hold robustly within high-rank kin units (correlation between predicted and absolute rank r=0.96 in his top kin unit, with only 22% of females disagreeing with strict MRI predictions) but break down in low-rank kin units (r=-0.69 in the fourth-ranked kin unit, with 50% of females disagreeing). Across his sample, 71% of low-ranking females disagreed with strict MRI vs. only 13% of high-rankers, and the proposed mechanism is weaker kin-unit cohesion in low-rank matrilines (low-rank females adopt more individualistic strategies -- alliances with the alpha male, occasional emigration). This is a useful prediction to extract from our model: under noise-driven SDE dynamics with strict-MRI initial conditions, we'd expect rank dispersion to emerge more strongly in low-rank kin units than in high-rank ones. We don't currently report stratified-by-kin-unit dispersion metrics, but it's a natural follow-up that would let us cross-validate against Kutsukake's empirical findings directly.

## 6. What this project adds beyond existing models

Three things distinguish our simulator from prior work:

1. **GrooFiWorld and similar discrete-event simulators** \cite{puga2009groofiworld} update dominance indices on encounter events but don't model continuous-time alliance dynamics. Our hybrid SDE/ODE/discrete-event coupling lets alliance strength evolve continuously between rank events, which matches the biological reality that grooming and affective memory are continuous processes punctuated by occasional discrete fights.

2. **Static and temporal network models** \cite{wasserman1994sna, holme2012temporal, grindrod2014dynamical, ahmad2021pagerank} compute centrality on snapshots and don't have explicit overthrow rules. Our model embeds biologically grounded rank rules (MRI initial conditions, hysteresis-margin overthrows) inside a continuous-time stochastic network, linking network structure to observable hierarchy events.

3. **Eigenvector centrality as a continuous-time observable.** Instead of repeatedly solving eigenvalue problems on static snapshots, we follow Grindrod & Higham \cite{grindrod2014dynamical} and Ahmad et al. \cite{ahmad2021pagerank} in defining a continuous-time ODE whose trajectory is the power redistribution \cite{fushing2011ranking, demoor1994poweriteration}. This gives us a smooth `P(t)` to feed the discrete overthrow rule, rather than a discrete sequence of `P` snapshots.

The final report should make the case that this combination -- biologically grounded initial conditions + continuous-time stochastic alliance dynamics + explicit discrete overthrow rules -- is what lets the model produce the four headline findings (sigma phase transition, coalition-vs-upward gap, the lambda paradox, and the bistability hysteresis loop) that prior approaches couldn't easily generate.

## 7. Citations used in this section

All citation keys used here already exist in `literature_review.tex`: `kawamura1958matriarchal`, `kutsukake2000inheritance`, `dettmer2014reproductive`, `schino2007support`, `schino2008reciprocation`, `matheson2000grooming`, `hodgson2024allogrooming`, `disarbois2024virtual`, `oatesobrien2010predictors`, `feller1951diffusion`, `puga2009groofiworld`, `wasserman1994sna`, `holme2012temporal`, `grindrod2014dynamical`, `ahmad2021pagerank`, `fushing2011ranking`, `demoor1994poweriteration`.

**Two additional references are mentioned in the prose but not yet in the project bibliography** -- both should be added to `egbib.bib` before the LaTeX final report compiles, because they're load-bearing for §3 (upward-biased grooming meta-analysis) and §4 (emotional bookkeeping framework):

- *Schino, G. 2001. Grooming, competition and social rank among female primates: a meta-analysis. Animal Behaviour 62: 265-271.* (Suggested key: `schino2001grooming`. Canonical source for the "primates direct grooming up the hierarchy" meta-analytic finding across 27 social groups in 14 species.)
- *Schino, G. & Aureli, F. 2009. Reciprocal Altruism in Primates: Partner Choice, Cognition, and Emotions. Advances in the Study of Behavior 39: 45-69.* (Suggested key: `schinoaureli2009reciprocal`. Canonical source for the emotional-bookkeeping framework that justifies treating partner choice as long-term affective tracking rather than short-term contingency.)
- (Also worth adding: *Aureli, F. & Schaffner, C.M. 2002. Relationship assessment through emotional mediation. Behaviour 139: 393-420.* as the original EMO paper, suggested key `aurelischaffner2002emotion`. Cited via Schino & Aureli 2009.)

**Verification status (2026-04-26):** Claims attributed to Dettmer 2014, Schino 2007, Kutsukake 2000, and Oates-O'Brien 2010 have been cross-checked against the actual papers and corrected where necessary. Earlier drafts misattributed several claims to Kutsukake 2000 (which is about MRI variation by absolute rank, NOT about overthrow events) -- those have been moved to Dettmer 2014 (post-overthrow consolidation, since Dettmer documents this directly) and Oates-O'Brien 2010 (alpha-female absence as overthrow predictor). The EMO framework attribution has been moved off Schino 2007 (which is a meta-analysis, not the EMO source paper) onto a flag for adding the Schino & Aureli 2009 chapter.
