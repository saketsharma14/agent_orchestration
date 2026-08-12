# Report Addendum: Fixes and Repositioning

This addendum documents the changes made to the codebase and report framing
in response to review feedback, and gives revised text for the Related Work
section. It's meant to be read alongside the original report, not to replace
it — Sections 1-4 and the engineering-failure-mode discussion in Section 4
still stand as written.

## 1. Revised Related Work / positioning (replaces the last two paragraphs of Section 2, and should inform a rewritten abstract)

LLM-guided evolutionary program search is now a crowded, active area.
FunSearch and AlphaEvolve established the propose-execute-select loop this
project also uses, at data-center compute scale and across broad program
classes. More directly relevant: Lion, a widely adopted optimizer, was
itself discovered via evolutionary program search over a small space of
gradient transforms, sign operations, and momentum terms — meaning
"discovering optimizers via LLM/program search" is already a solved,
shipped result, not a novel target on its own. A cluster of general-purpose
LLM-driven search frameworks — OpenEvolve, ShinkaEvolve, CodeEvolve, and
AdaEvolve (a UC Berkeley framework released in early 2026) — has since
pushed this further, treating the search process itself as adaptive.
AdaEvolve in particular frames LLM-driven evolution as a hierarchical
optimization problem, using signals from recent progress to decide where to
allocate search budget across a population, and benchmarks against
GEPA/ShinkaEvolve/OpenEvolve (and, where available, AlphaEvolve and human
best-known solutions) across roughly 185 problems with repeated-trial
statistics. That is the methodological bar current work in this space is
held to: multiple independent trials per condition, reported as a
distribution, not one or two runs.

Given that, this project's contribution is not "LLM-guided optimizer
discovery" (done, at a scale and rigor this one-month project cannot match)
and, as originally framed, was not yet "multi-agent role-decomposition
helps" either, since no experiment isolated the effect of role-decomposition
specifically — Section 5.2's ablation isolates *structural* search value,
which is a different, narrower question from the title's "multi-agent"
claim. The honest, defensible contribution is now two separate, narrow
empirical questions, each with its own ablation:

1. **Does allowing structural search (not just hyperparameter tuning) help,
   in this bounded DSL, on this proxy task?** (`scripts/run_ablation.py`,
   N trials, mean/std — this is what Section 5.2 already tested and is
   retained.)
2. **Does splitting the search loop across a Proposer role and a
   periodic-Reflector role beat a single-agent propose-evaluate-refine
   loop, same model, same generation budget?** (`scripts/run_agent_ablation.py`,
   new.) Note this specifically compares Proposer-only vs.
   Proposer+Reflector, not a full three-role comparison, because the
   Critic's LLM pass is disabled for reliability reasons documented in
   Section 4 — the Critic contributes only a deterministic gate in both
   conditions. State this precisely rather than as an unqualified
   "multi-agent vs single-agent" claim.

Neither question was answered by prior work at this DSL/model/proxy-task
scope, which is a legitimate, if narrow, workshop-paper-scale contribution.
It is not a claim to have out-discovered Lion or to compete with
AdaEvolve-class systems, and the abstract/introduction should say so
explicitly rather than implying novelty at the "discovering optimizers with
LLMs" level.

## 2. Fixes made, mapped to feedback

| Feedback point | What was fixed | Where |
|---|---|---|
| 1. Missing multi-agent-vs-single-agent ablation | Added `agent_mode` param to `run_search` (multi = Proposer+Reflector, single = Proposer-only) and a new ablation script | `orchestrator/loop.py`, `scripts/run_agent_ablation.py` |
| 2. Two cherry-picked examples, not a distribution | Added a script that runs N full search runs and reports mean/std/best-of-N + how many runs beat the best baseline | `scripts/run_distribution.py` |
| 3. Lion/SGD −9.3 fitness look like a broken LR range | Root-caused: **not primarily an LR problem** — the fitness function's `NEVER_CONVERGED_STEPS = 10_000` penalty constant was ~53x the run's actual step budget (189 steps), so "converged slowly" scored almost as badly as "diverged." Fixed the constant *and* added a per-family LR sweep as defense-in-depth, since Lion genuinely does need a smaller LR than Adam-family optimizers | `tasks/fitness.py`, `tasks/proxy_cnn.py` (`TOTAL_STEPS`), `search_space/seeds.py` (`SEED_LR_GRID`), `scripts/seed_lr_sweep.py` |
| 4. Fitness function is an opaque scalar | Fitness now returns a decomposed vector (`performance_component`, `cost_component`) in addition to the scalar; `compare_baselines.py` prints both | `tasks/fitness.py`, `scripts/compare_baselines.py` |
| 5. No hardware validation, N=5 ablation is small | Added a real (not proxy) wall-clock/memory benchmark script, runnable `--device cpu` on genuinely edge-scale hardware; ablation scripts now default to configurable N and write full per-trial CSVs | `scripts/hardware_validation.py`, `scripts/run_ablation.py`, `scripts/run_agent_ablation.py` |
| 6. Position against AdaEvolve/Lion explicitly | Section 1 above | `report_addendum.md` (this file) |
| Repo/report appendix mismatch | Repo already contained `search_space/`, `executor/`, `tasks/`, `scripts/` matching the appendix by the time this addendum was written; removed two genuinely stale, unreferenced leftover files (`configs/config.yaml`, `models/local_llm.py`) that didn't match the actual Ollama-based system and would confuse a reader; added `README.md` with exact repro commands and a note to tag the commit that produces reported numbers | repo root |

## 3. What's still not done (be upfront about this in the paper too)

- `run_distribution.py` and both ablation scripts have not been executed
  end-to-end against a live Ollama server as part of producing this
  addendum — they're ready to run, but the resulting numbers (and whether
  the Reflector ablation shows a real effect) are not yet known. Run them
  and report whatever comes out, including a null or negative result for
  the Reflector ablation if that's what happens (see the note at the end of
  `scripts/run_agent_ablation.py`).
- `hardware_validation.py` gives real numbers on whatever machine it's run
  on, but a genuinely edge-scale (Jetson/Raspberry Pi) number is still
  outstanding — the "edge-efficient" framing shouldn't be finalized in the
  title/abstract until that's in hand, or the framing should be softened to
  match what's actually measured (a single consumer GPU vs. a data-center
  cluster, as Section 1.2 already scopes it).
- A statistical significance test for the structural-search ablation
  (Section 5.2/7) is still not implemented — `run_ablation.py` reports
  mean/std and a win/tie/loss count, which supports "consistent, modest,
  non-negative advantage" but not a p-value. Add one if trial count is
  increased enough to make it meaningful (rule of thumb: N ≥ 10-15 per
  condition before a t-test/Wilcoxon result is worth reporting).