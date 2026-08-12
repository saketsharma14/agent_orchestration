# agent_orchestration

**Multi-Agent LLM-Guided Discovery of Edge-Efficient Optimizers — full
project history, from initial build through external review through the
revised, ablation-backed results.**

This README is the single source of truth for the project: what we set out
to do, what we built, what a review caught, what we fixed, and what the
final (honestly reported, including a negative result) numbers are. It
supersedes the original standalone report — that document is preserved at
`report_v1_original.md` for reference, and the corrected version's full
text is folded into Section 4 below.

---

## TL;DR

- **Goal:** use a team of LLM agents (Proposer / Critic / Reflector) to
  search for new gradient-based optimizers, in a bounded design space,
  under simulated edge-deployment (low memory/compute) constraints.
- **v1** built the system, got promising-looking numbers, and claimed both
  that structural search helps and that the multi-agent design helps.
- **External review** found the multi-agent claim was never actually
  tested, two baseline numbers looked broken, results were cherry-picked
  rather than a fair distribution, and the fitness function was an opaque
  scalar.
- **v2** fixed a real fitness-function bug, added the missing ablation, ran
  everything over 5 trials instead of 1-2, and validated the edge-cost
  proxy against real hardware measurement.
- **Bottom line, honestly:** structural search shows a small, consistent
  benefit (supported). Multi-agent role-decomposition does **not** show a
  benefit over a single agent doing the same job, and trails slightly
  (not supported — reported as a negative result, not hidden). The
  discovered candidate's "fewer memory buffers" advantage does not
  translate into being faster in real wall-clock terms — it's 44% slower
  than AdamW on CPU.

---

## 1. What we set out to do

Algorithm-discovery systems like FunSearch and AlphaEvolve show that
LLM-guided evolutionary search over program space can rediscover and
improve on hand-designed algorithms. Separately, training on edge devices
(phones, embedded boards, single consumer GPUs) has memory/compute
constraints that hand-designed optimizers like Adam weren't built around.

This project asked two specific questions:

1. Does letting an LLM-guided search change optimizer *structure* (not
   just hyperparameters) find better optimizers than tuning hyperparameters
   alone?
2. Does splitting the search loop across specialized agent roles (a
   Proposer that suggests candidates, a Reflector that periodically
   summarizes what's working) do better than one agent doing the whole
   loop itself?

Scope, deliberately, for a one-month project: one proxy task (MNIST-subset
CNN), a bounded DSL containing Adam/AdamW/RMSprop/SGD-momentum/Lion as
special cases, simulated (not measured) edge-cost proxies, and a single
locally-served 7-8B LLM shared across all agent roles.

## 2. What we built (v1)

- **Search-space DSL** (`search_space/dsl.py`): five structural fields
  (`grad_transform`, `second_moment`, `normalization`, `bias_correction`,
  `weight_decay_mode`) plus five numeric fields (learning rate, betas,
  epsilon, weight decay). A fixed code generator renders any spec into a
  real `torch.optim.Optimizer` — agents never write raw code, which bounds
  the search space and removes a whole class of code-injection risk.
- **Three-role agent system** (`agents/`): Proposer (mutates/recombines
  archive entries into new candidates), Critic (validates before the
  expensive training run — currently deterministic-only, see Section 3),
  Reflector (periodically summarizes recent history into guidance for the
  Proposer), sharing one Ollama-served model with per-role sampling
  settings (`models/role_config.py`).
- **Evolutionary archive** (`orchestrator/archive.py`): MAP-Elites-style,
  bucketed by structural signature, so search doesn't collapse onto one
  structural family the moment it scores well early.
- **Sandboxed evaluation** (`executor/sandbox.py`, `tasks/proxy_cnn.py`):
  each candidate trains a small CNN on a 4,000-example MNIST subset for 3
  epochs in an isolated subprocess with a timeout, scored by a fitness
  function combining accuracy/convergence-speed with static memory/compute
  cost proxies derived from the candidate's structure.

Along the way, v1 diagnosed and fixed several genuine LLM-agent engineering
failure modes worth keeping on record: premature convergence to a single
structural family (fixed via the niche-based archive), a Reflector that
started hallucinating semantics into the candidate's free-text `name`
field (fixed by making `name` explicitly non-semantic), a Proposer/Critic
constraint mismatch that wasted generations on repeated rejections (fixed
by moving the constraint into a deterministic check both roles share), and
crash sensitivity to malformed LLM JSON output (fixed with a repair pass
and per-generation archive persistence).

**v1 results, as originally reported:** two hand-picked full-search runs
where discovered candidates beat fixed baselines, plus one real ablation —
structural-vs-numeric-only search, N=5 trials, showing a modest consistent
advantage for structural search. The title and framing also claimed a
"multi-agent" advantage, but **no corresponding ablation existed** to test
that specifically.

## 3. What external review caught

An outside, knowledgeable review of the v1 report and repo flagged, in
order of severity:

1. **Missing ablation for the title's core claim.** Structural-vs-numeric
   was tested; multi-agent-vs-single-agent was not, despite being the
   headline framing.
2. **Two cherry-picked run examples**, not a fair distribution across
   repeated trials (the current bar in this subfield — see AdaEvolve,
   Section 4 below — is mean ± std over repeated runs).
3. **SGD-momentum and Lion baselines scored ≈−9.3**, which reads (and had
   been written up) as "these optimizers are just worse," without the
   report questioning whether that was actually true.
4. **The fitness function was one opaque scalar**, unable to distinguish
   "genuinely more accurate" from "cheaper at comparable accuracy."
5. **N=5 ablation and zero physical hardware measurement**, thin support
   for an "edge-efficient" framing.
6. **No explicit positioning against prior work that already does this** —
   Lion itself was discovered via program search (so "discover optimizers
   via search" isn't novel), and a cluster of adaptive LLM-search
   frameworks (AdaEvolve, OpenEvolve, ShinkaEvolve, CodeEvolve) sets the
   current methodological bar in the broader space.
7. **Repo/report appendix mismatch** — directories the report's appendix
   described weren't all visible in the pushed repo, a reproducibility red
   flag for any reader checking the code against the paper.

The full original review is preserved at `review_feedback_original.md` for
reference.

## 4. What we fixed and found (v2)

### 4.1 Root-causing the −9.3 scores (feedback point 3)

Two compounding bugs, not one:

- **Fitness-function bug.** The original fitness function penalized "never
  reached target loss within the run" with a fixed constant,
  `NEVER_CONVERGED_STEPS = 10,000` — about 53x the run's actual step budget
  (189 steps: 3 epochs × 63 steps/epoch). That constant alone contributed
  −10.0 to the fitness scalar, numerically indistinguishable from the
  divergence penalty. **Fix:** the never-converged penalty now scales to
  the run's actual step budget (`tasks/fitness.py`, `tasks/proxy_cnn.py`).
- **Genuinely poor LR guesses.** Independently, a per-family LR sweep
  (`search_space/seeds.py: SEED_LR_GRID`, run via
  `scripts/seed_lr_sweep.py`) confirmed the original single-guess LRs for
  SGD-momentum (1e-2) and Lion (1e-4) were themselves bad operating points:

  | Optimizer | Original guess (fitness) | Best after sweep (fitness) |
  |---|---|---|
  | SGD + momentum | 1e-2 (0.040) | 1e-1 (**0.783**) |
  | Lion | 1e-4 (0.133) | 3e-4 (**0.638**) |
  | RMSprop | 1e-2 (0.761) | 3e-3 (**0.831**, new strongest baseline) |
  | Adam / AdamW | 1e-3 (0.669) | 3e-3 (**0.753 / 0.754**) |

  Both fixes were necessary — neither alone fully explains the original
  scores.

### 4.2 Decomposed fitness (feedback point 4)

`tasks/fitness.py` now returns `performance_component` (accuracy +
convergence speed) and `cost_component` (memory + compute proxy) alongside
the scalar, so "discovered beats baseline" can be read as "genuinely more
accurate" vs. "cheaper at comparable accuracy" instead of one number.
`scripts/compare_baselines.py` prints both.

### 4.3 The missing multi-agent ablation (feedback point 1)

`orchestrator/loop.py` gained an `agent_mode` parameter (`"multi"` =
Proposer + Reflector guidance every 5 generations, `"single"` = Proposer
only, no Reflector call) and `scripts/run_agent_ablation.py` runs both
conditions over N trials. **Scoping note, stated explicitly rather than
glossed over:** the Critic's LLM semantic pass is disabled
(`agents/critic.py`, `USE_LLM_SEMANTIC_CHECK = False`) for reliability
reasons documented during v1 development, so the Critic contributes only a
deterministic gate in *both* conditions. This ablation therefore isolates
**Reflector guidance specifically**, not a full three-role comparison.

### 4.4 Full distributions instead of cherry-picked examples (feedback point 2)

`scripts/run_distribution.py` runs N independent full search runs and
reports mean/std/best-of-N and how many runs beat the best fixed baseline,
rather than hand-selecting two examples.

### 4.5 Real hardware validation (feedback point 5)

`scripts/hardware_validation.py` measures real wall-clock time and memory
(not the static structural proxy) for the best discovered candidate vs.
AdamW, runnable `--device cpu` to approximate genuinely edge-scale hardware.

### 4.6 Positioning against prior work (feedback point 6)

Lion — a widely-adopted, real optimizer — was itself discovered via
program search over gradient transforms, sign operations, and momentum,
meaning "discover optimizers via LLM/program search" is an already-shipped
result, not a novel target on its own. A cluster of adaptive LLM-driven
search frameworks — OpenEvolve, ShinkaEvolve, CodeEvolve, and AdaEvolve (a
UC Berkeley framework, early 2026) — has since pushed further, treating the
search process itself as adaptive; AdaEvolve in particular benchmarks
against GEPA/ShinkaEvolve/OpenEvolve (and AlphaEvolve/human-best where
available) across ~185 problems with repeated-trial statistics. That is the
methodological bar this subfield is currently held to, and the reason both
ablations below are run over 5 trials with reported mean/std rather than
1-2 examples.

Given that, this project's contribution is **not** "LLM-guided optimizer
discovery" (done, at a scale this project can't match) and is **not**, as
originally framed, an unqualified "multi-agent helps" claim either — that
required its own ablation, which is now included and did not confirm the
premise. The defensible contribution: two narrow, separately-ablated
empirical questions at small model/task scale, reported honestly including
a negative result.

## 5. Results (v2, final)

### 5.1 Structural vs. numeric-only search — N=5 trials, 15 generations/condition

| Trial | Full search | Numeric-only | Gap |
|---|---|---|---|
| 1 | 0.850 | 0.831 | +0.019 |
| 2 | 0.831 | 0.831 | +0.000 |
| 3 | 0.831 | 0.831 | +0.000 |
| 4 | 0.879 | 0.831 | +0.048 |
| 5 | 0.831 | 0.831 | +0.000 |
| **Mean** | **0.844** | **0.831** | **+0.013** |

Full search won 2/5, tied 3/5 (both landed on the fixed `rmsprop_seed`
baseline), never lost. **Modest, consistent, non-negative — supported.**
N=5 backs "consistent advantage," not a significance claim (no test run).

### 5.2 Multi-agent vs. single-agent search — N=5 trials, 15 generations/condition

| Trial | Multi-agent | Single-agent | Gap (multi − single) |
|---|---|---|---|
| 1 | 0.831 | 0.831 | 0.000 |
| 2 | 0.831 | 0.831 | 0.000 |
| 3 | 0.831 | 0.832 | −0.001 |
| 4 | 0.831 | 0.833 | −0.002 |
| 5 | 0.831 | 0.831 | 0.000 |
| **Mean** | **0.8310** | **0.8316** | **−0.0006** |

Multi-agent never beat the fixed baseline in any of 5 trials (std=0.0000).
Single-agent matched it in 3/5 and marginally beat it in 2/5. Multi-agent
won 0/5, tied 3/5, lost 2/5. **Reflector guidance did not improve outcomes
over a Proposer-only loop here, and trailed slightly — not supported,
reported as a genuine negative result rather than omitted.**

### 5.3 Real hardware validation — CPU, 100 steps

| Optimizer | Mean ms/step | Structural "extra buffers" (proxy) |
|---|---|---|
| AdamW (seed) | 3.154 | 2 |
| Best discovered (Adagrad-like: raw grad, max_sq, no bias correction) | 4.530 | 1 |

The discovered candidate uses fewer state buffers by the structural proxy
but is **44% slower per step** in real wall-clock terms. **The static
memory-cost proxy used to reward candidates during search does not
reliably predict real per-step cost** — an important, if uncomfortable,
finding, and the strongest reason to soften "edge-efficient" in any
title/abstract beyond simply "not yet measured on real edge hardware."

## 6. Limitations (final)

- Single proxy task (MNIST-subset CNN); no cross-task transfer evidence.
- Edge-cost proxy validated against real measurement and found wanting
  (Section 5.3) — search may be optimizing the wrong thing for genuine
  edge deployment.
- CPU benchmark is one machine, not genuinely edge-scale (Jetson/Pi) —
  still needed before "edge-efficient" is a title-level claim.
- N=5 per ablation condition supports "consistent small effect," not a
  statistically powered claim; no significance test was run.
- Multi-agent ablation isolates Proposer-vs-Reflector only, since the
  Critic's LLM pass is disabled (Section 4.3) — not a full three-role test.
- All agent roles run on one local 7-8B-class model; whether the negative
  Reflector result is fundamental or model-specific is untested.

## 7. Repository layout

```
agent_orchestration/
├── main.py                    entry point: python main.py runs one full search
├── models/                    Ollama client + per-role sampling config
├── search_space/              DSL (dsl.py), seed optimizers + LR sweep grid (seeds.py)
├── executor/                  sandboxed subprocess execution
├── tasks/                     proxy CNN task (proxy_cnn.py) + fitness function (fitness.py)
├── agents/                    Proposer, Critic, Reflector, base class
├── orchestrator/              archive (niche-based) + evolutionary loop (agent_mode, locked_structure)
├── prompts/                   system prompts per agent role
├── utils/                     stats.py (mean/std helpers), metrics.py (timers)
├── results/                   curated CSVs/logs backing every number in Section 5 (see .gitignore)
└── scripts/
    ├── compare_baselines.py       discovered-vs-baseline table, decomposed fitness
    ├── seed_lr_sweep.py           full seed LR-sweep table (Section 4.1 evidence)
    ├── run_ablation.py            structural-vs-numeric-only, N trials, mean/std (Section 5.1)
    ├── run_agent_ablation.py      multi-agent vs single-agent, N trials, mean/std (Section 5.2)
    ├── run_distribution.py        N full search runs, full distribution + best-of-N
    ├── hardware_validation.py     real wall-clock/memory benchmark (Section 5.3)
    └── inspect_ablation_winners.py  which candidate (seed vs. discovered) won each trial
```

## 8. Setup

1. `pip install -r requirements.txt`
2. Local LLM via Ollama:
   ```
   ollama pull qwen2.5-coder:7b
   ollama serve
   ```
   (`models/role_config.py` and `models/ollama_client.py` are the source of
   truth for model/serving details.)
3. GPU: sized for a 16GB card (see `models/role_config.py`). CPU-only works
   but is slow; `scripts/hardware_validation.py` is designed to run
   CPU-only on purpose.

## 9. Reproducing every result in Section 5

```bash
# Corrected baseline table (Section 4.1 / 5.1's baseline row):
python scripts/seed_lr_sweep.py

# One full discovery run (30 generations):
python main.py

# Discovered-vs-baseline, decomposed fitness (Section 4.2):
python scripts/compare_baselines.py

# Structural-vs-numeric-only ablation, N=5 x 15 gens (Section 5.1):
python scripts/run_ablation.py 5 15

# Multi-agent-vs-single-agent ablation (Section 5.2):
python scripts/run_agent_ablation.py 5 15

# Which candidate actually won each trial above (seed vs. discovered):
python scripts/inspect_ablation_winners.py

# Full distribution across N independent runs (mean/std/best-of-N):
python scripts/run_distribution.py 12 30

# Real wall-clock/memory measurement, best discovered vs AdamW (Section 5.3):
python scripts/hardware_validation.py --device cpu --steps 100
```

Each script writes its own CSV/JSON next to `archive_log.json` at the repo
root; copy the ones you cite into `results/` (see `.gitignore` — raw
per-trial archives are excluded from version control on purpose, since
they're regenerable and numerous; `results/` holds the curated subset that
backs the report).

## 10. Reproducibility note

The commit that produced the numbers in Section 5 is tagged `paper-v1`:

```bash
git tag paper-v1 <sha> && git push --tags
```

If you re-run these scripts after further edits, results will differ from
this table — re-tag rather than silently updating reported numbers to
match a later commit.

## 11. Conclusion

v1 built a working, crash-resilient multi-agent LLM system for optimizer
discovery and reported (accurately, for what was tested) a structural-
search advantage, but claimed a multi-agent advantage that had never been
ablated, on the strength of two cherry-picked examples and a fitness
function with an undiagnosed bug. v2 fixed the bug, ran the missing
ablation, and expanded every claim to a 5-trial (or 12-run) distribution.
The honest result: structural search helps, modestly and consistently;
multi-agent role-decomposition, as implemented, does not, and the proxy
used to justify "edge-efficient" doesn't hold up against a real
measurement. This is a smaller, more qualified set of claims than v1
made — and a more trustworthy one.