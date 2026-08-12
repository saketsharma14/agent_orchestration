# agent_orchestration

Multi-agent LLM-guided discovery of edge-efficient gradient-based optimizers.
Companion code for the report *"Multi-Agent LLM-Guided Discovery of
Edge-Efficient Optimizers."* See `report_addendum.md` for what changed since
the version the report describes, and why.

## What's here

```
agent_orchestration/
├── main.py                    entry point: python main.py runs one full search
├── models/                    Ollama client + per-role sampling config
├── search_space/              DSL (dsl.py), seed optimizers + LR sweep grid (seeds.py)
├── executor/                  sandboxed subprocess execution
├── tasks/                     proxy CNN task (proxy_cnn.py) + fitness function (fitness.py)
├── agents/                    Proposer, Critic, Reflector, base class
├── orchestrator/              archive (niche-based) + evolutionary loop
├── prompts/                   system prompts per agent role
├── utils/                     stats.py (mean/std helpers), metrics.py (timers)
└── scripts/
    ├── compare_baselines.py     discovered-vs-baseline comparison table (decomposed fitness)
    ├── seed_lr_sweep.py         full seed LR-sweep table (why SGD/Lion baselines were wrong)
    ├── run_ablation.py          structural-vs-numeric-only search, N trials, mean/std
    ├── run_agent_ablation.py    multi-agent (Proposer+Reflector) vs single-agent (Proposer-only)
    └── run_distribution.py      N full search runs, full result distribution + best-of-N
```

## Setup

1. Python deps: `pip install -r requirements.txt`
2. Local LLM via Ollama:
   ```
   ollama pull qwen2.5-coder:7b
   ollama serve
   ```
   (Model name/served-via details live in `models/role_config.py` and
   `models/ollama_client.py` — that file is the source of truth, not any
   config elsewhere in the repo.)
3. GPU: sized for a 16GB card (see `models/role_config.py` for the sizing
   rationale). CPU-only works but is slow; `scripts/hardware_validation.py`
   is designed to run CPU-only on purpose (see below).

## Reproducing the report's results

```bash
# Sanity-check the harness + get the seed baseline table (Adam/AdamW/RMSprop/
# SGD-momentum/Lion), each swept over its LR grid rather than one guessed LR:
python scripts/seed_lr_sweep.py

# One full discovery run (30 generations, matches the report's default budget):
python main.py

# Discovered-vs-baseline comparison table, with fitness decomposed into
# performance vs. cost components:
python scripts/compare_baselines.py

# Structural-vs-numeric-only ablation, N=5 trials x 15 generations each
# (this is what backs report Section 5.2's table):
python scripts/run_ablation.py 5 15

# Multi-agent-vs-single-agent ablation (the ablation the report's title
# claim actually depends on, and the one flagged as missing):
python scripts/run_agent_ablation.py 5 15

# Full distribution across N independent runs (mean/std/best-of-N, not two
# cherry-picked examples):
python scripts/run_distribution.py 12 30

# Real (not simulated) wall-clock/memory measurement, best discovered vs
# AdamW. Run with --device cpu on a genuinely edge-scale device (Jetson/
# Raspberry Pi) for the number the "edge-efficient" framing needs:
python scripts/hardware_validation.py --device cpu --steps 100
```

Each script writes its own CSV/JSON of per-trial or per-run results next to
`archive_log.json` at the repo root, so raw numbers behind any report table
are always available, not just the summary.

## Reproducibility note

The commit that produced any numbers reported in the paper should be tagged
(e.g. `git tag paper-v1 <sha> && git push --tags`) and referenced in the
paper by tag or SHA. If you re-run these scripts after further edits, the
numbers will differ from a previously reported table — re-tag rather than
silently updating the paper's numbers to match a later commit.

## Known scope limits (see report Sections 1.2 and 7)

- Single proxy task (MNIST-subset CNN); no cross-task transfer evidence yet.
- "Edge" cost is a static structural proxy during search
  (`OptimizerCandidate.memory_cost()` / `.compute_cost()` in
  `search_space/dsl.py`); `scripts/hardware_validation.py` is the first step
  toward a real measurement, but still needs to be run on actual edge
  hardware to fully back an "edge-efficient" claim.
- The Critic's LLM semantic pass is currently disabled
  (`agents/critic.py`, `USE_LLM_SEMANTIC_CHECK = False`) — both ablations
  above therefore only compare Proposer-only vs. Proposer+Reflector, not a
  three-role vs. one-role comparison. See `scripts/run_agent_ablation.py`'s
  docstring and `report_addendum.md` for why, and what a fair three-role
  ablation would require.