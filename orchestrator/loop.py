"""The evolutionary loop: propose -> critique -> evaluate -> update archive
-> (periodically) reflect.

This is the piece that ties agents/, search_space/, executor/, and tasks/
together. Run it directly (`python orchestrator/loop.py`) or via main.py.

Two ablations depend on run_search's parameters below:
  - structural-vs-numeric-only (scripts/run_ablation.py): locked_structure
  - multi-agent-vs-single-agent (scripts/run_agent_ablation.py): agent_mode

agent_mode note: the Critic's LLM semantic pass is already disabled
(agents/critic.py, USE_LLM_SEMANTIC_CHECK = False) for reliability reasons
documented in the report (Section 4), so the Critic contributes only a
deterministic gate in *both* conditions below -- it is not part of what
"multi-agent" vs "single-agent" is testing. The comparison this makes is
therefore specifically: does the Reflector's periodic guidance help, given
the same Proposer, same model, same generation budget? That is a narrower
and more honest framing than "multi-agent vs single-agent" in the abstract,
and scripts/run_agent_ablation.py's output/README says so explicitly.
"""

from __future__ import annotations

from pathlib import Path

from agents.critic import CriticAgent
from agents.proposer import ProposerAgent
from agents.reflector import ReflectorAgent
from executor.sandbox import run_candidate
from orchestrator.archive import Archive, ArchiveEntry
from search_space.seeds import SEED_CANDIDATES, lr_sweep_variants
from tasks.fitness import score

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
REFLECT_EVERY = 5
N_GENERATIONS = 30
ARCHIVE_PATH = str(Path(__file__).resolve().parent.parent / "archive_log.json")
SEED_SWEEP_LOG_PATH = str(Path(__file__).resolve().parent.parent / "seed_lr_sweep_log.json")


def seed_archive(archive: Archive, sweep_log: list | None = None) -> None:
    """Run every seed candidate once so the archive starts non-empty and
    the harness is sanity-checked against known optimizers before search
    begins. If a seed (e.g. adam_seed) fails here, fix the DSL/executor
    before trusting any discovery results.

    Each seed is evaluated across its LR sweep grid (search_space/seeds.py)
    rather than at a single hand-picked LR, and only the best-fitness
    variant is kept as that seed's archive entry -- see fitness.py and
    seeds.py docstrings for why a single guessed LR per seed produced
    misleading baseline numbers for SGD-momentum and Lion in particular.
    If sweep_log is given, every variant tried is appended to it (used by
    scripts/seed_lr_sweep.py to show the full sweep table, not just the
    winners).
    """
    for base_cand in SEED_CANDIDATES:
        variants = lr_sweep_variants(base_cand)
        best_entry = None
        for variant in variants:
            result = run_candidate(variant, REPO_ROOT)
            if result.success:
                fit = score(variant, result.metrics)
                entry = ArchiveEntry(variant, fit.scalar, fit.vector, valid=True)
            else:
                entry = ArchiveEntry(variant, -999.0, {}, valid=False, note=result.error)

            if sweep_log is not None:
                sweep_log.append(
                    {
                        "seed_family": base_cand.name,
                        "lr": variant.lr,
                        "fitness_scalar": entry.fitness_scalar,
                        "valid": entry.valid,
                        "note": entry.note,
                    }
                )

            if entry.valid and (best_entry is None or entry.fitness_scalar > best_entry.fitness_scalar):
                best_entry = entry

        # If every LR in the grid failed outright (crash, not just low
        # fitness), still record the last attempt so the seed shows up as
        # invalid rather than silently vanishing from the archive.
        archive.add(best_entry if best_entry is not None else entry)


def run_search(
    n_generations: int = N_GENERATIONS,
    locked_structure: dict | None = None,
    archive_path: str = ARCHIVE_PATH,
    agent_mode: str = "multi",
    reflect_every: int = REFLECT_EVERY,
) -> Archive:
    """Run one full discovery loop.

    Args:
        n_generations: number of propose/evaluate generations to run.
        locked_structure: if given, every proposed candidate has these
            structural fields forced to the given values (see
            agents/proposer.py) -- used for the numeric-only control in
            scripts/run_ablation.py. None means unrestricted structural
            search.
        archive_path: where to persist the archive after every generation.
        agent_mode: "multi" (Proposer + periodic Reflector guidance, the
            system as described in the report) or "single" (Proposer only,
            no Reflector call ever -- a single-agent propose-evaluate-refine
            loop with the same model and per-generation budget). See module
            docstring for what this ablation does and doesn't isolate.
        reflect_every: how many generations between Reflector calls, when
            agent_mode == "multi". Ignored in "single" mode.
    """
    if agent_mode not in ("multi", "single"):
        raise ValueError(f"agent_mode must be 'multi' or 'single', got {agent_mode!r}")

    archive = Archive()
    seed_archive(archive)
    archive.save(archive_path)

    proposer = ProposerAgent()
    critic = CriticAgent()
    reflector = ReflectorAgent() if agent_mode == "multi" else None
    guidance = ""

    for gen in range(n_generations):
        print(f"--- generation {gen} ({agent_mode}-agent) ---")

        try:
            candidate = proposer.propose(
                archive.summary_for_proposer(),
                guidance,
                locked_structure=locked_structure,
            )
        except Exception as e:
            # Malformed LLM output (bad JSON, missing fields, etc.) should
            # cost you one generation, not the whole run. Log it as a
            # skipped generation and keep going.
            print(f"proposer failed, skipping generation: {e}")
            archive.save(archive_path)
            continue

        try:
            valid, reason = critic.review(candidate)
        except Exception as e:
            print(f"critic failed, skipping generation: {e}")
            archive.save(archive_path)
            continue

        if not valid:
            archive.add(ArchiveEntry(candidate, -999.0, {}, valid=False, note=reason))
            print(f"rejected by critic: {reason}")
            archive.save(archive_path)
            continue

        result = run_candidate(candidate, REPO_ROOT)
        if not result.success:
            archive.add(ArchiveEntry(candidate, -999.0, {}, valid=False, note=result.error))
            print(f"execution failed: {result.error}")
            archive.save(archive_path)
            continue

        fit = score(candidate, result.metrics)
        archive.add(ArchiveEntry(candidate, fit.scalar, fit.vector, valid=True))
        print(f"fitness={fit.scalar:.4f} vector={fit.vector}")
        archive.save(archive_path)

        if reflector is not None and (gen + 1) % reflect_every == 0:
            try:
                guidance = reflector.reflect(archive.summary_for_reflector())
                print(f"reflector guidance: {guidance}")
            except Exception as e:
                print(f"reflector failed, keeping previous guidance: {e}")

    return archive


if __name__ == "__main__":
    run_search()