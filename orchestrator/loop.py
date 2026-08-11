"""The evolutionary loop: propose -> critique -> evaluate -> update archive
-> (periodically) reflect.

This is the piece that ties agents/, search_space/, executor/, and tasks/
together. Run it directly (`python orchestrator/loop.py`) or via main.py.
"""

from __future__ import annotations

from pathlib import Path

from agents.critic import CriticAgent
from agents.proposer import ProposerAgent
from agents.reflector import ReflectorAgent
from executor.sandbox import run_candidate
from orchestrator.archive import Archive, ArchiveEntry
from search_space.seeds import SEED_CANDIDATES
from tasks.fitness import score

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
REFLECT_EVERY = 5
N_GENERATIONS = 30
ARCHIVE_PATH = str(Path(__file__).resolve().parent.parent / "archive_log.json")


def seed_archive(archive: Archive) -> None:
    """Run every seed candidate once so the archive starts non-empty and
    the harness is sanity-checked against known optimizers before search
    begins. If a seed (e.g. adam_seed) fails here, fix the DSL/executor
    before trusting any discovery results."""
    for cand in SEED_CANDIDATES:
        result = run_candidate(cand, REPO_ROOT)
        if result.success:
            fit = score(cand, result.metrics)
            archive.add(ArchiveEntry(cand, fit.scalar, fit.vector, valid=True))
        else:
            archive.add(ArchiveEntry(cand, -999.0, {}, valid=False, note=result.error))


def run_search(n_generations: int = N_GENERATIONS) -> Archive:
    archive = Archive()
    seed_archive(archive)
    archive.save(ARCHIVE_PATH)

    proposer = ProposerAgent()
    critic = CriticAgent()
    reflector = ReflectorAgent()
    guidance = ""

    for gen in range(n_generations):
        print(f"--- generation {gen} ---")

        try:
            candidate = proposer.propose(archive.summary_for_proposer(), guidance)
        except Exception as e:
            # Malformed LLM output (bad JSON, missing fields, etc.) should
            # cost you one generation, not the whole run. Log it as a
            # skipped generation and keep going.
            print(f"proposer failed, skipping generation: {e}")
            archive.save(ARCHIVE_PATH)
            continue

        try:
            valid, reason = critic.review(candidate)
        except Exception as e:
            print(f"critic failed, skipping generation: {e}")
            archive.save(ARCHIVE_PATH)
            continue

        if not valid:
            archive.add(ArchiveEntry(candidate, -999.0, {}, valid=False, note=reason))
            print(f"rejected by critic: {reason}")
            archive.save(ARCHIVE_PATH)
            continue

        result = run_candidate(candidate, REPO_ROOT)
        if not result.success:
            archive.add(ArchiveEntry(candidate, -999.0, {}, valid=False, note=result.error))
            print(f"execution failed: {result.error}")
            archive.save(ARCHIVE_PATH)
            continue

        fit = score(candidate, result.metrics)
        archive.add(ArchiveEntry(candidate, fit.scalar, fit.vector, valid=True))
        print(f"fitness={fit.scalar:.4f} vector={fit.vector}")
        archive.save(ARCHIVE_PATH)

        if (gen + 1) % REFLECT_EVERY == 0:
            try:
                guidance = reflector.reflect(archive.summary_for_reflector())
                print(f"reflector guidance: {guidance}")
            except Exception as e:
                print(f"reflector failed, keeping previous guidance: {e}")

    return archive


if __name__ == "__main__":
    run_search()