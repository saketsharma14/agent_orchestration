"""Archive of evaluated candidates -- the population the search maintains
and draws from, and the source of the summaries fed back to the Proposer
and Reflector agents each generation."""

from __future__ import annotations

from dataclasses import dataclass

from search_space.dsl import OptimizerCandidate


@dataclass
class ArchiveEntry:
    candidate: OptimizerCandidate
    fitness_scalar: float
    fitness_vector: dict
    valid: bool
    note: str = ""


class Archive:
    def __init__(self):
        self.entries: list[ArchiveEntry] = []

    def add(self, entry: ArchiveEntry) -> None:
        self.entries.append(entry)

    def top_k(self, k: int = 5) -> list[ArchiveEntry]:
        valid_entries = [e for e in self.entries if e.valid]
        return sorted(valid_entries, key=lambda e: e.fitness_scalar, reverse=True)[:k]

    def summary_for_proposer(self, k: int = 5) -> str:
        lines = []
        for e in self.top_k(k):
            lines.append(
                f"- {e.candidate.to_json()} -> fitness={e.fitness_scalar:.4f}, {e.fitness_vector}"
            )
        return "\n".join(lines) if lines else "(archive empty)"

    def summary_for_reflector(self, n: int = 10) -> str:
        recent = self.entries[-n:]
        lines = []
        for e in recent:
            status = "valid" if e.valid else f"INVALID ({e.note})"
            lines.append(f"- {e.candidate.name}: {status}, fitness={e.fitness_scalar:.4f}")
        return "\n".join(lines) if lines else "(no history yet)"