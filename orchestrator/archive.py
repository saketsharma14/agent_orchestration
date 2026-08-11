"""Archive of evaluated candidates.

Uses a MAP-Elites-style structure: candidates are bucketed into "niches" by
their *structural* choices (grad_transform, second_moment, normalization,
weight_decay_mode, bias_correction), and only the best-scoring candidate per
niche is kept as that niche's representative. This exists specifically to
stop the search from collapsing into pure hyperparameter tuning around
whichever structure happened to score well early -- a plain top-k-by-fitness
archive keeps re-selecting near-duplicates of the current best structure and
starves out every other structural family. (If you're reading this after a
run where memory_cost/compute_cost never varied across generations, that's
this bug -- the fix is what's below.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from search_space.dsl import OptimizerCandidate

NicheKey = tuple


@dataclass
class ArchiveEntry:
    candidate: OptimizerCandidate
    fitness_scalar: float
    fitness_vector: dict
    valid: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.__dict__,
            "fitness_scalar": self.fitness_scalar,
            "fitness_vector": self.fitness_vector,
            "valid": self.valid,
            "note": self.note,
        }


def niche_key(candidate: OptimizerCandidate) -> NicheKey:
    """The structural signature defining a niche. Continuous fields (lr,
    betas, weight_decay, eps) are deliberately excluded -- those are what's
    allowed to vary *within* a niche, via ordinary mutation."""
    return (
        candidate.grad_transform,
        candidate.second_moment,
        candidate.normalization,
        candidate.weight_decay_mode,
        candidate.bias_correction,
    )


class Archive:
    def __init__(self):
        self.entries: list[ArchiveEntry] = []
        self.niches: dict[NicheKey, ArchiveEntry] = {}

    def add(self, entry: ArchiveEntry) -> None:
        self.entries.append(entry)
        if not entry.valid:
            return
        key = niche_key(entry.candidate)
        current = self.niches.get(key)
        if current is None or entry.fitness_scalar > current.fitness_scalar:
            self.niches[key] = entry

    def save(self, path: str) -> None:
        """Persist every entry so far to disk. Call this after EVERY
        generation, not just at the end -- if the process crashes partway
        through (a bad LLM response, an OOM, anything), you still keep
        everything scored up to that point instead of losing the whole run."""
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def top_k(self, k: int = 5) -> list[ArchiveEntry]:
        valid_entries = [e for e in self.entries if e.valid]
        return sorted(valid_entries, key=lambda e: e.fitness_scalar, reverse=True)[:k]

    def niche_representatives(self) -> list[ArchiveEntry]:
        """One best-so-far entry per structural niche discovered -- shown
        to the Proposer instead of global top-k, so it always sees the full
        breadth of structures explored, not just whichever currently wins."""
        return sorted(self.niches.values(), key=lambda e: e.fitness_scalar, reverse=True)

    def summary_for_proposer(self, max_niches: int = 8) -> str:
        reps = self.niche_representatives()[:max_niches]
        if not reps:
            return "(archive empty)"
        lines = [
            f"- {e.candidate.to_json()} -> fitness={e.fitness_scalar:.4f}, {e.fitness_vector}"
            for e in reps
        ]
        lines.append(
            f"\n({len(self.niches)} distinct structural niches explored so far -- "
            f"prefer proposing a candidate in an unrepresented niche unless "
            f"guidance says to refine an existing one.)"
        )
        return "\n".join(lines)

    def summary_for_reflector(self, n: int = 10) -> str:
        recent = self.entries[-n:]
        lines = []
        for e in recent:
            status = "valid" if e.valid else f"INVALID ({e.note})"
            lines.append(f"- {e.candidate.name}: {status}, fitness={e.fitness_scalar:.4f}")
        distinct = len({niche_key(e.candidate) for e in recent if e.valid})
        lines.append(f"\n(only {distinct} distinct structural niche(s) among these {len(recent)} candidates)")
        return "\n".join(lines) if lines else "(no history yet)"