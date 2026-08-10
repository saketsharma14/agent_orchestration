"""Combines proxy-task training metrics with edge-cost proxies (memory,
compute) into a single scalar fitness score, while keeping the raw vector
around for reporting/plotting/ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

from search_space.dsl import OptimizerCandidate


@dataclass
class FitnessResult:
    scalar: float
    vector: dict


# Starting weights -- tune these once you see the raw metric ranges from
# your first real runs. Keep them here, in one place, so it's obvious where
# to adjust if e.g. accuracy differences are being swamped by the memory term.
W_ACCURACY = 1.0
W_STEPS_TO_TARGET = -0.001  # fewer steps to target loss is better
W_MEMORY = -0.05            # fewer extra state buffers is better
W_COMPUTE = -0.02           # fewer expensive ops per step is better

DIVERGED_PENALTY = -10.0
NEVER_CONVERGED_STEPS = 10_000  # penalty value when steps_to_target is None


def score(candidate: OptimizerCandidate, metrics: dict) -> FitnessResult:
    if metrics.get("diverged"):
        return FitnessResult(scalar=DIVERGED_PENALTY, vector={"diverged": True})

    accuracy = metrics["accuracy"]
    steps_to_target = metrics["steps_to_target"] or NEVER_CONVERGED_STEPS
    memory = candidate.memory_cost()
    compute = candidate.compute_cost()

    scalar = (
        W_ACCURACY * accuracy
        + W_STEPS_TO_TARGET * steps_to_target
        + W_MEMORY * memory
        + W_COMPUTE * compute
    )

    vector = {
        "accuracy": accuracy,
        "steps_to_target": steps_to_target,
        "memory_cost": memory,
        "compute_cost": compute,
        "wall_time_s": metrics["wall_time_s"],
    }
    return FitnessResult(scalar=scalar, vector=vector)