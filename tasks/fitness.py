"""Combines proxy-task training metrics with edge-cost proxies (memory,
compute) into a single scalar fitness score, while keeping the raw vector
around for reporting/plotting/ablations.
"""
from __future__ import annotations

from dataclasses import dataclass

from search_space.dsl import OptimizerCandidate
from tasks.proxy_cnn import TOTAL_STEPS


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

# The "never converged" penalty needs to score worse than any candidate that
# did converge, but shouldn't be an arbitrary constant unrelated to the real
# step budget -- a fixed 10_000 against a ~189-step budget (TOTAL_STEPS) made
# "converged slowly" score almost as badly as "diverged", which is what
# produced the misleading -9.3 SGD-momentum/Lion numbers in the original
# report (report feedback point 3). Scale it relative to TOTAL_STEPS instead,
# so the penalty stays proportionate if the proxy task's budget ever changes.
NEVER_CONVERGED_MULTIPLIER = 3
NEVER_CONVERGED_STEPS = TOTAL_STEPS * NEVER_CONVERGED_MULTIPLIER


def score(candidate: OptimizerCandidate, metrics: dict) -> FitnessResult:
    if metrics.get("diverged"):
        return FitnessResult(scalar=DIVERGED_PENALTY, vector={"diverged": True})

    accuracy = metrics["accuracy"]
    steps_to_target = metrics["steps_to_target"] or NEVER_CONVERGED_STEPS
    memory = candidate.memory_cost()
    compute = candidate.compute_cost()

    # Split into the two components report feedback point 4 asked for, so a
    # reader can tell "genuinely more accurate" apart from "cheaper, roughly
    # as good" -- scalar is still just their sum, so nothing about the
    # search or existing ablations changes, only what gets reported.
    performance_component = (
        W_ACCURACY * accuracy + W_STEPS_TO_TARGET * steps_to_target
    )
    cost_component = W_MEMORY * memory + W_COMPUTE * compute
    scalar = performance_component + cost_component

    vector = {
        "accuracy": accuracy,
        "steps_to_target": steps_to_target,
        "memory_cost": memory,
        "compute_cost": compute,
        "wall_time_s": metrics["wall_time_s"],
        "performance_component": performance_component,
        "cost_component": cost_component,
    }
    return FitnessResult(scalar=scalar, vector=vector)