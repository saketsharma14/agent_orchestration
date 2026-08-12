"""Seed candidates: known hand-designed optimizers expressed in the DSL.

Used to (a) initialize the archive so the Proposer has good examples to
mutate from, and (b) sanity-check the executor/DSL -- if the harness can't
reproduce Adam-level performance from the Adam seed, something is broken
upstream of the search itself, before you go debugging "bad" discovery
results.
"""

from __future__ import annotations

from dataclasses import replace

from search_space.dsl import OptimizerCandidate

# Per-seed LR sweep grids, used by orchestrator.loop.seed_archive instead of
# evaluating each seed at one hand-picked LR.
#
# Why this exists: the original report scored SGD-momentum and Lion at a
# single guessed LR each and got fitness ~-9.3 for both, which read (and was
# written up) as "Lion/SGD are worse." Two things were actually going on:
# (1) a fitness-function bug (see tasks/fitness.py) that made "converged
# slowly" look almost as bad as "diverged", and (2) evaluating sign-based
# (Lion) and un-normalized (SGD-momentum) updates at a single LR is not a
# fair comparison to begin with -- these families are known to need
# different LR scales than Adam/AdamW/RMSprop (Lion roughly 3-10x smaller
# than Adam; see Chen et al. 2023). Sweeping a small grid per family and
# keeping the best result is the minimum needed for the baseline comparison
# in the report to mean what it claims to mean.
SEED_LR_GRID: dict[str, list[float]] = {
    "adam_seed": [3e-4, 1e-3, 3e-3],
    "adamw_seed": [3e-4, 1e-3, 3e-3],
    "rmsprop_seed": [3e-3, 1e-2, 3e-2],
    "sgd_momentum_seed": [3e-3, 1e-2, 3e-2, 1e-1],
    # Lion's sign-based update has unit-magnitude-per-coordinate steps, so it
    # needs a much smaller LR than magnitude-based updates -- sweep an order
    # of magnitude below the (already 10x-reduced-from-Adam) original guess.
    "lion_seed": [1e-5, 3e-5, 1e-4, 3e-4],
}


def lr_sweep_variants(candidate: OptimizerCandidate) -> list[OptimizerCandidate]:
    """All LR variants of a seed candidate to evaluate, per SEED_LR_GRID.
    Falls back to the seed's own single lr if it has no configured grid."""
    grid = SEED_LR_GRID.get(candidate.name, [candidate.lr])
    return [replace(candidate, lr=lr) for lr in grid]


SEED_CANDIDATES = [
    OptimizerCandidate(
        name="adam_seed",
        grad_transform="momentum",
        momentum_beta=0.9,
        second_moment="ema_sq",
        second_moment_beta=0.999,
        normalization="sqrt_eps",
        eps=1e-8,
        bias_correction=True,
        weight_decay_mode="none",
        weight_decay=0.0,
        lr=1e-3,
    ),
    OptimizerCandidate(
        name="adamw_seed",
        grad_transform="momentum",
        momentum_beta=0.9,
        second_moment="ema_sq",
        second_moment_beta=0.999,
        normalization="sqrt_eps",
        eps=1e-8,
        bias_correction=True,
        weight_decay_mode="decoupled",
        weight_decay=0.01,
        lr=1e-3,
    ),
    OptimizerCandidate(
        name="rmsprop_seed",
        grad_transform="raw",
        momentum_beta=0.0,
        second_moment="ema_sq",
        second_moment_beta=0.99,
        normalization="sqrt_eps",
        eps=1e-8,
        bias_correction=False,
        weight_decay_mode="none",
        weight_decay=0.0,
        lr=1e-2,
    ),
    OptimizerCandidate(
        name="sgd_momentum_seed",
        grad_transform="momentum",
        momentum_beta=0.9,
        second_moment="none",
        second_moment_beta=0.0,
        normalization="none",
        eps=1e-8,
        bias_correction=False,
        weight_decay_mode="none",
        weight_decay=0.0,
        lr=1e-2,
    ),
    OptimizerCandidate(
        name="lion_seed",
        grad_transform="sign_momentum",
        momentum_beta=0.9,
        second_moment="none",
        second_moment_beta=0.0,
        normalization="none",
        eps=1e-8,
        bias_correction=False,
        weight_decay_mode="decoupled",
        weight_decay=0.01,
        lr=1e-4,
    ),
]