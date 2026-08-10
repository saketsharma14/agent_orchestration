"""Seed candidates: known hand-designed optimizers expressed in the DSL.

Used to (a) initialize the archive so the Proposer has good examples to
mutate from, and (b) sanity-check the executor/DSL -- if the harness can't
reproduce Adam-level performance from the Adam seed, something is broken
upstream of the search itself, before you go debugging "bad" discovery
results.
"""

from search_space.dsl import OptimizerCandidate

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