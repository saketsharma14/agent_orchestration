"""Ablation: full structural search vs. numeric-only search.

Runs two searches with the same generation budget:
  (A) full DSL freedom -- the normal system
  (B) structure locked to AdamW's shape -- Proposer can only tune lr,
      momentum_beta, second_moment_beta, eps, weight_decay

If (A)'s best fitness meaningfully beats (B)'s, that's direct evidence
structural exploration is adding value beyond hyperparameter tuning --
not just an impression from reading logs.

Usage: python scripts/run_ablation.py [n_generations_per_condition]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from orchestrator.loop import run_search

N_GENERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 15

# AdamW's structural signature -- numeric-only condition is locked to this
ADAMW_STRUCTURE = {
    "grad_transform": "momentum",
    "second_moment": "ema_sq",
    "normalization": "sqrt_eps",
    "weight_decay_mode": "decoupled",
    "bias_correction": True,
}


def main():
    print(f"=== Condition A: full structural search ({N_GENERATIONS} generations) ===")
    archive_full = run_search(
        n_generations=N_GENERATIONS,
        locked_structure=None,
        archive_path=str(Path(REPO_ROOT) / "archive_ablation_full.json"),
    )

    print(f"\n=== Condition B: numeric-only search, locked to AdamW structure ({N_GENERATIONS} generations) ===")
    archive_numeric = run_search(
        n_generations=N_GENERATIONS,
        locked_structure=ADAMW_STRUCTURE,
        archive_path=str(Path(REPO_ROOT) / "archive_ablation_numeric_only.json"),
    )

    best_full = archive_full.top_k(1)
    best_numeric = archive_numeric.top_k(1)

    print("\n=== Results ===")
    if best_full:
        print(f"Full structural search best fitness:  {best_full[0].fitness_scalar:.4f}  ({best_full[0].candidate.name})")
    else:
        print("Full structural search: no valid candidates found")
    if best_numeric:
        print(f"Numeric-only search best fitness:     {best_numeric[0].fitness_scalar:.4f}  ({best_numeric[0].candidate.name})")
    else:
        print("Numeric-only search: no valid candidates found")

    if best_full and best_numeric:
        gap = best_full[0].fitness_scalar - best_numeric[0].fitness_scalar
        print(f"\nGap (full - numeric-only): {gap:+.4f}")
        print("(A positive, meaningful gap is evidence structural search adds value beyond hyperparameter tuning.")
        print(" A single run of each is a first look, not a proof -- re-run a couple times if time allows,")
        print(" since either condition can get lucky/unlucky on one run.)")


if __name__ == "__main__":
    main()