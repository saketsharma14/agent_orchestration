"""Ablation: full structural search vs. numeric-only search.

Runs N independent trials of two conditions, same generation budget each:
  (A) full DSL freedom -- the normal system
  (B) structure locked to AdamW's shape -- Proposer can only tune lr,
      momentum_beta, second_moment_beta, eps, weight_decay

This is what produced report Section 5.2's table -- re-run it any time the
DSL, fitness function, or prompts change, rather than hand-copying old
numbers forward. Every trial's best fitness is written to a CSV so the full
distribution (not just mean/std) is available for the report appendix.

Usage: python scripts/run_ablation.py [n_trials] [n_generations_per_trial]
Default: 5 trials x 15 generations per condition (matches the report).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from orchestrator.loop import run_search
from utils.stats import format_summary

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_GENERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 15

# AdamW's structural signature -- numeric-only condition is locked to this
ADAMW_STRUCTURE = {
    "grad_transform": "momentum",
    "second_moment": "ema_sq",
    "normalization": "sqrt_eps",
    "weight_decay_mode": "decoupled",
    "bias_correction": True,
}

OUT_CSV = Path(REPO_ROOT) / "ablation_structural_vs_numeric_results.csv"


def main():
    rows = []
    full_scores, numeric_scores = [], []

    for trial in range(N_TRIALS):
        print(f"\n########## TRIAL {trial + 1}/{N_TRIALS} ##########")

        print(f"=== Condition A: full structural search ({N_GENERATIONS} generations) ===")
        archive_full = run_search(
            n_generations=N_GENERATIONS,
            locked_structure=None,
            archive_path=str(Path(REPO_ROOT) / f"archive_ablation_full_trial{trial}.json"),
        )
        best_full = archive_full.top_k(1)
        full_score = best_full[0].fitness_scalar if best_full else float("nan")

        print(f"=== Condition B: numeric-only search, locked to AdamW structure ({N_GENERATIONS} generations) ===")
        archive_numeric = run_search(
            n_generations=N_GENERATIONS,
            locked_structure=ADAMW_STRUCTURE,
            archive_path=str(Path(REPO_ROOT) / f"archive_ablation_numeric_trial{trial}.json"),
        )
        best_numeric = archive_numeric.top_k(1)
        numeric_score = best_numeric[0].fitness_scalar if best_numeric else float("nan")

        gap = full_score - numeric_score
        print(f"Trial {trial + 1}: full={full_score:.4f}  numeric_only={numeric_score:.4f}  gap={gap:+.4f}")

        full_scores.append(full_score)
        numeric_scores.append(numeric_score)
        rows.append({"trial": trial + 1, "full": full_score, "numeric_only": numeric_score, "gap": gap})

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "full", "numeric_only", "gap"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Summary across all trials ===")
    print(format_summary(full_scores, "Full structural search"))
    print(format_summary(numeric_scores, "Numeric-only (locked to AdamW)"))
    gaps = [r["gap"] for r in rows]
    print(format_summary(gaps, "Gap (full - numeric_only)"))
    wins = sum(1 for g in gaps if g > 1e-9)
    ties = sum(1 for g in gaps if abs(g) <= 1e-9)
    losses = sum(1 for g in gaps if g < -1e-9)
    print(f"Full search won {wins}/{N_TRIALS} trials, tied {ties}, lost {losses}.")
    print(f"\nPer-trial results written to {OUT_CSV}")
    print(
        "\nNote: this reports a mean gap across trials, not a significance "
        "test (see report Section 5.2/7 -- N trials here supports a "
        "'consistent, non-negative advantage' claim, not a p-value)."
    )


if __name__ == "__main__":
    main()