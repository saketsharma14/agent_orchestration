"""Run N independent full (unrestricted) search runs and report the full
distribution of outcomes: mean/std/best-of-N discovered-vs-baseline, in the
spirit of how AdaEvolve/ShinkaEvolve-style papers report results (mean +/-
std over repeated runs, not 1-2 examples). Report Section 5.1 originally
showed exactly two hand-picked runs; this script is what should back that
section instead.

Usage: python scripts/run_distribution.py [n_runs] [n_generations_per_run]
Default: 12 runs x 30 generations (report's original single-run budget).

This can take a long time (N_RUNS x N_GENERATIONS candidate evaluations,
plus the seed LR sweep each run). Reduce n_runs/n_generations for a quick
smoke test before committing to the full budget.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from orchestrator.loop import run_search
from search_space.seeds import SEED_CANDIDATES
from utils.stats import format_summary, summarize

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_GENERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

OUT_CSV = Path(REPO_ROOT) / "distribution_results.csv"


def niche_signature(candidate) -> str:
    return (
        f"{candidate.grad_transform}/{candidate.second_moment}/"
        f"{candidate.normalization}/{candidate.weight_decay_mode}/"
        f"bc={candidate.bias_correction}"
    )


def main():
    rows = []
    best_fitnesses = []
    baseline_best_fitness = None  # filled in from the first run's seeded baselines

    for run_idx in range(N_RUNS):
        print(f"\n########## RUN {run_idx + 1}/{N_RUNS} ##########")
        archive = run_search(
            n_generations=N_GENERATIONS,
            archive_path=str(Path(REPO_ROOT) / f"archive_dist_run{run_idx}.json"),
        )

        if baseline_best_fitness is None:
            seed_entries = [
                e for e in archive.entries
                if e.valid and e.candidate.name in {c.name for c in SEED_CANDIDATES}
            ]
            if seed_entries:
                baseline_best_fitness = max(e.fitness_scalar for e in seed_entries)

        best = archive.top_k(1)
        if best:
            entry = best[0]
            row = {
                "run": run_idx + 1,
                "best_fitness": entry.fitness_scalar,
                "accuracy": entry.fitness_vector.get("accuracy"),
                "steps_to_target": entry.fitness_vector.get("steps_to_target"),
                "structural_niche": niche_signature(entry.candidate),
                "n_niches_explored": len(archive.niches),
            }
        else:
            row = {
                "run": run_idx + 1,
                "best_fitness": float("nan"),
                "accuracy": None,
                "steps_to_target": None,
                "structural_niche": None,
                "n_niches_explored": len(archive.niches),
            }

        print(f"Run {run_idx + 1} best: {row}")
        rows.append(row)
        best_fitnesses.append(row["best_fitness"])

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "best_fitness", "accuracy", "steps_to_target", "structural_niche", "n_niches_explored"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Distribution summary across all runs ===")
    print(format_summary(best_fitnesses, "Discovered best fitness"))
    s = summarize(best_fitnesses)
    print(f"Best-of-{N_RUNS}: {s['best']:.4f}")
    if baseline_best_fitness is not None:
        print(f"Best fixed-baseline fitness (from seed sweep): {baseline_best_fitness:.4f}")
        print(f"Mean discovered vs best baseline: {s['mean'] - baseline_best_fitness:+.4f}")
        beats_baseline = sum(1 for x in best_fitnesses if x > baseline_best_fitness)
        print(f"Runs where discovered beat the best baseline: {beats_baseline}/{N_RUNS}")

    niches = [r["structural_niche"] for r in rows if r["structural_niche"]]
    distinct_niches = len(set(niches))
    print(f"\nDistinct winning structural niches across runs: {distinct_niches}/{len(niches)}")
    print(
        "(If this is 1, every run converged on the same structure -- report "
        "that plainly, it's a real finding about the search, not a null "
        "result to hide.)"
    )
    print(f"\nPer-run results written to {OUT_CSV}")


if __name__ == "__main__":
    main()