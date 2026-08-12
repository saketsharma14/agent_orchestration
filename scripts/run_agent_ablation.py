"""Ablation: multi-agent search vs. single-agent search.

This is the ablation the title's core claim ("multi-agent...discovery")
actually depends on, and the one flagged as missing: report Section 5.2
tests structural-vs-numeric search, which is a different question (does
changing optimizer *structure* help, at all) from "does agent
role-decomposition help" (does splitting propose/critique/reflect across
roles help, versus one agent doing it all).

What this specifically isolates: the Critic's LLM semantic pass is already
disabled in this codebase (agents/critic.py, USE_LLM_SEMANTIC_CHECK=False --
see report Section 4), so in both conditions below the Critic contributes
only a deterministic validity gate, not an LLM judgment. That means the only
LLM-driven role difference between "multi-agent" and "single-agent" here is
whether the Reflector's periodic guidance is used. This ablation therefore
answers a narrower, honest question: does Reflector guidance improve search
over a Proposer-only loop, same model, same per-generation cost? Report this
framing explicitly rather than the broader "multi-agent vs single-agent"
phrase from the title -- see report_addendum.md.

Usage: python scripts/run_agent_ablation.py [n_trials] [n_generations_per_trial]
Default: 5 trials x 15 generations per condition.
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
REFLECT_EVERY = 5  # same cadence as the main system (orchestrator/loop.py)

OUT_CSV = Path(REPO_ROOT) / "ablation_agent_mode_results.csv"


def main():
    rows = []
    multi_scores, single_scores = [], []

    for trial in range(N_TRIALS):
        print(f"\n########## TRIAL {trial + 1}/{N_TRIALS} ##########")

        print(f"=== Condition A: multi-agent (Proposer + Reflector every {REFLECT_EVERY} gens) ===")
        archive_multi = run_search(
            n_generations=N_GENERATIONS,
            agent_mode="multi",
            reflect_every=REFLECT_EVERY,
            archive_path=str(Path(REPO_ROOT) / f"archive_agentabl_multi_trial{trial}.json"),
        )
        best_multi = archive_multi.top_k(1)
        multi_score = best_multi[0].fitness_scalar if best_multi else float("nan")

        print(f"=== Condition B: single-agent (Proposer only, no Reflector call) ===")
        archive_single = run_search(
            n_generations=N_GENERATIONS,
            agent_mode="single",
            archive_path=str(Path(REPO_ROOT) / f"archive_agentabl_single_trial{trial}.json"),
        )
        best_single = archive_single.top_k(1)
        single_score = best_single[0].fitness_scalar if best_single else float("nan")

        gap = multi_score - single_score
        print(f"Trial {trial + 1}: multi={multi_score:.4f}  single={single_score:.4f}  gap={gap:+.4f}")

        multi_scores.append(multi_score)
        single_scores.append(single_score)
        rows.append({"trial": trial + 1, "multi_agent": multi_score, "single_agent": single_score, "gap": gap})

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "multi_agent", "single_agent", "gap"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Summary across all trials ===")
    print(format_summary(multi_scores, "Multi-agent (Proposer + Reflector)"))
    print(format_summary(single_scores, "Single-agent (Proposer only)"))
    gaps = [r["gap"] for r in rows]
    print(format_summary(gaps, "Gap (multi - single)"))
    wins = sum(1 for g in gaps if g > 1e-9)
    ties = sum(1 for g in gaps if abs(g) <= 1e-9)
    losses = sum(1 for g in gaps if g < -1e-9)
    print(f"Multi-agent won {wins}/{N_TRIALS} trials, tied {ties}, lost {losses}.")
    print(f"\nPer-trial results written to {OUT_CSV}")
    print(
        "\nIf this gap is near zero or inconsistent in sign, that is a real "
        "and reportable result -- it means Reflector guidance isn't earning "
        "its LLM-call cost in this DSL/proxy-task regime, which is exactly "
        "the kind of finding a workshop-scale ablation should surface "
        "either way (see report feedback point 1)."
    )


if __name__ == "__main__":
    main()