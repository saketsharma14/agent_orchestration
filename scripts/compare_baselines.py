"""Compare the best discovered candidate against the fixed hand-designed
baselines, on the same proxy task. This is the "something to show" script:
run it after a search run and get a clean table for a write-up/slide.

Usage: python scripts/compare_baselines.py [path to archive_log.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from search_space.dsl import OptimizerCandidate, render_optimizer_code
from search_space.seeds import SEED_CANDIDATES
from tasks.fitness import score

ARCHIVE_PATH = sys.argv[1] if len(sys.argv) > 1 else str(Path(REPO_ROOT) / "archive_log.json")


def load_best_discovered(path: str) -> OptimizerCandidate | None:
    with open(path) as f:
        entries = json.load(f)
    valid = [e for e in entries if e["valid"]]
    if not valid:
        return None
    best = max(valid, key=lambda e: e["fitness_scalar"])
    return OptimizerCandidate(**best["candidate"])


def run_one(candidate: OptimizerCandidate) -> dict:
    code = render_optimizer_code(candidate)
    class_name = "".join(w.capitalize() for w in candidate.name.split("_")) or "Candidate"
    namespace: dict = {}
    exec(code, namespace)
    optimizer_cls = namespace[class_name]

    from tasks.proxy_cnn import train_and_score
    metrics = train_and_score(optimizer_cls)
    fit = score(candidate, metrics)
    return {
        "name": candidate.name,
        "metrics": metrics,
        "fitness": fit.scalar,
        "performance_component": fit.vector.get("performance_component"),
        "cost_component": fit.vector.get("cost_component"),
    }


def main():
    candidates = list(SEED_CANDIDATES)  # Adam, AdamW, RMSprop, SGD-momentum, Lion

    best_discovered = load_best_discovered(ARCHIVE_PATH)
    if best_discovered is not None:
        best_discovered.name = "BEST_DISCOVERED_" + best_discovered.name
        candidates.append(best_discovered)
    else:
        print(f"(no valid entries found in {ARCHIVE_PATH} -- showing seed baselines only)\n")

    rows = []
    for cand in candidates:
        try:
            rows.append(run_one(cand))
        except Exception as e:
            rows.append({"name": cand.name, "metrics": None, "fitness": None, "error": str(e)})

    header = (
        f"{'name':32s} {'accuracy':>9s} {'steps_to_target':>16s} {'fitness':>9s} "
        f"{'perf_component':>15s} {'cost_component':>15s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("metrics") is None:
            print(f"{r['name']:32s} {'ERROR: ' + r.get('error', ''):>40s}")
            continue
        acc = r["metrics"]["accuracy"]
        steps = r["metrics"]["steps_to_target"]
        fit = r["fitness"]
        perf = r.get("performance_component")
        cost = r.get("cost_component")
        perf_s = f"{perf:.4f}" if perf is not None else "n/a"
        cost_s = f"{cost:.4f}" if cost is not None else "n/a"
        print(f"{r['name']:32s} {acc:9.3f} {str(steps):>16s} {fit:9.4f} {perf_s:>15s} {cost_s:>15s}")
    print(
        "\nperf_component isolates task performance (accuracy + convergence "
        "speed); cost_component isolates the edge-cost proxy (memory + "
        "compute). A discovered candidate beating baselines mainly on "
        "cost_component is 'cheaper, comparably good' rather than "
        "'genuinely more accurate' -- report feedback point 4."
    )


if __name__ == "__main__":
    main()