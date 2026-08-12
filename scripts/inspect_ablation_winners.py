"""Inspect which candidate actually won each trial of an ablation run --
critical for interpreting run_ablation.py / run_agent_ablation.py output,
since a fitness gap of 0.0000 across trials usually means the winner in
both conditions was the *same seed candidate* (the proxy task is seeded,
so identical hyperparameters always score identically), not that search
found and then re-found some new candidate by coincidence.

Usage: python scripts/inspect_ablation_winners.py [glob_prefix]
Default prefix matches both run_ablation.py and run_agent_ablation.py
output file naming.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)

DEFAULT_PATTERNS = [
    ("archive_ablation_full_trial*.json", "FULL STRUCTURAL SEARCH"),
    ("archive_ablation_numeric_trial*.json", "NUMERIC-ONLY (locked to AdamW)"),
    ("archive_agentabl_multi_trial*.json", "MULTI-AGENT (Proposer+Reflector)"),
    ("archive_agentabl_single_trial*.json", "SINGLE-AGENT (Proposer-only)"),
]

SEED_NAMES = {"adam_seed", "adamw_seed", "rmsprop_seed", "sgd_momentum_seed", "lion_seed"}


def main():
    root = Path(REPO_ROOT)
    for pattern, label in DEFAULT_PATTERNS:
        paths = sorted(root.glob(pattern))
        if not paths:
            continue
        print(f"=== {label} ===")
        for path in paths:
            with open(path) as f:
                entries = json.load(f)
            valid = [e for e in entries if e["valid"]]
            if not valid:
                print(f"{path.name}: no valid entries")
                continue
            best = max(valid, key=lambda e: e["fitness_scalar"])
            name = best["candidate"]["name"]
            tag = "SEED (search found nothing better)" if name in SEED_NAMES else "DISCOVERED (new candidate)"
            print(
                f"{path.name}: winner={name:24s} lr={best['candidate']['lr']:.2e}  "
                f"fitness={best['fitness_scalar']:.4f}  [{tag}]"
            )
        print()


if __name__ == "__main__":
    main()