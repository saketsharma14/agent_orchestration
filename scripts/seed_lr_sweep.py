"""Run and print the full seed LR sweep table (search_space.seeds.SEED_LR_GRID),
including every LR tried per optimizer family, not just the winner that ends
up in the archive. This is the evidence for the report's claim that the
original SGD-momentum/Lion baseline numbers were an artifact of evaluating
one hand-picked LR each, not a real performance gap (report feedback point 3).

Usage: python scripts/seed_lr_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from orchestrator.archive import Archive
from orchestrator.loop import seed_archive

OUT_JSON = Path(REPO_ROOT) / "seed_lr_sweep_results.json"


def main():
    archive = Archive()
    sweep_log = []
    seed_archive(archive, sweep_log=sweep_log)

    with open(OUT_JSON, "w") as f:
        json.dump(sweep_log, f, indent=2)

    by_family: dict[str, list[dict]] = {}
    for row in sweep_log:
        by_family.setdefault(row["seed_family"], []).append(row)

    print(f"{'family':22s} {'lr':>10s} {'fitness':>10s}  status")
    print("-" * 60)
    for family, rows in by_family.items():
        rows_sorted = sorted(rows, key=lambda r: r["fitness_scalar"], reverse=True)
        best = rows_sorted[0]
        for row in rows_sorted:
            marker = "  <- best" if row is best else ""
            status = "ok" if row["valid"] else f"INVALID ({row['note']})"
            print(f"{row['seed_family']:22s} {row['lr']:>10.2e} {row['fitness_scalar']:>10.4f}  {status}{marker}")
        print()

    print(f"Full sweep log written to {OUT_JSON}")
    print(
        "\nCompare the best-per-family fitness here against the single-LR "
        "numbers in the original report Section 5.1 table -- a large jump "
        "for sgd_momentum_seed / lion_seed specifically supports the "
        "diagnosis in tasks/fitness.py and search_space/seeds.py."
    )


if __name__ == "__main__":
    main()