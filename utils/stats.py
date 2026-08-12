"""Small statistics helpers for reporting run distributions -- shared by
scripts/run_ablation.py-style ablations and scripts/run_distribution.py, so
every script reports mean/std/best-of-N the same way instead of each
re-implementing it slightly differently.
"""

from __future__ import annotations

import math


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    """Sample standard deviation (n-1 denominator). Returns 0.0 for n<2
    rather than raising, since a single-trial run is still a valid (if
    uninformative) input to these scripts."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "best": float("nan"), "worst": float("nan")}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "std": stdev(xs),
        "best": max(xs),
        "worst": min(xs),
    }


def format_summary(xs: list[float], label: str = "") -> str:
    s = summarize(xs)
    prefix = f"{label}: " if label else ""
    if s["n"] == 0:
        return f"{prefix}(no data)"
    return (
        f"{prefix}n={s['n']}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
        f"best={s['best']:.4f}  worst={s['worst']:.4f}"
    )