"""Real wall-clock and memory measurement for the best discovered candidate
vs. AdamW, replacing the static memory/compute *proxies* used during search
(search_space/dsl.py: memory_cost()/compute_cost() are structural counts,
not measurements) with an actual benchmark.

This still runs on whatever machine you execute it on -- it does NOT by
itself constitute the Raspberry Pi/Jetson number the report feedback asked
for (point 5). What it gives you: (a) a real per-step wall-clock and peak-
RSS number here, which is already more evidence than a static proxy, and
(b) a script that's ready to hand to a Pi/Jetson-class device (CPU-only,
`--device cpu`, small batch) to get the real edge number the "edge-
efficient" framing needs. Run it there and paste the output into the report
Section 5/7 in place of the proxy-only claim.

Usage:
    python scripts/hardware_validation.py [--device cpu|cuda] [--steps 100] [--archive path/to/archive_log.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from search_space.dsl import OptimizerCandidate, render_optimizer_code
from search_space.seeds import SEED_CANDIDATES
from tasks.proxy_cnn import TinyCNN, BATCH_SIZE

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


def load_best_discovered(archive_path: str) -> OptimizerCandidate | None:
    with open(archive_path) as f:
        entries = json.load(f)
    valid = [e for e in entries if e["valid"]]
    if not valid:
        return None
    best = max(valid, key=lambda e: e["fitness_scalar"])
    return OptimizerCandidate(**best["candidate"])


def build_optimizer_cls(candidate: OptimizerCandidate):
    code = render_optimizer_code(candidate)
    class_name = "".join(w.capitalize() for w in candidate.name.split("_")) or "Candidate"
    namespace: dict = {}
    exec(code, namespace)
    return namespace[class_name]


def benchmark(candidate: OptimizerCandidate, device: str, n_steps: int) -> dict:
    optimizer_cls = build_optimizer_cls(candidate)
    torch.manual_seed(0)
    model = TinyCNN().to(device)
    optimizer = optimizer_cls(model.parameters())

    x = torch.randn(BATCH_SIZE, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (BATCH_SIZE,), device=device)

    # Warmup (exclude JIT/allocator warmup from the timed measurement).
    for _ in range(5):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    tracemalloc.start()
    if _HAVE_PSUTIL:
        proc = psutil.Process()
        rss_before = proc.memory_info().rss

    step_times = []
    for _ in range(n_steps):
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

    _, peak_py_alloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "name": candidate.name,
        "device": device,
        "mean_step_time_ms": 1000 * sum(step_times) / len(step_times),
        "median_step_time_ms": 1000 * sorted(step_times)[len(step_times) // 2],
        "optimizer_state_extra_buffers": candidate.memory_cost(),  # the static proxy, for comparison
    }
    if device == "cuda":
        result["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    if _HAVE_PSUTIL:
        rss_after = proc.memory_info().rss
        result["rss_delta_mb"] = (rss_after - rss_before) / (1024 ** 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--archive", default=str(Path(REPO_ROOT) / "archive_log.json"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("--device cuda requested but CUDA is not available here; falling back to cpu.")
        args.device = "cpu"

    adamw_seed = next(c for c in SEED_CANDIDATES if c.name == "adamw_seed")
    candidates = [adamw_seed]

    best_discovered = load_best_discovered(args.archive)
    if best_discovered is not None:
        best_discovered.name = "BEST_DISCOVERED_" + best_discovered.name
        candidates.append(best_discovered)
    else:
        print(f"(no valid entries found in {args.archive} -- benchmarking AdamW baseline only)\n")

    print(f"Device: {args.device}  |  steps per candidate: {args.steps}")
    if args.device == "cpu":
        print(
            "Running on CPU is the closer proxy for genuinely edge-scale "
            "(mobile/embedded) hardware than the GPU numbers used during "
            "search -- see report Section 1.2/7 on what 'edge' meant there."
        )
    print()

    rows = [benchmark(c, args.device, args.steps) for c in candidates]

    header = f"{'name':32s} {'mean_ms/step':>13s} {'median_ms/step':>15s} {'extra_buffers':>14s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:32s} {r['mean_step_time_ms']:13.3f} {r['median_step_time_ms']:15.3f} "
            f"{r['optimizer_state_extra_buffers']:14d}"
        )

    print(
        "\nNote: this is one machine's numbers, not a substitute for a "
        "Jetson/Raspberry-Pi measurement -- run this same script with "
        "--device cpu on that hardware for the number the 'edge-efficient' "
        "framing in the title needs (report feedback point 5)."
    )


if __name__ == "__main__":
    main()