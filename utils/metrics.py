from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    label: str
    seconds: float


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    result = {"label": label, "seconds": None}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        print(f"{label}: {result['seconds']:.2f}s")