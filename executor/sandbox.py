"""Sandboxed execution of a candidate optimizer against the proxy task.

Runs each candidate's training+eval job in a fresh subprocess with a
wall-clock timeout, so a broken candidate (NaN blowup, infinite loop,
crash) can never take down the orchestrator loop or corrupt shared state.

This is a lightweight sandbox (process isolation + timeout), not full
container isolation -- adequate for a one-month research scope. If you
later run untrusted/adversarial candidates, upgrade to a Docker sandbox.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from search_space.dsl import OptimizerCandidate, render_optimizer_code

TIMEOUT_SECONDS = 120


@dataclass
class ExecutionResult:
    success: bool
    metrics: dict | None
    error: str | None


_RUNNER_TEMPLATE = """
import json
import sys
sys.path.insert(0, {repo_root!r})

{optimizer_code}

from tasks.proxy_cnn import train_and_score

try:
    metrics = train_and_score({class_name})
    print("RESULT_JSON:" + json.dumps(metrics))
except Exception as e:
    print("ERROR:" + str(e))
    sys.exit(1)
"""


def run_candidate(candidate: OptimizerCandidate, repo_root: str) -> ExecutionResult:
    ok, reason = candidate.validate()
    if not ok:
        return ExecutionResult(success=False, metrics=None, error=f"invalid spec: {reason}")

    optimizer_code = render_optimizer_code(candidate)
    class_name = "".join(w.capitalize() for w in candidate.name.split("_")) or "Candidate"

    script = _RUNNER_TEMPLATE.format(
        repo_root=repo_root,
        optimizer_code=optimizer_code,
        class_name=class_name,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(success=False, metrics=None, error="timeout")
    finally:
        Path(script_path).unlink(missing_ok=True)

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            metrics = json.loads(line[len("RESULT_JSON:"):])
            return ExecutionResult(success=True, metrics=metrics, error=None)
        if line.startswith("ERROR:"):
            return ExecutionResult(success=False, metrics=None, error=line[len("ERROR:"):])

    return ExecutionResult(
        success=False,
        metrics=None,
        error=f"no result found; stderr tail: {proc.stderr[-500:]}",
    )