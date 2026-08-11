"""Critic agent: static validity check on a proposed candidate, run before
the (expensive) training-based evaluation.

Runs the cheap structural check (OptimizerCandidate.validate()) first --
only calls the LLM if that already passes, so a malformed candidate never
costs you an API/inference call.
"""

from __future__ import annotations

import json
import re

from agents.base_agent import BaseAgent
from search_space.dsl import OptimizerCandidate


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in Critic output: {text!r}")
    return json.loads(match.group(0))


class CriticAgent(BaseAgent):
    role_name = "critic"

    # For a fast, stable milestone run: gate purely on the deterministic
    # validate() check and skip the LLM semantic pass. It was rejecting
    # valid candidates (e.g. normalization="none" with second_moment set,
    # which is legal -- it just means the tracked moment isn't used) and
    # re-deriving the same numeric checks validate() already enforces, in
    # inconsistent wording. Flip this back on once there's time to give it
    # a narrower, more specific job (e.g. only flagging lr/update-magnitude
    # mismatches, which is the one thing that genuinely needs judgment).
    USE_LLM_SEMANTIC_CHECK = False

    def review(self, candidate: OptimizerCandidate) -> tuple[bool, str]:
        ok, reason = candidate.validate()
        if not ok:
            return False, reason

        if not self.USE_LLM_SEMANTIC_CHECK:
            return True, "passed deterministic validation"

        raw = self._call(candidate.to_json())
        result = _extract_json(raw)
        return bool(result.get("valid", False)), result.get("reason", "")