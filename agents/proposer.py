"""Proposer agent: generates a new candidate optimizer spec given the
current archive and (optionally) Reflector guidance."""

from __future__ import annotations

import json
import re

from agents.base_agent import BaseAgent
from search_space.dsl import OptimizerCandidate


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in Proposer output: {text!r}")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Common small-model mistakes: single quotes instead of double,
        # Python-style True/False/None, trailing commas. One cheap repair
        # pass before giving up -- if this still fails, let it raise so the
        # caller can record it as a failed generation instead of crashing.
        repaired = raw.replace("'", '"')
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return json.loads(repaired)


class ProposerAgent(BaseAgent):
    role_name = "proposer"

    def propose(self, archive_summary: str, reflector_guidance: str = "") -> OptimizerCandidate:
        prompt = (
            f"Archive:\n{archive_summary}\n\n"
            f"Reflector guidance:\n{reflector_guidance or '(none yet)'}"
        )
        raw = self._call(prompt)
        spec = _extract_json(raw)
        return OptimizerCandidate(**spec)