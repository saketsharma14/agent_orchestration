"""Reflector agent: periodically summarizes archive trends into short,
actionable guidance for the Proposer."""

from __future__ import annotations

from agents.base_agent import BaseAgent


class ReflectorAgent(BaseAgent):
    role_name = "reflector"

    def reflect(self, recent_history_summary: str) -> str:
        return self._call(recent_history_summary)