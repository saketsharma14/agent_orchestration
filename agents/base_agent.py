"""Shared plumbing for all agent roles: looks up its own RoleConfig and
wraps the Ollama client call so subclasses only implement role logic."""

from __future__ import annotations

from models.ollama_client import generate
from models.role_config import ROLE_CONFIGS


class BaseAgent:
    role_name: str = "base"

    def __init__(self):
        self.config = ROLE_CONFIGS[self.role_name]

    def _call(self, user_prompt: str) -> str:
        return generate(
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )