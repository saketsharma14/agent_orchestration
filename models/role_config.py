"""Per-role model configuration: which system prompt and sampling params
each agent role uses.

Keeping this separate from the agent classes (agents/*.py) matters for two
things down the line: (1) running the single-agent-vs-multi-agent ablation
later just means swapping which configs get used, not changing code; and
(2) if you later add a "capability ceiling" or "true-edge-scale" comparison
model, you only edit MODEL_NAME / add a second dict here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Served via: ollama pull qwen3-coder:8b && ollama serve
#
# Sized for a 16GB GPU: Qwen3-Coder-8B at Q4 needs roughly 5-6GB of weights,
# leaving comfortable headroom for KV cache and the (tiny, <1GB) proxy-task
# training job that runs concurrently on the same card. The 30B-A3B MoE from
# the original plan needs ~18GB and will not fit here -- swap back to it if
# you ever move to a 24GB+ card.
MODEL_NAME = "qwen2.5-coder:7b"


@dataclass
class RoleConfig:
    model: str
    system_prompt_path: Path
    temperature: float
    top_p: float
    max_tokens: int

    @property
    def system_prompt(self) -> str:
        return self.system_prompt_path.read_text()


ROLE_CONFIGS = {
    "proposer": RoleConfig(
        model=MODEL_NAME,
        system_prompt_path=PROMPTS_DIR / "proposer_system.txt",
        temperature=0.9,  # higher: want diverse mutation proposals
        top_p=0.95,
        max_tokens=400,
    ),
    "critic": RoleConfig(
        model=MODEL_NAME,
        system_prompt_path=PROMPTS_DIR / "critic_system.txt",
        temperature=0.1,  # low: want consistent, near-deterministic judgments
        top_p=0.9,
        max_tokens=200,
    ),
    "reflector": RoleConfig(
        model=MODEL_NAME,
        system_prompt_path=PROMPTS_DIR / "reflector_system.txt",
        temperature=0.5,
        top_p=0.9,
        max_tokens=300,
    ),
}