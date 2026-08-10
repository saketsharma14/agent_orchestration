"""Thin client for a locally-served Ollama model.

Replaces the old ModelManager: instead of holding model weights in this
Python process, we talk to a model served by Ollama over HTTP. This lets
multiple agent roles share one served model cheaply (no reloading, no
duplicated VRAM) and keeps generation settings (temperature, top_p) fully
per-call instead of baked into a single config.

Prereq: `ollama pull qwen3-coder:30b` and `ollama serve` running locally.
"""

from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


def generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    num_ctx: int = 4096,
) -> str:
    # num_ctx is set explicitly (rather than left at the model's default,
    # which can be much larger) so KV cache stays small and predictable on
    # a 16GB card. Our prompts (system prompt + top-k archive summary) are
    # a few hundred to a couple thousand tokens -- 4096 is generous headroom.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]