
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: torch.dtype = torch.float16
    max_new_tokens: int = 256


class ModelManager:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        print("Model loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("ModelManager is not initialized correctly.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        messages = [
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        )

        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()
