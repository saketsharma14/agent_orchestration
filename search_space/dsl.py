"""
Optimizer search-space DSL.

A candidate optimizer is represented as a small, constrained JSON-serializable
spec (OptimizerCandidate) built from a fixed set of building blocks. Agents
(Proposer) only ever produce/mutate this spec -- never raw code -- which keeps
the search space bounded and keeps generated candidates safe to render and run.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import json

GradTransform = Literal["raw", "momentum", "sign_momentum"]
SecondMoment = Literal["none", "ema_sq", "max_sq"]
Normalization = Literal["none", "sqrt_eps", "raw_eps"]
WeightDecayMode = Literal["none", "decoupled", "coupled"]


@dataclass
class OptimizerCandidate:
    name: str = "candidate"
    grad_transform: GradTransform = "momentum"
    momentum_beta: float = 0.9
    second_moment: SecondMoment = "ema_sq"
    second_moment_beta: float = 0.999
    normalization: Normalization = "sqrt_eps"
    eps: float = 1e-8
    bias_correction: bool = True
    weight_decay_mode: WeightDecayMode = "decoupled"
    weight_decay: float = 0.0
    lr: float = 1e-3

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "OptimizerCandidate":
        return OptimizerCandidate(**json.loads(s))

    def validate(self) -> tuple[bool, str]:
        """Cheap structural/range validity check (the Critic agent also
        does a semantic pass on top of this). Bounds here MUST match what
        the Proposer and Critic system prompts state -- if they drift apart,
        the Proposer ends up guessing at a constraint it was never actually
        given, which wastes generations on repeated, avoidable rejections."""
        if not (0.0 <= self.momentum_beta < 1.0):
            return False, "momentum_beta must be in [0, 1)"
        if not (0.0 <= self.second_moment_beta < 1.0):
            return False, "second_moment_beta must be in [0, 1)"
        if not (1e-10 <= self.eps <= 1e-4):
            return False, "eps must be in [1e-10, 1e-4]"
        if self.lr <= 0:
            return False, "lr must be > 0"
        if self.normalization != "none" and self.second_moment == "none":
            return False, "normalization requires a second-moment estimate"
        if self.weight_decay_mode != "none" and self.weight_decay <= 0:
            return False, "weight_decay_mode set but weight_decay <= 0"
        if self.weight_decay_mode == "none" and self.weight_decay != 0:
            return False, "weight_decay_mode is none but weight_decay != 0"
        return True, "ok"

    def memory_cost(self) -> int:
        """Number of extra per-parameter state buffers this candidate needs
        (used as the memory term in the fitness function)."""
        cost = 0
        if self.grad_transform in ("momentum", "sign_momentum"):
            cost += 1
        if self.second_moment != "none":
            cost += 1
        return cost

    def compute_cost(self) -> int:
        """Rough op-count proxy: counts expensive ops (sqrt/div) implied by
        this configuration. Static, not measured -- used as a cheap fitness
        term the Critic can check before an evaluation run."""
        cost = 0
        if self.normalization == "sqrt_eps":
            cost += 2  # sqrt + div
        elif self.normalization == "raw_eps":
            cost += 1  # div only
        if self.bias_correction:
            cost += 1
        return cost


def render_optimizer_code(candidate: OptimizerCandidate) -> str:
    """Render a candidate spec into a standalone torch.optim.Optimizer
    subclass. This is the only place DSL specs turn into executable code --
    the LLM agents never write this code directly, they only ever produce
    the constrained JSON spec above."""

    class_name = "".join(w.capitalize() for w in candidate.name.split("_")) or "Candidate"

    lines = []
    lines.append("import torch")
    lines.append("from torch.optim import Optimizer")
    lines.append("")
    lines.append(f"class {class_name}(Optimizer):")
    lines.append(f"    def __init__(self, params, lr={candidate.lr!r}):")
    lines.append("        defaults = dict(lr=lr)")
    lines.append("        super().__init__(params, defaults)")
    lines.append("")
    lines.append("    @torch.no_grad()")
    lines.append("    def step(self, closure=None):")
    lines.append("        loss = None")
    lines.append("        if closure is not None:")
    lines.append("            with torch.enable_grad():")
    lines.append("                loss = closure()")
    lines.append("        for group in self.param_groups:")
    lines.append("            lr = group['lr']")
    lines.append("            for p in group['params']:")
    lines.append("                if p.grad is None:")
    lines.append("                    continue")
    lines.append("                grad = p.grad")
    lines.append("                state = self.state[p]")
    lines.append("                if len(state) == 0:")
    lines.append("                    state['step'] = 0")
    if candidate.grad_transform in ("momentum", "sign_momentum"):
        lines.append("                    state['m'] = torch.zeros_like(p)")
    if candidate.second_moment != "none":
        lines.append("                    state['v'] = torch.zeros_like(p)")
    lines.append("                state['step'] += 1")
    lines.append("                t = state['step']")
    lines.append("")

    if candidate.weight_decay_mode == "coupled":
        lines.append(f"                grad = grad.add(p, alpha={candidate.weight_decay!r})")

    if candidate.grad_transform == "raw":
        lines.append("                m = grad")
    elif candidate.grad_transform == "momentum":
        lines.append(
            f"                state['m'].mul_({candidate.momentum_beta!r}).add_(grad, alpha={1 - candidate.momentum_beta!r})"
        )
        lines.append("                m = state['m']")
    elif candidate.grad_transform == "sign_momentum":
        lines.append(
            f"                state['m'].mul_({candidate.momentum_beta!r}).add_(grad, alpha={1 - candidate.momentum_beta!r})"
        )
        lines.append("                m = state['m'].sign()")

    if candidate.second_moment == "ema_sq":
        lines.append(
            f"                state['v'].mul_({candidate.second_moment_beta!r}).addcmul_(grad, grad, value={1 - candidate.second_moment_beta!r})"
        )
        lines.append("                v = state['v']")
    elif candidate.second_moment == "max_sq":
        lines.append("                state['v'] = torch.maximum(state['v'], grad * grad)")
        lines.append("                v = state['v']")

    if candidate.grad_transform == "momentum" and candidate.bias_correction:
        lines.append(f"                m_hat = m / (1 - {candidate.momentum_beta!r} ** t)")
    else:
        lines.append("                m_hat = m")

    if candidate.second_moment != "none":
        if candidate.bias_correction and candidate.second_moment == "ema_sq":
            lines.append(f"                v_hat = v / (1 - {candidate.second_moment_beta!r} ** t)")
        else:
            lines.append("                v_hat = v")
    else:
        lines.append("                v_hat = None")

    if candidate.normalization == "sqrt_eps":
        lines.append(f"                update = m_hat / (v_hat.sqrt() + {candidate.eps!r})")
    elif candidate.normalization == "raw_eps":
        lines.append(f"                update = m_hat / (v_hat + {candidate.eps!r})")
    else:
        lines.append("                update = m_hat")

    lines.append("                p.add_(update, alpha=-lr)")

    if candidate.weight_decay_mode == "decoupled":
        lines.append(f"                p.add_(p, alpha=-lr * {candidate.weight_decay!r})")

    lines.append("        return loss")
    lines.append("")

    return "\n".join(lines)