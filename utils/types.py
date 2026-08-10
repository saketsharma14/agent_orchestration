
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    speaker: str
    content: str


@dataclass
class AgentSpec:
    name: str
    system_prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256


@dataclass
class TaskState:
    task: str
    messages: List[Message] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
