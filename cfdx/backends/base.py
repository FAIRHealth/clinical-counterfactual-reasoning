from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GenParams:
    temperature: float = 0.8
    top_p: Optional[float] = 0.95
    max_new_tokens: int = 32768
    max_top_logprobs: int = 5


@dataclass
class BackendCapabilities:
    supports_logprobs: bool = True
    supports_echo: bool = True
    supports_top_p: bool = True


class LLMBackend(abc.ABC):
    name: str = "base"
    params: GenParams = field(default_factory=GenParams)
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def __init__(self, params: GenParams, capabilities: Optional[BackendCapabilities] = None):
        self.params = params
        self.capabilities = capabilities or BackendCapabilities()

    @abc.abstractmethod
    async def wait_ready(self, timeout_seconds: int = 120) -> None:
        """Block until the backend is reachable (or raise)."""

    @abc.abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        output_json: bool = False,
    ) -> Tuple[str, str]:
        """Return ``(thinking, content)`` for a chat completion."""

    @abc.abstractmethod
    async def chat_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: Optional[int] = None,
    ) -> Tuple[str, str, Any]:
        """Return ``(thinking, content, choice)``; ``choice`` carries logprobs if available."""

    async def completion_echo(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        logprobs: bool = True,
        echo: bool = True,
    ) -> Any:
        raise NotImplementedError
