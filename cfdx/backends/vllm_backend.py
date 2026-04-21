from __future__ import annotations
import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import openai
from .base import BackendCapabilities, GenParams, LLMBackend


_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
_THINK_STRIP_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def _split_think(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    blocks = _THINK_RE.findall(text)
    thinking = "\n\n---\n\n".join(b.strip() for b in blocks) if blocks else ""
    body = _THINK_RE.sub("", text, count=0)
    body = _THINK_STRIP_RE.sub("", body).strip()
    return thinking, body


class VLLMBackend(LLMBackend):
    name = "vllm"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "vllm",
        params: Optional[GenParams] = None,
        timeout: float = 60.0,
    ):
        super().__init__(
            params or GenParams(),
            BackendCapabilities(supports_logprobs=True, supports_echo=True, supports_top_p=True),
        )
        self.model = model
        self.base_url = base_url
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def wait_ready(self, timeout_seconds: int = 120) -> None:
        start = time.time()
        last_error: Optional[Exception] = None
        while time.time() - start < timeout_seconds:
            try:
                await self.client.models.list()
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                await asyncio.sleep(1.0)
        raise RuntimeError(f"vLLM server not reachable after {timeout_seconds}s: {last_error}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        output_json: bool = False,
    ) -> Tuple[str, str]:
        hf_messages = [
            {"role": ("system" if m.get("role") == "developer" else m.get("role", "user")),
             "content": m.get("content", "")}
            for m in messages
        ]
        max_tok = max_new_tokens or self.params.max_new_tokens

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=hf_messages,
            max_tokens=max_tok,
            temperature=self.params.temperature,
        )
        if self.params.top_p is not None:
            kwargs["top_p"] = self.params.top_p
        if output_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                completion = await self.client.chat.completions.create(**kwargs)
                text = completion.choices[0].message.content or ""
                return _split_think(text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"[vllm] chat error (attempt {attempt + 1}/3): {exc}")
                await asyncio.sleep(1.5 * (attempt + 1))
        return "", f"Error: {last_err}"

    async def chat_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: Optional[int] = None,
    ) -> Tuple[str, str, Any]:
        hf_messages = [
            {"role": ("system" if m.get("role") == "developer" else m.get("role", "user")),
             "content": m.get("content", "")}
            for m in messages
        ]
        req_top = max(
            1,
            min(int(top_logprobs or self.params.max_top_logprobs), int(self.params.max_top_logprobs)),
        )
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=hf_messages,
            max_tokens=max_new_tokens,
            temperature=self.params.temperature if temperature is None else temperature,
            extra_body={"logprobs": True, "top_logprobs": req_top},
        )
        effective_top_p = self.params.top_p if top_p is None else top_p
        if effective_top_p is not None:
            kwargs["top_p"] = effective_top_p

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                completion = await self.client.chat.completions.create(**kwargs)
                text = completion.choices[0].message.content or ""
                thinking, body = _split_think(text)
                return thinking, body, completion.choices[0]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"[vllm] chat_with_logprobs error (attempt {attempt + 1}/3): {exc}")
                await asyncio.sleep(1.5 * (attempt + 1))
        return "", f"Error: {last_err}", None

    async def completion_echo(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        logprobs: bool = True,
        echo: bool = True,
    ) -> Any:
        return await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
        )
