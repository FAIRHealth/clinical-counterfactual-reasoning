from __future__ import annotations
import asyncio
import re
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


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        params: Optional[GenParams] = None,
        timeout: float = 120.0,
    ):
        super().__init__(
            params or GenParams(temperature=1.0, top_p=None),
            BackendCapabilities(
                supports_logprobs=False,  
                supports_echo=False,
                supports_top_p=False,
            ),
        )
        self.model = model
        if base_url and api_version:
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
            )
        else:
            self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout)
        self._logprobs_supported: Optional[bool] = None

    async def wait_ready(self, timeout_seconds: int = 120) -> None:
        return

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
            max_completion_tokens=max_tok,
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
                print(f"[openai] chat error (attempt {attempt + 1}/3): {exc}")
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
        base_kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=hf_messages,
            max_completion_tokens=max_new_tokens,
            temperature=self.params.temperature if temperature is None else temperature,
        )
        effective_top_p = self.params.top_p if top_p is None else top_p
        if effective_top_p is not None:
            base_kwargs["top_p"] = effective_top_p

        last_err: Optional[Exception] = None
        want_logprobs = self._logprobs_supported is not False
        for attempt in range(3):
            try:
                kwargs = dict(base_kwargs)
                if want_logprobs:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = req_top
                completion = await self.client.chat.completions.create(**kwargs)
                if want_logprobs and self._logprobs_supported is None:
                    self._logprobs_supported = True
                text = completion.choices[0].message.content or ""
                thinking, body = _split_think(text)
                return thinking, body, completion.choices[0]
            except Exception as exc:  # noqa: BLE001
                err_str = str(exc).lower()
                if want_logprobs and ("logprobs" in err_str and ("unsupported" in err_str or "not support" in err_str)):
                    print("[openai] logprobs unsupported for this model; retrying without them.")
                    self._logprobs_supported = False
                    want_logprobs = False
                    continue
                last_err = exc
                print(f"[openai] chat_with_logprobs error (attempt {attempt + 1}/3): {exc}")
                await asyncio.sleep(1.5 * (attempt + 1))
        return "", f"Error: {last_err}", None
