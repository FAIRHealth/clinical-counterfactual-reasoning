from __future__ import annotations
import os
from typing import Optional
from .base import BackendCapabilities, GenParams, LLMBackend
from .openai_backend import OpenAIBackend
from .vllm_backend import VLLMBackend


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def build_backend(kind: Optional[str] = None) -> LLMBackend:
    kind = (kind or os.environ.get("CFDX_BACKEND", "vllm")).strip().lower()

    params = GenParams(
        temperature=_get_float("TEMPERATURE", 0.8 if kind == "vllm" else 1.0),
        top_p=(_get_float("TOP_P", 0.95) if kind == "vllm" else None),
        max_new_tokens=_get_int("MAX_NEW_TOKENS", 32768),
        max_top_logprobs=_get_int("MAX_TOP_LOGPROBS", 5),
    )

    if kind == "vllm":
        model = os.environ.get("VLLM_MODEL") or os.environ.get("MODEL_NAME")
        if not model:
            raise RuntimeError("VLLM_MODEL (or MODEL_NAME) must be set for vllm backend.")
        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8004/v1")
        api_key = os.environ.get("VLLM_API_KEY", "vllm")
        return VLLMBackend(model=model, base_url=base_url, api_key=api_key, params=params)

    if kind == "openai":
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL_NAME")
        if not model:
            raise RuntimeError("OPENAI_MODEL (or MODEL_NAME) must be set for openai backend.")
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for openai backend.")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        api_version = os.environ.get("OPENAI_API_VERSION", "").strip() or None
        return OpenAIBackend(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            params=params,
        )

    raise ValueError(f"Unknown backend kind: {kind!r}. Use 'vllm' or 'openai'.")


__all__ = [
    "BackendCapabilities",
    "GenParams",
    "LLMBackend",
    "OpenAIBackend",
    "VLLMBackend",
    "build_backend",
]
