from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .backends import LLMBackend, build_backend


_BACKEND: Optional[LLMBackend] = None


def set_backend(backend: LLMBackend) -> None:
    global _BACKEND
    _BACKEND = backend


def get_backend() -> LLMBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = build_backend()
    return _BACKEND


async def wait_ready(timeout_seconds: int = 180) -> None:
    await get_backend().wait_ready(timeout_seconds=timeout_seconds)


async def chat(
    messages: List[Dict[str, str]],
    max_new_tokens: Optional[int] = None,
    output_json: bool = False,
) -> Tuple[str, str]:
    return await get_backend().chat(messages, max_new_tokens=max_new_tokens, output_json=output_json)


async def chat_batch(
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: Optional[int] = None,
) -> List[Tuple[str, str]]:
    import asyncio

    return await asyncio.gather(
        *(chat(m, max_new_tokens=max_new_tokens) for m in messages_list)
    )


async def chat_with_logprobs(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 8192,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_logprobs: Optional[int] = None,
) -> Tuple[str, str, Any]:
    return await get_backend().chat_with_logprobs(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_logprobs=top_logprobs,
    )


def has_logprobs(choice: Any) -> bool:
    if choice is None:
        return False
    lp = getattr(choice, "logprobs", None)
    if lp is None and isinstance(choice, dict):
        lp = choice.get("logprobs")
    if lp is None:
        return False
    content = getattr(lp, "content", None)
    if content is None and isinstance(lp, dict):
        content = lp.get("content")
    return bool(content)


def extract_logprob_items(choice: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        lp = getattr(choice, "logprobs", None)
        if lp is None and isinstance(choice, dict):
            lp = choice.get("logprobs")
        if lp is None:
            return items
        content = getattr(lp, "content", None)
        if content is None and isinstance(lp, dict):
            content = lp.get("content")
        for it in content or []:
            token = getattr(it, "token", None)
            if token is None and isinstance(it, dict):
                token = it.get("token")
            prob = getattr(it, "logprob", None)
            if prob is None and isinstance(it, dict):
                prob = it.get("logprob")
            top = getattr(it, "top_logprobs", None)
            if top is None and isinstance(it, dict):
                top = it.get("top_logprobs")
            norm_top: List[Dict[str, Any]] = []
            for c in top or []:
                ctok = getattr(c, "token", None)
                if ctok is None and isinstance(c, dict):
                    ctok = c.get("token")
                clp = getattr(c, "logprob", None)
                if clp is None and isinstance(c, dict):
                    clp = c.get("logprob")
                if ctok is not None and clp is not None:
                    norm_top.append({"token": str(ctok), "logprob": float(clp)})
            items.append({
                "token": "" if token is None else str(token),
                "logprob": None if prob is None else float(prob),
                "top_logprobs": norm_top,
            })
    except Exception:
        return []
    return items


def extract_tokens_between_tags(
    all_token_data: List[Any],
    start_tag: str,
    end_tag: str,
) -> Tuple[List[Any], float]:
    try:
        full_text = ""
        positions: List[Tuple[int, int, Any]] = []
        for td in all_token_data:
            start = len(full_text)
            tok = getattr(td, "token", "") or ""
            full_text += tok
            positions.append((start, len(full_text), td))

        start_idx = full_text.find(start_tag)
        if start_idx == -1:
            return [], 0.0
        start_idx += len(start_tag)
        end_idx = full_text.find(end_tag, start_idx)
        if end_idx == -1:
            return [], 0.0

        content_tokens = [td for s, e, td in positions if s >= start_idx and e <= end_idx]
        while content_tokens and getattr(content_tokens[0], "token", "").strip() == "":
            content_tokens = content_tokens[1:]
        while content_tokens and getattr(content_tokens[-1], "token", "").strip() == "":
            content_tokens = content_tokens[:-1]
        content_tokens = [t for t in content_tokens if getattr(t, "token", "").strip() != ""]

        total_lp = 0.0
        for t in content_tokens:
            lp = getattr(t, "logprob", 0.0)
            if lp is not None:
                total_lp += float(lp)
        return content_tokens, total_lp
    except Exception as exc:  # noqa: BLE001
        print(f"  [TOKEN_EXTRACTION_ERROR] {start_tag}..{end_tag}: {exc}")
        return [], 0.0


MAX_NEW_TOKENS_DEFAULT = 32768

async def prompt_based_probability(
    case_text: str,
    label: str,
    role: Optional[str] = None,
    specialist_report: Optional[str] = None,
    debate_summary: Optional[str] = None,
) -> float:
    """Ask the model to self-report a numeric confidence in [0, 1]."""

    role_ctx = f"You are a {role}." if role else "You are a clinical diagnostician."
    extra = ""
    if specialist_report and role:
        extra += f"\n\nYour Specialty Context as {role}:\n{specialist_report}"
    if debate_summary:
        extra += f"\n\Discussion History:\n{debate_summary}"

    system_msg = (
        f"{role_ctx}{extra}\n\n"
        "You will be given a clinical case and a candidate diagnosis. "
        "Estimate the probability (between 0.00 and 1.00) that the candidate "
        "diagnosis is the correct diagnosis for this case. "
        "After your reasoning, you MUST end your response with exactly one line:\n"
        "PROBABILITY: <number>\n"
        "where <number> is a decimal between 0.00 and 1.00."
    )
    user_msg = (
        f"Case:\n{case_text}\n\n"
        f"Candidate Diagnosis: {label}\n\n"
        "Estimate the probability and end with PROBABILITY: <number>"
    )
    _, response = await chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT,
    )
    resp = (response or "").strip()

    m = re.search(r"PROBABILITY\s*[:=]\s*([01](?:\.\d+)?|\.\d+)", resp, re.IGNORECASE)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    nums = re.findall(r"\b([01]\.\d+|0\.\d+|\.\d+)\b", resp)
    if nums:
        val = float(nums[-1])
        if 0.0 <= val <= 1.0:
            return val
    pct = re.search(r"(\d{1,3})\s*%", resp)
    if pct:
        return max(0.0, min(1.0, float(pct.group(1)) / 100.0))

    print(f"  [PROB_PARSE_WARNING] Could not parse probability from: {resp[:200]!r}")
    return 0.5