from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from . import llm
from .utils import (
    cached_or_compute,
    clean_diagnosis_label,
    is_plausible_diagnosis_label,
    prob_cache_make_key,
)


async def _logprob_based_probability(
    case_text: str,
    label: str,
    role: Optional[str],
    specialist_report: Optional[str],
    debate_summary: Optional[str],
) -> Tuple[float, str]:
    role_ctx = f"You are a {role}" if role else "You are a clinical diagnostician"
    parts = []
    if specialist_report and role:
        parts.append(f"Your Specialty Context as {role}:\n{specialist_report}")
    if debate_summary and role:
        parts.append(
            f"Discussion History:\n{debate_summary}\n\n"
            "Consider your domain expertise and evolved understanding from the discussion."
        )
    specialist_ctx = "\n\n".join(parts)

    system_prompt = (
        f"{role_ctx}. Evaluate whether the following case matches the candidate diagnosis.\n\n"
        f"{specialist_ctx}\n\n"
        "Given a case presentation and a candidate diagnosis, determine if the case corresponds to that diagnosis.\n"
        "Output your evaluation in this format:\n\n"
        "<final_diagnosis>\n"
        "[Either repeat the candidate diagnosis if it matches, or provide the correct diagnosis if it doesn't]\n"
        "</final_diagnosis>"
    )
    user_prompt = (
        f"Case Presentation:\n{case_text}\n\n"
        f"Candidate Diagnosis: {label}\n\n"
        "Does this case match the candidate diagnosis? Output your final diagnosis within <final_diagnosis></final_diagnosis> tags."
    )

    _, response, choice = await llm.chat_with_logprobs(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_new_tokens=llm.MAX_NEW_TOKENS_DEFAULT,
    )

    if llm.has_logprobs(choice):
        try:
            tokens = choice.logprobs.content  # type: ignore[attr-defined]
            dx_tokens, dx_logprob = llm.extract_tokens_between_tags(
                tokens, "<final_diagnosis>", "</final_diagnosis>"
            )
            if dx_tokens:
                prob = float(np.exp(dx_logprob))
                extracted = "".join(getattr(t, "token", "") for t in dx_tokens).strip()
                print(f"  [PROB/logprobs] '{label}' -> '{extracted}' | P={prob:.4f}")
                return max(0.0, min(1.0, prob)), response or ""
            print(f"  [PROB] no tokens between <final_diagnosis> tags for '{label}'")
        except Exception as exc:  # noqa: BLE001
            print(f"  [PROB] logprob extraction failed for '{label}': {exc}")

    fallback = await llm.prompt_based_probability(
        case_text, label, role=role, specialist_report=specialist_report, debate_summary=debate_summary
    )
    return max(0.0, min(1.0, float(fallback))), response or ""


async def probability_of_label(
    case_text: str,
    label: str,
    role: Optional[str] = None,
    specialist_report: Optional[str] = None,
    debate_summary: Optional[str] = None,
) -> Tuple[float, str]:
    if not is_plausible_diagnosis_label(label):
        return 0.0, "Invalid candidate label"

    key = prob_cache_make_key(case_text, label, role, debate_summary)

    async def compute() -> Tuple[float, str]:
        return await _logprob_based_probability(case_text, label, role, specialist_report, debate_summary)

    return await cached_or_compute(key, compute)


# ---------------------------------------------------------------------------
# Top-1 diagnosis + confidence (used when evaluating edited/original cases)
# ---------------------------------------------------------------------------

async def diagnose_top1_with_confidence(
    case_text: str,
    allowed_labels: Optional[list] = None,  # unused; kept for API compat
    role: Optional[str] = None,
    specialist_report: Optional[str] = None,
    debate_summary: Optional[str] = None,
) -> Tuple[str, float, str]:
    from .prompts import format_diagnosis_prompt  # local to avoid cycle

    system_prompt = format_diagnosis_prompt(role=role, specialist_report=specialist_report, debate_summary=debate_summary)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Case:\n{case_text}\n\nOutput ONLY the single most likely diagnosis label."},
    ]
    _, response, choice = await llm.chat_with_logprobs(messages, max_new_tokens=llm.MAX_NEW_TOKENS_DEFAULT)
    primary_dx = clean_diagnosis_label(response)

    confidence = 0.5
    if llm.has_logprobs(choice):
        try:
            tokens = choice.logprobs.content  # type: ignore[attr-defined]
            start_index = 0
            buf = ""
            for i, td in enumerate(tokens):
                buf += getattr(td, "token", "")
                if "</think>" in buf:
                    start_index = i + 1
                    break
            body = tokens[start_index:]
            while body and getattr(body[0], "token", "").strip() == "":
                body = body[1:]
            specials = ("<|im_end|>", "<|endoftext|>", "<|end|>", "</s>", "<eos>")
            while body and any(s in getattr(body[-1], "token", "") for s in specials):
                body = body[:-1]
            if body:
                total = sum(float(getattr(t, "logprob", 0.0) or 0.0) for t in body)
                confidence = max(0.0, min(1.0, float(np.exp(total))))
        except Exception as exc:  # noqa: BLE001
            print(f"  [DX/logprobs] failed, using prompt fallback: {exc}")
            confidence = await llm.prompt_based_probability(
                case_text, primary_dx, role=role, specialist_report=specialist_report, debate_summary=debate_summary
            )
    else:
        confidence = await llm.prompt_based_probability(
            case_text, primary_dx, role=role, specialist_report=specialist_report, debate_summary=debate_summary
        )

    print(f"  [Eval CF Case Diagnosis] '{primary_dx}' (P={confidence:.4f})")
    return primary_dx, confidence, response
