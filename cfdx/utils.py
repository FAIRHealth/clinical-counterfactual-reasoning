from __future__ import annotations
import asyncio
import hashlib
import json
import math
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from .config import LABEL_PROB_CACHE_SIZE


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"</?think>", "", out, flags=re.IGNORECASE)
    return out.strip()


def extract_between_tags(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_json_from_text(text: str) -> Optional[Any]:
    if not text:
        return None
    body = strip_think_tags(text).strip()
    if "```json" in body:
        s = body.find("```json") + 7
        e = body.find("```", s)
        if e != -1:
            try:
                return json.loads(body[s:e].strip())
            except Exception:
                pass
    if "```" in body:
        s = body.find("```") + 3
        e = body.rfind("```")
        if e != -1:
            try:
                return json.loads(body[s:e].strip())
            except Exception:
                pass
    first, last = body.find("{"), body.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(body[first : last + 1])
        except Exception:
            return None
    return None


def normalize_label(value: str) -> str:
    return (value or "").strip().lower().replace("  ", " ")


def clean_diagnosis_label(text: str) -> str:
    t = (text or "").strip()
    if "\n" in t:
        t = t.split("\n", 1)[0]
    t = t.strip().strip('"').strip("'").strip()
    if t and t[-1] in {".", ":", ";", ","}:
        t = t[:-1]
    return t


def is_plausible_diagnosis_label(label: str) -> bool:
    lbl = (label or "").strip()
    if not lbl or len(lbl) > 80:
        return False
    if any(ch in lbl for ch in ["\n", ":", ";", ".", ","]):
        return False
    bad_tokens = [
        "okay", "let's", "patient", "presents", "presented", "history",
        "year-old", "days", "day", "exam", "on ", "he ", "she ", "they ",
    ]
    low = lbl.lower()
    if any(bt in low for bt in bad_tokens):
        return False
    words = [w for w in re.split(r"\s+", lbl) if w]
    return 1 <= len(words) <= 30


def labels_equal_ci(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def format_evidence_item(item: Any, include_feature_type: bool = True, bullet: bool = False) -> str:
    prefix = "• " if bullet else ""
    if isinstance(item, dict):
        span = item.get("span", "")
        if include_feature_type:
            ft = item.get("feature_type", "unknown")
            return f"{prefix}{span} ({ft})"
        return f"{prefix}{span}"
    if isinstance(item, str):
        return f"{prefix}{item}"
    return f"{prefix}{item!s}"


def consensus_from_stances(role_to_stance: Dict[str, str]) -> Tuple[bool, str]:
    values: List[str] = []
    for role, stance in (role_to_stance or {}).items():
        if str(role).strip().lower() == "independent clinician":
            continue
        lbl = normalize_label(stance)
        if lbl:
            values.append(lbl)
    if not values:
        return False, ""

    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    name, count = max(counts.items(), key=lambda x: x[1])
    majority = math.ceil(len(values) * 0.75)
    if count >= majority:
        return True, name
    return False, ""


def parse_questions_simplified(
    question_block: str,
    asker_role: str,
    available_roles: List[str],
) -> Dict[str, List[str]]:
    """Parse ``Q-TO-[Role]: ...`` lines into a {target_role: [questions]} dict."""
    out: Dict[str, List[str]] = {role: [] for role in available_roles}
    if not question_block or question_block.strip().lower() == "none":
        return out

    normalized_to_actual: Dict[str, str] = {}
    for role in available_roles:
        normalized_to_actual[role.replace(" ", "_").lower()] = role
        normalized_to_actual[role.lower()] = role

    for raw in question_block.splitlines():
        line = raw.strip()
        if not line or line.lower() == "none":
            continue
        m = re.match(r"^[-•*]?\s*Q-TO-\[(.*?)\]:\s*(.+)$", line, re.IGNORECASE)
        if not m:
            m = re.match(r"^[-•*]?\s*Q-TO-([^:]+):\s*(.+)$", line, re.IGNORECASE)
        if not m:
            continue
        target_raw = m.group(1).strip()
        question = m.group(2).strip()

        target_role: Optional[str] = None
        for role in available_roles:
            if role.lower() == target_raw.lower():
                target_role = role
                break
        if not target_role:
            target_role = normalized_to_actual.get(target_raw.replace(" ", "_").lower())
        if not target_role:
            target_role = normalized_to_actual.get(target_raw.replace("_", " ").lower())
        if not target_role:
            target_words = set(target_raw.lower().split())
            for role in available_roles:
                role_words = set(role.replace("_", " ").lower().split())
                if target_words & role_words and len(target_words & role_words) >= min(2, len(target_words), len(role_words)):
                    target_role = role
                    break

        if target_role:
            out[target_role].append(f"From {asker_role}: {question}")
        else:
            print(f"  [Q-ROUTING WARNING] Unknown target role '{target_raw}' from {asker_role}. Available: {available_roles}")

    return out


def parse_answers_simplified(answer_block: str, answerer_role: str) -> List[Tuple[str, str]]:
    """Parse ``A-TO-[Role]: ...`` lines into a list of (questioner_role, answer)."""
    answers: List[Tuple[str, str]] = []
    if not answer_block or answer_block.strip().lower() == "none":
        return answers
    for raw in answer_block.splitlines():
        line = raw.strip()
        if not line or line.lower() == "none":
            continue
        m = re.match(r"^[-•*]?\s*A-TO-\[(.*?)\]:\s*(.+)$", line, re.IGNORECASE)
        if not m:
            m = re.match(r"^[-•*]?\s*A-TO-([^:]+):\s*(.+)$", line, re.IGNORECASE)
        if not m:
            continue
        answers.append((m.group(1).strip(), m.group(2).strip()))
    return answers


_prob_label_cache: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
_prob_label_inflight: Dict[str, "asyncio.Task[Tuple[float, str]]"] = {}
_prob_cache_lock: asyncio.Lock = asyncio.Lock()


def prob_cache_make_key(
    case_text: str,
    label: str,
    role: Optional[str] = None,
    debate_summary: Optional[str] = None,
) -> str:
    lbl = (label or "").strip().lower()
    role_str = (role or "generic").strip().lower()
    debate_hash = "none"
    if debate_summary:
        try:
            debate_hash = hashlib.sha1(debate_summary.encode("utf-8", "ignore")).hexdigest()[:8]
        except Exception:
            debate_hash = "none"
    case_hash = hashlib.sha1((case_text or "").encode("utf-8", "ignore")).hexdigest()
    return f"{case_hash}::{lbl}::{role_str}::{debate_hash}"


async def cached_or_compute(
    key: str,
    compute_coro_factory,
) -> Tuple[float, str]:
    async with _prob_cache_lock:
        cached = _prob_label_cache.get(key)
        if cached is not None:
            _prob_label_cache.move_to_end(key, last=True)
            return cached
        task = _prob_label_inflight.get(key)
        creator = False
        if task is None:
            task = asyncio.create_task(compute_coro_factory())
            _prob_label_inflight[key] = task
            creator = True

    try:
        result = await task
    except Exception:
        if creator:
            async with _prob_cache_lock:
                _prob_label_inflight.pop(key, None)
        raise

    if creator:
        async with _prob_cache_lock:
            _prob_label_inflight.pop(key, None)
            _prob_label_cache[key] = result
            while len(_prob_label_cache) > LABEL_PROB_CACHE_SIZE:
                _prob_label_cache.popitem(last=False)
    return result
