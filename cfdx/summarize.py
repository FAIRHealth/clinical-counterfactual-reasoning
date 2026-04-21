from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from . import llm
from .config import MAX_NEW_TOKENS
from .prompts import (
    FINAL_JUDGE_SYSTEM_PROMPT,
    FINAL_SUMMARIZER_SYSTEM_PROMPT,
    FINAL_SUMMARIZER_SYSTEM_PROMPT_ROUND0,
    ROUND_SUMMARIZER_SYSTEM_PROMPT,
)
from .utils import extract_between_tags, extract_json_from_text


def _stamp_final_stance(content: str, final_dx: str) -> str:
    if not final_dx:
        return content
    return content + f"\n\n[CORRECTED FINAL STANCE after stance inertia validation: {final_dx}]"


async def summarize_round(
    all_round_responses: List[Dict[str, Any]],
    round_idx: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, str]:
    included = [r for r in all_round_responses if int(r.get("round", -1)) <= round_idx]
    names: List[str] = []
    blocks: List[str] = []
    for r in included:
        role = r.get("role", "Specialist")
        if role not in names:
            names.append(role)
        content = _stamp_final_stance(r.get("response", ""), r.get("final_diagnosis", ""))
        blocks.append(f"Round {r.get('round')} - {role}:\n{content}")
    context = "\n\n".join(blocks)

    prompt = f"""Summarize rounds 0 to {round_idx} for the following participants: {', '.join(names)}.

Context:
{context}

IMPORTANT: When reporting each specialist's final stance, use the [CORRECTED FINAL STANCE] field if present, as it reflects the validated diagnosis after stance inertia checks. The stance in the main response may have been blocked if it lacked sufficient counterfactual evidence.

Return within <summary_log>...</summary_log> only.
"""
    thinking, response = await llm.chat(
        [
            {"role": "system", "content": ROUND_SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_new_tokens=max_new_tokens,
    )
    return thinking, extract_between_tags(response, "summary_log") or response.strip()


async def summarize_final(
    all_round_responses: List[Dict[str, Any]],
    consensus_round: Optional[int] = None,
) -> Tuple[str, str]:
    blocks: List[str] = []
    for r in all_round_responses:
        content = _stamp_final_stance(r.get("response", ""), r.get("final_diagnosis", ""))
        blocks.append(f"Round {r.get('round')} - {r.get('role')}:\n{content}")
    context = "\n\n".join(blocks)

    user_prompt = f"""Summarize the entire multi-round discussion across specialists with stance evolution and key reasoning highlights.

Context:
{context}

IMPORTANT: When reporting each specialist's final stance for each round, use the [CORRECTED FINAL STANCE] field if present, as it reflects the validated diagnosis after stance inertia checks. The stance in the main response may have been blocked if it lacked sufficient counterfactual evidence.

Return within <summary_log>...</summary_log> only.
"""
    system_prompt = (
        FINAL_SUMMARIZER_SYSTEM_PROMPT_ROUND0
        if consensus_round is not None and consensus_round <= 1
        else FINAL_SUMMARIZER_SYSTEM_PROMPT
    )
    thinking, response = await llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return thinking, extract_between_tags(response, "summary_log") or response.strip()


async def validate_judge_consistency(judge_response: str) -> Tuple[bool, str]:
    try:
        data = extract_json_from_text(judge_response) or {}
        final = (data.get("final_diagnosis") or "").strip()
        rationale = (data.get("rationale") or "").strip()
        if not final or not rationale:
            return False, "Missing final_diagnosis or rationale"
        if final.lower() in rationale.lower():
            return True, ""
        words = [w for w in final.lower().split() if len(w) > 2]
        rw = set(rationale.lower().split())
        if any(w in rw for w in words):
            return True, ""
        return False, f"Rationale does not mention final diagnosis '{final}'"
    except Exception as exc: 
        return False, f"Error validating judge consistency: {exc}"


async def judge_final(
    case_presentation: str,
    final_summary_text: str,
    consensus_flag: bool,
    consensus_dx: str,
    probability_scoreboard: str,
    top_ddx_list: Optional[List[Dict[str, str]]] = None,
    all_round_responses: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    debate_blocks: List[str] = []
    for r in (all_round_responses or []):
        content = r.get("response") or ""
        if content:
            debate_blocks.append(f"Round {r.get('round', 0)} - {r.get('role', '')}:\n{content}")
    debate_text = "\n\n".join(debate_blocks) if debate_blocks else final_summary_text

    if top_ddx_list:
        ddx_text = "\n".join(f"{i+1}. {d.get('diagnosis', 'Unknown')}" for i, d in enumerate(top_ddx_list))
    else:
        ddx_text = "Not provided"

    if consensus_flag and consensus_dx:
        consensus_status = (
            f"had_consensus = True\nconsensus_diagnosis = {consensus_dx}\n"
            f"IMPORTANT: Consensus was reached. You MUST use '{consensus_dx}' as the final_diagnosis. Focus your rationale on summarizing why the clinical evidence supports this diagnosis."
        )
    else:
        consensus_status = "had_consensus = False\nconsensus_diagnosis = None"

    user_prompt = f"""Please decide the most likely final diagnosis given the case and specialists discussion summary

CASE PRESENTATION:
{case_presentation}

TOP 3 DIFFERENTIAL DIAGNOSES (You MUST select from this list):
{ddx_text}

FULL SPECIALISTS DISCUSSION HISTORY:
{debate_text}

CONSENSUS STATUS:
{consensus_status}

SPECIALISTS' COUNTERFACTUAL HYPOTHESIS TESTING RESULTS:
{probability_scoreboard or 'Not provided'}

IMPORTANT RATIONALE INSTRUCTIONS:
Your "rationale" field must be an evidence-grounded diagnostic reasoning chain. Follow this 3-part structure:

(1) CASE ANALYSIS: Concisely highlight the key clinical features that were identified as diagnostically relevant during analysis. Cite SPECIFIC values and results from the original case — do NOT use vague language like "elevated white blood cells" or "imaging showed inflammation." Only include features that matter for the diagnostic reasoning.

(2) EVIDENCE & DIFFERENTIAL NARROWING: For each key finding, explain what diagnosis it supports and what it helps rule out. Ground each reasoning step in the specific case data or established clinical knowledge. Show how differentials were narrowed step-by-step using the evidence based on the discussion history.

(3) CONCLUSION: Synthesize why the totality of evidence converges on the final diagnosis — what combination of findings makes this the most likely diagnosis and why the alternatives do not fit as well.

- NEVER mention specialist names, discussion rounds, agents, or the multi-agent process in the rationale. Write as a single expert clinician.
{"- Since consensus was reached, focus on the shared reasoning that led to convergence on " + consensus_dx + " — ground the reasoning in specific case data." if consensus_flag and consensus_dx else "- Since no consensus was reached, explain the evidence-grounded reasoning for why the chosen diagnosis best fits over competing ones."}

Output JSON only.
"""
    thinking, response = await llm.chat(
        [
            {"role": "system", "content": FINAL_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_new_tokens=MAX_NEW_TOKENS,
        output_json=True,
    )

    is_consistent, error_msg = await validate_judge_consistency(response)
    if not is_consistent:
        print(f"WARNING: Judge consistency check failed: {error_msg}")
        try:
            data = extract_json_from_text(response) or {}
            data["validation_warning"] = error_msg
            response = json.dumps(data, indent=2)
        except Exception:
            pass
    return thinking, response
