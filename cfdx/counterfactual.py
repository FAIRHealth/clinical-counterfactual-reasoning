from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from . import llm
from .config import EDIT_SIM_THRESHOLD, MAX_NEW_TOKENS, PRESERVATION_THRESHOLD
from .confidence import diagnose_top1_with_confidence
from .prompts import SPECIALIST_CF_GENERATION_SYSTEM_PROMPT
from .scoring import (
    CounterfactualMetrics,
    combine_scores,
    compute_label_shift_score,
    compute_sip_score,
)
from .utils import (
    clean_diagnosis_label,
    extract_json_from_text,
    format_evidence_item,
    is_plausible_diagnosis_label,
)

MIN_LENGTH_RATIO = 0.70


async def generate_counterfactual_candidates(
    case_presentation: str,
    role: str,
    baseline_diagnosis: str,
    top_ddx_guidance: Optional[List[Dict[str, str]]] = None,
    num_candidates: int = 3,
    summary_context: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    sys_prompt = SPECIALIST_CF_GENERATION_SYSTEM_PROMPT.replace("{role}", role)

    ddx_context = "Not provided"
    if top_ddx_guidance:
        ddx_context = "\n".join(
            f"{i+1}. {d.get('diagnosis', 'Unknown')}" for i, d in enumerate(top_ddx_guidance)
        )
    debate_context = ""
    if summary_context:
        debate_context = (
            f"\nSummarized Discussion History:\n{summary_context}\n\n"
            "Use the discussion history to inform which evidence groups to target for counterfactual testing. Consider what specialists have discussed and what evidence has been contested.\n"
        )

    user_prompt = f"""Propose {num_candidates} hypothesis-testing counterfactual tweaks that help discriminate between diagnoses. 

IMPORTANT INSTRUCTIONS:
- Focus your edits on evidence groups of related discriminating features that distinguish the baseline diagnosis from other differentials. 
- For each counterfactual, you MUST provide the COMPLETE, FULL edited case text. 
- Do not truncate, summarize, or abbreviate the edited case. The edited_case field should contain the entire case with only the targeted evidence group changes applied.

Original Case:
{case_presentation}

Baseline Diagnosis (from original case):
{baseline_diagnosis}

Top 3 Differential Diagnoses to Test Against:
{ddx_context}

{debate_context}

Remember: edited_case = original case with ONLY the target spans modified. Everything else stays EXACTLY the same.
"""

    _, response = await llm.chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_new_tokens=MAX_NEW_TOKENS,
        output_json=True,
    )
    data = extract_json_from_text(response) or {}
    edits = data.get("proposed_edits", []) or []

    log_prefix = f"[{role}|{baseline_diagnosis[:30]}]"
    normalized: List[Dict[str, Any]] = []
    rejected = 0
    original_length = max(1, len(case_presentation))

    for i, e in enumerate(edits):
        edited_case = (e.get("edited_case") or "").strip() or case_presentation
        ratio = len(edited_case) / original_length
        if ratio < MIN_LENGTH_RATIO:
            rejected += 1
            tgt = e.get("target_evidence_group")
            tgt_display = (
                "; ".join(format_evidence_item(it, include_feature_type=False) for it in tgt)
                if isinstance(tgt, list)
                else (e.get("target_span") or "N/A")
            )
            print(
                f"  {log_prefix} [CF REJECTED] #{i+1} truncated "
                f"({len(edited_case)}/{original_length} chars, {ratio:.0%}) target={tgt_display}"
            )
            continue

        tgt = e.get("target_evidence_group")
        if isinstance(tgt, list):
            target_span_display = "; ".join(
                format_evidence_item(it, include_feature_type=False) for it in tgt
            )
        else:
            target_span_display = (e.get("target_span") or "").strip() or "N/A"

        cand = {
            "edit_id": int(e.get("edit_id", i + 1)),
            "operation": (e.get("operation") or "").strip() or "replace",
            "target_span": target_span_display,
            "target_evidence_group": tgt,
            "edited_case": edited_case,
            "rationale": (e.get("rationale") or "").strip(),
        }
        sip = compute_sip_score(case_presentation, edited_case)
        cand["_sip_semantic"] = sip["semantic_similarity"]
        cand["_sip_edit"] = sip["edit_similarity"]
        cand["_sip_preservation"] = sip["preservation_score"]
        normalized.append(cand)

    if rejected:
        print(f"  {log_prefix} [CF FILTER] rejected {rejected}/{len(edits)} (kept {len(normalized)})")

    if not normalized:
        return [], response

    realism_kept = [
        c for c in normalized
        if c["_sip_preservation"] >= PRESERVATION_THRESHOLD and c["_sip_edit"] >= EDIT_SIM_THRESHOLD
    ]
    pool = realism_kept if realism_kept else normalized
    pool.sort(key=lambda x: (x["_sip_preservation"], x["_sip_edit"]), reverse=True)
    cleaned = [
        {
            "edit_id": c["edit_id"],
            "operation": c["operation"],
            "target_span": c["target_span"],
            "target_evidence_group": c["target_evidence_group"],
            "edited_case": c["edited_case"],
            "rationale": c["rationale"],
        }
        for c in pool[:num_candidates]
    ]
    # if cleaned:
    #     print(f"  {log_prefix} [CF FILTER] kept {len(cleaned)}/{len(edits)} candidates")
    return cleaned, response


async def evaluate_counterfactual_candidate(
    original_case: str,
    edited_case: str,
    baseline_label: str,
    baseline_label_prob: float,
    role: Optional[str] = None,
    specialist_report: Optional[str] = None,
    debate_summary: Optional[str] = None,
) -> CounterfactualMetrics:
    cf_label, cf_prob, _ = await diagnose_top1_with_confidence(
        edited_case,
        allowed_labels=None,
        role=role,
        specialist_report=specialist_report,
        debate_summary=debate_summary,
    )
    if not cf_label:
        cf_label = clean_diagnosis_label(baseline_label)
        cf_prob = 0.5

    cpg = abs(baseline_label_prob - cf_prob)
    if is_plausible_diagnosis_label(baseline_label) and is_plausible_diagnosis_label(cf_label):
        cfr = 1.0 if cf_label.lower() != baseline_label.lower() else 0.0
    else:
        cfr = 0.0

    label_shift = compute_label_shift_score(baseline_label, cf_label)
    sip = compute_sip_score(original_case, edited_case)
    combined = combine_scores(cpg, label_shift, sip["preservation_score"], sip["edit_similarity"])

    return CounterfactualMetrics(
        baseline_label=baseline_label,
        baseline_label_prob=baseline_label_prob,
        cf_label=cf_label,
        cf_baseline_label_prob=cf_prob,
        cpg_text=cpg,
        cfr_text=cfr,
        label_shift_score=label_shift,
        semantic_similarity=sip["semantic_similarity"],
        edit_similarity=sip["edit_similarity"],
        preservation_score=sip["preservation_score"],
        combined_score=combined,
    )
