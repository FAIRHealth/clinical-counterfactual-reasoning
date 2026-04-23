from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from . import llm
from .config import MAX_NEW_TOKENS
from .confidence import probability_of_label
from .counterfactual import (
    evaluate_counterfactual_candidate,
    generate_counterfactual_candidates,
)
from .prompts import (
    AUDIT_CF_SYSTEM_PROMPT,
    DDX_TOPK_SYSTEM_PROMPT,
    SPECIALIST_CF_SYNTHESIS_SYSTEM_PROMPT,
    SPECIALIST_POOL,
    SPECIALIST_REPORT_SYSTEM_PROMPT,
    TRIAGE_ROLE_ASSIGNMENT_SYSTEM_PROMPT,
    format_operation_description,
)
from .utils import (
    clean_diagnosis_label,
    extract_between_tags,
    extract_json_from_text,
    format_evidence_item,
    is_plausible_diagnosis_label,
    normalize_label,
)


# ---------------------------------------------------------------------------
# Triage / DDx / Report
# ---------------------------------------------------------------------------

async def assign_specialists_and_problems(case_presentation: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": TRIAGE_ROLE_ASSIGNMENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"From the following patient case, detect main symptoms, problems, and assign up to five most relevant specialists.\n\nCase:\n{case_presentation}\n"},
    ]
    _, response = await llm.chat(messages, max_new_tokens=MAX_NEW_TOKENS, output_json=True)
    data = extract_json_from_text(response) or {}

    filtered: List[Dict[str, str]] = []
    for item in data.get("assigned_specialists", []) or []:
        role = (item.get("role") or "").strip()
        rationale = (item.get("rationale") or "").strip()
        if role in SPECIALIST_POOL:
            filtered.append({"role": role, "rationale": rationale})
    if not filtered:
        filtered = [{"role": "General Internal Medicine Doctor",
                     "rationale": "Baseline triage fallback."}]
    filtered = filtered[:5]

    return {
        "main_symptoms": data.get("main_symptoms", []) or [],
        "problems": data.get("problems", []) or [],
        "assigned_specialists": filtered,
        "num_agents": len(filtered),
    }


async def generate_specialist_report(case_presentation: str, role: str) -> Tuple[str, str]:
    messages = [
        {"role": "system", "content": SPECIALIST_REPORT_SYSTEM_PROMPT.format(role=role)},
        {"role": "user", "content": f"Generate your domain-specific report from this case.\n\nCase:\n{case_presentation}\n"},
    ]
    _, response = await llm.chat(messages, max_new_tokens=MAX_NEW_TOKENS)
    report = extract_between_tags(response, "report") or response.strip()
    return report, response


async def generate_top_k_ddx(case_presentation: str, k: int = 3) -> Tuple[List[Dict[str, str]], str, str]:
    messages = [
        {"role": "system", "content": DDX_TOPK_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the following case and provide the top {k} differential diagnoses.\n\nCase:\n{case_presentation}\n\nOutput JSON only.\n"},
    ]
    _, response = await llm.chat(messages, max_new_tokens=MAX_NEW_TOKENS, output_json=True)
    data = extract_json_from_text(response) or {}
    case_summary = data.get("case_summary", "")
    ddx_raw = data.get("most_likely_diagnoses", []) or []
    cleaned: List[Dict[str, str]] = []
    for item in ddx_raw[:k]:
        dx = clean_diagnosis_label(item.get("diagnosis", ""))
        if dx:
            cleaned.append({"diagnosis": dx, "rationale": (item.get("rationale") or "").strip()})
    if not cleaned:
        cleaned = [{"diagnosis": "Unknown", "rationale": "Unable to generate differential diagnoses"}]
    return cleaned, case_summary, response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_ddx_guidance(top_ddx_guidance: Optional[List[Dict[str, str]]]) -> str:
    if not top_ddx_guidance:
        return "Not provided"
    lines: List[str] = []
    for idx, ddx in enumerate(top_ddx_guidance, 1):
        lines.append(f"{idx}. {ddx.get('diagnosis', 'Unknown')}")
        rationale = ddx.get("rationale") or ""
        if rationale:
            lines.append(f"   Rationale: {rationale}")
    return "\n".join(lines)


def _format_cf_block(cf_data: Dict[str, Any]) -> str:
    cand = cf_data["candidate"]
    scores = cf_data["metrics"]["scores"]
    baseline_label = cf_data["metrics"]["baseline"]["diagnosis"]
    baseline_prob = cf_data["metrics"]["baseline"]["probability_of_baseline_label"] or 0.0
    cf_label = cf_data["metrics"]["counterfactual"]["diagnosis"]
    cf_prob = cf_data["metrics"]["counterfactual"]["probability_of_baseline_label"] or 0.0

    tag = " (unchanged)" if cf_label.lower() == baseline_label.lower() else f" -> {cf_label}"

    tgt = cand.get("target_evidence_group")
    if isinstance(tgt, list):
        evidence = "\n  ".join(format_evidence_item(it, include_feature_type=True, bullet=True) for it in tgt)
        target_display = f"Target Evidence Group Modified:\n  {evidence}"
    else:
        target_display = f"Target Evidence Group Modified: {(cand.get('target_span', 'N/A') or '')[:100]}"

    op_desc = format_operation_description(cand.get("operation", "N/A"), cand.get("hypothesis_tested", "N/A"), baseline_label)
    cpg = scores["CPG"]
    if cpg > 0.2:
        cpg_interp = "PRIMARY evidence group (critical impact)"
    elif cpg > 0.1:
        cpg_interp = "SUPPORTING evidence group (moderate impact)"
    else:
        cpg_interp = "NON-DISCRIMINATING (minimal impact)"
    shift_flag = " (diagnosis changed!)" if scores["label_shift_score"] > 0.3 else ""

    return (
        f"Counterfactual #{cf_data['rank']} - Hypothesis Testing Results (Combined Score: {scores['combined_score']:.3f})\n"
        f"- **Counterfactual Operation**: {op_desc}\n"
        f"- Original Diagnosis Used to Extract Target Evidence Group: {baseline_label}\n"
        f"- {target_display}\n"
        f"- Rationale: {cand.get('rationale', 'N/A')}\n"
        f"- **Diagnosis Impact**: {baseline_label} (P={baseline_prob:.3f}){tag} (P={cf_prob:.3f})\n"
        f"- **CPG (Probability Gap)**: {cpg:.3f}\n"
        f"  └─ Interpretation: {cpg_interp}\n"
        f"- Label Shift Semantic Score: {scores['label_shift_score']:.3f}{shift_flag}\n"
    )


def _assigned_roles_block(role: str, assigned_roles: Optional[List[str]], peers: Optional[Dict[str, str]]) -> str:
    if assigned_roles:
        roles = sorted(set(assigned_roles))
    else:
        roles = sorted(set(list((peers or {}).keys()) + [role]))
    return f"""
        Available Specialist Roles (for asking questions):
        {', '.join(roles)}

        IMPORTANT: When asking questions using the format "Q-TO-[RoleName]: question?", you MUST only use roles from the list above. Do NOT ask questions to yourself (your role is "{role}").
        """


def _confidence_band(tag_text: str, prob: Optional[float]) -> str:
    t = (tag_text or "").strip().lower()
    if t.startswith("high"):
        return "High"
    if t.startswith("moderate"):
        return "Moderate"
    if t.startswith("low"):
        return "Low"
    p = 0.5 if prob is None else float(prob)
    if p >= 0.66:
        return "High"
    if p >= 0.33:
        return "Moderate"
    return "Low"


async def _extract_final_dx_prob(response: str, choice: Any, case_text: str,
                                  role: Optional[str] = None,
                                  specialist_report: Optional[str] = None,
                                  debate_summary: Optional[str] = None) -> float:
    final_dx_raw = extract_between_tags(response, "final_diagnosis") or ""
    if llm.has_logprobs(choice):
        try:
            tokens = choice.logprobs.content  # type: ignore[attr-defined]
            dx_tokens, dx_logprob = llm.extract_tokens_between_tags(
                tokens, "<final_diagnosis>", "</final_diagnosis>"
            )
            if dx_tokens:
                return max(0.0, min(1.0, float(np.exp(dx_logprob))))
        except Exception as exc:  # noqa: BLE001
            print(f"  [FINAL_DX_PROB] logprob extraction failed: {exc}")

    final_dx = clean_diagnosis_label(final_dx_raw)
    if not is_plausible_diagnosis_label(final_dx):
        return 0.5
    prob = await llm.prompt_based_probability(
        case_text, final_dx, role=role, specialist_report=specialist_report, debate_summary=debate_summary
    )
    return max(0.0, min(1.0, float(prob)))


# ---------------------------------------------------------------------------
# Specialist round
# ---------------------------------------------------------------------------

async def _generate_cfs_for_ddx(
    case_presentation: str,
    role: str,
    top_ddx_guidance: Optional[List[Dict[str, str]]],
    num_candidates: int,
    summary_context: str,
    fallback_baseline: str,
) -> Tuple[List[Dict[str, Any]], str]:
    all_candidates: List[Dict[str, Any]] = []
    raw_chunks: List[str] = []

    if top_ddx_guidance and len(top_ddx_guidance) >= 2:
        for ddx_item in top_ddx_guidance:
            ddx_label = ddx_item.get("diagnosis", "")
            if not ddx_label:
                continue
            cands, raw = await generate_counterfactual_candidates(
                case_presentation=case_presentation,
                role=role,
                baseline_diagnosis=ddx_label,
                top_ddx_guidance=top_ddx_guidance,
                num_candidates=num_candidates,
                summary_context=summary_context,
            )
            for c in cands:
                c["tested_diagnosis"] = ddx_label
            all_candidates.extend(cands)
            raw_chunks.append(f"CFs for {ddx_label}:\n{raw}")
        return all_candidates, "\n\n---\n\n".join(raw_chunks)

    cands, raw = await generate_counterfactual_candidates(
        case_presentation=case_presentation,
        role=role,
        baseline_diagnosis=fallback_baseline,
        top_ddx_guidance=top_ddx_guidance,
        num_candidates=max(5, num_candidates + 2),
        summary_context=summary_context,
    )
    for c in cands:
        c["tested_diagnosis"] = fallback_baseline
    return cands, raw


async def specialist_round_with_counterfactual(
    role: str,
    case_presentation: str,
    all_reports_text: str,
    round_idx: int,
    summary_context: str,
    top_ddx_guidance: Optional[List[Dict[str, str]]] = None,
    incoming_questions_for_role: Optional[List[str]] = None,
    peers_latest_stances: Optional[Dict[str, str]] = None,
    previous_round_stance: Optional[str] = None,
    previous_round_diagnosis_prob: Optional[float] = None,
    num_candidates: int = 3,
    specialist_report: Optional[str] = None,
    assigned_specialist_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    allowed_labels = [d.get("diagnosis", "") for d in (top_ddx_guidance or []) if d.get("diagnosis")]

    if round_idx == 0:
        base_label = allowed_labels[0] if allowed_labels else "Unknown"
    else:
        base_label = previous_round_stance or (allowed_labels[0] if allowed_labels else "Unknown")
        if not is_plausible_diagnosis_label(base_label):
            base_label = allowed_labels[0] if allowed_labels else "Unknown"

    print(f"  [{role}] Round {round_idx}: generating CFs for DDx hypotheses")
    candidates, _ = await _generate_cfs_for_ddx(
        case_presentation=case_presentation,
        role=role,
        top_ddx_guidance=top_ddx_guidance,
        num_candidates=num_candidates,
        summary_context=summary_context,
        fallback_baseline=base_label,
    )

    if not candidates:
        edited = case_presentation.replace("fever", "").replace("Fever", "")
        candidates = [{
            "edit_id": 1,
            "operation": "remove",
            "target_span": "fever",
            "tested_diagnosis": base_label,
            "edited_case": edited,
            "rationale": "Fallback minimal removal to test sensitivity to fever",
        }]

    if round_idx == 0:
        eval_tasks = []
        for cand in candidates:
            tested_hyp = cand.get("tested_diagnosis", base_label)
            hyp_prob, _ = await probability_of_label(
                case_presentation, tested_hyp,
                role=role, specialist_report=specialist_report, debate_summary=summary_context,
            )
            eval_tasks.append(evaluate_counterfactual_candidate(
                original_case=case_presentation,
                edited_case=cand["edited_case"],
                baseline_label=tested_hyp,
                baseline_label_prob=hyp_prob,
                role=role,
                specialist_report=specialist_report,
                debate_summary=summary_context,
            ))
    else:
        baseline_label_for_cf = previous_round_stance or base_label
        baseline_prob_for_cf = previous_round_diagnosis_prob if previous_round_diagnosis_prob is not None else 0.5
        print(f"  [{role}] Round {round_idx}: evaluating against '{baseline_label_for_cf}' (P={baseline_prob_for_cf:.3f})")
        eval_tasks = [
            evaluate_counterfactual_candidate(
                original_case=case_presentation,
                edited_case=cand["edited_case"],
                baseline_label=baseline_label_for_cf,
                baseline_label_prob=baseline_prob_for_cf,
                role=role,
                specialist_report=specialist_report,
                debate_summary=summary_context,
            )
            for cand in candidates
        ]

    metrics_list = await asyncio.gather(*eval_tasks)
    scored = sorted(zip(candidates, metrics_list), key=lambda x: x[1].combined_score, reverse=True)
    top_cfs = scored[: min(3, len(scored))]

    top3_cf_data: List[Dict[str, Any]] = []
    for rank, (cand, metrics) in enumerate(top_cfs, 1):
        top3_cf_data.append({
            "rank": rank,
            "candidate": cand,
            "metrics": {
                "baseline": {
                    "diagnosis": metrics.baseline_label,
                    "probability_of_baseline_label": metrics.baseline_label_prob,
                },
                "counterfactual": {
                    "diagnosis": metrics.cf_label,
                    "probability_of_baseline_label": metrics.cf_baseline_label_prob,
                },
                "scores": {
                    "CPG": metrics.cpg_text,
                    "CFR": metrics.cfr_text,
                    "label_shift_score": metrics.label_shift_score,
                    "semantic_similarity": metrics.semantic_similarity,
                    "edit_similarity": metrics.edit_similarity,
                    "preservation_score": metrics.preservation_score,
                    "combined_score": metrics.combined_score,
                },
            },
        })

    top3_cf_text = (
        "\n\n---\n\n".join(_format_cf_block(d) for d in top3_cf_data)
        if top3_cf_data else "No counterfactuals available"
    )

    peers_text = (
        "None" if not peers_latest_stances
        else "\n".join(f"- {k}: {v}" for k, v in peers_latest_stances.items())
    )
    ddx_guidance_text = _format_ddx_guidance(top_ddx_guidance)
    assigned_roles_text = _assigned_roles_block(role, assigned_specialist_roles, peers_latest_stances)

    if round_idx == 0:
        round_instructions = (
            "Round 0: Do NOT ask counterargument questions. Set <counterargument_question> and "
            "<counterargument_answer> to 'None'. Provide your final stance and confidence."
        )
    elif round_idx == 1:
        round_instructions = (
            f"Round {round_idx}: Update your reasoning after reviewing the discussion history. Propose up to 3 counterargument questions. Provide your final stance and confidence."
        )
    else:
        round_instructions = (
            f"Round {round_idx}: Update your reasoning after reviewing the discussion history. "
            f"**YOU MUST answer ALL questions directed to you in the summarized discussion history above.** "
            f"Look for questions with to=\"{role}\" in the <question> tags. "
            f"Use format: A-TO-[AskerRole]: [answer]. "
            f"Check the 'from' attribute in the question tags to identify who asked each question. "
            f"Then propose up to 3 counterargument questions in <counterargument_question> tag. "
            f"Provide your final stance and confidence."
        )

    if round_idx == 0:
        synthesis_user = f"""
        Original Case:
        {case_presentation}

        All Specialists' Reports:
        {all_reports_text}

        Top 3 Differential Diagnoses (Guidance):
        {ddx_guidance_text}

        Top 3 Counterfactual (CF) Hypothesis Testing Results:
        {top3_cf_text}

        {assigned_roles_text}

        {round_instructions}
        CRITICAL INSTRUCTIONS FOR COUNTERFACTUAL REASONING:
        1. **Understand what the CFs test**: Each CF removes/modifies a GROUP of clinical evidence related to the original diagnosis and shows:
           - Whether the diagnosis CHANGES (shown as "→ New Diagnosis" or "unchanged")
           - Whether the confidence CHANGES (CPG = probability gap)
        
        2. **Interpret the metrics**:
           - **CPG > 0.2**: PRIMARY evidence group (removing it causes >20% confidence change) → CRITICAL
           - **CPG 0.1-0.2**: SUPPORTING evidence group (moderate impact) → Important but not essential  
           - **CPG < 0.1**: NON-DISCRIMINATING evidence group (minimal impact) → Not diagnostically useful
           - **Label Shift > 0.3**: Removing this evidence group makes model switch to different diagnosis
        
        3. **Synthesize evidence across all 3 CFs**:
           - Which evidence groups consistently show high CPG? → These are PRIMARY evidence clusters for your diagnosis
           - Which evidence groups cause diagnosis changes? → These are critical discriminators
           - Which evidence groups have low CPG? → These are NOT supporting your diagnosis
        
        4. **Consider diagnosis changes**:
           - If a CF shows a diagnosis change after editing the extracted evidence group in "Diagnosis Impact", that evidence group may be critical for supporting the original diagnosis
           - If multiple CFs shows no diagnosis changes, that evidence group may be non-discriminating to support the original diagnosis, reconsider the original diagnosis as your stance and select another diagnosis from the DDx list as your final diagnosis.
        
        5. **Clinical reasoning > Probability**: 
           - Don't blindly follow probability; evaluate based on which evidence groups are truly causal
           - Evidence groups with high CPG/label shift are the ones that matter clinically
        
        6. **Match diagnostic criteria**: Does your diagnosis match established criteria given the PRIMARY evidence groups?
        
        7. **Explain initial symptoms**: Your diagnosis must explain WHY initial symptoms occurred based on PRIMARY evidence groups

        Round 0 Instructions:
        - Use the top 3 differential diagnoses as guidance, but choose based on clinical reasoning from your specialty's perspective.
        - Do NOT ask or answer counterarguments. Set <counterargument_question> and <counterargument_answer> to "None".

        IMPORTANT: The CFs show what happens when evidence groups are removed - use this to identify which evidence groups are PRIMARY (high CPG/label shift) vs. NON-DISCRIMINATING (low CPG/label shift) for your diagnosis.
        """
    else:
        synthesis_user = f"""
        Original Case:
        {case_presentation}

        All Specialists' Reports:
        {all_reports_text}

        Top 3 Differential Diagnoses (Guidance):
        {ddx_guidance_text}

        Top 3 Counterfactual (CF) Hypothesis Testing Results:
        {top3_cf_text}

        {assigned_roles_text}

        Summarized Discussion So Far:
        {summary_context or 'None'}

        Other Specialists' Latest Stances:
        {peers_text}

        {round_instructions}

        CRITICAL INSTRUCTIONS FOR COUNTERFACTUAL REASONING:
        1. **Understand what the CFs test**: Each CF removes/modifies a GROUP of related clinical features and shows:
           - Whether the diagnosis CHANGES after Counterfactual Operation (shown as "→ New Diagnosis" or "unchanged")
           - Whether the confidence CHANGES (CPG = probability gap)
        
        2. **Interpret the metrics**:
           - **CPG > 0.2**: PRIMARY evidence group (removing it causes >20% confidence change) → CRITICAL
           - **CPG 0.1-0.2**: SUPPORTING evidence group (moderate impact) → Important but not essential  
           - **CPG < 0.1**: NON-DISCRIMINATING evidence group (minimal impact) → Not diagnostically useful
           - **Label Shift > 0.3**: Removing this evidence group makes model switch to different diagnosis
        
        3. **Synthesize evidence across all 3 CFs**:
           - Which evidence groups consistently show high CPG? → These are PRIMARY evidence clusters for your diagnosis
           - Which evidence groups cause diagnosis changes? → These are critical discriminators
           - Which evidence groups have low CPG? → These are NOT supporting your diagnosis
        
        4. **Consider diagnosis changes**:
           - If a CF shows a high CPG score after editing the extracted evidence group in "Diagnosis Impact", that evidence group may be critical for supporting the original diagnosis
           - If multiple CFs shows no diagnosis changes, that evidence group may be non-discriminating to support the original diagnosis, reconsider the original diagnosis as your stance and select another diagnosis from the DDx list as your final diagnosis.
        
        5. **Clinical reasoning > Probability**: 
           - Don't blindly follow probability; evaluate based on which evidence groups are truly causal
           - Evidence groups with high CPG/label shift are the ones that matter clinically
        
        6. **Match diagnostic criteria**: Does your diagnosis match established criteria given the PRIMARY evidence groups?
        
        7. **Explain initial symptoms**: Your diagnosis must explain WHY initial symptoms occurred based on PRIMARY evidence groups

        IMPORTANT: The CFs show what happens when evidence groups are removed - use this to identify which evidence groups are PRIMARY (high CPG/label shift) vs. NON-DISCRIMINATING (low CPG/label shift) for your diagnosis.
        """

    sys_prompt = SPECIALIST_CF_SYNTHESIS_SYSTEM_PROMPT.replace("{role}", role)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": synthesis_user},
    ]

    thinking, response, choice = await llm.chat_with_logprobs(messages, max_new_tokens=MAX_NEW_TOKENS)

    final_dx_raw = extract_between_tags(response, "final_diagnosis") or ""
    final_dx = clean_diagnosis_label(final_dx_raw)

    if not is_plausible_diagnosis_label(final_dx):
        final_dx = allowed_labels[0] if allowed_labels else (previous_round_stance or "Unknown")

    final_dx_prob = await _extract_final_dx_prob(
        response, choice, case_presentation,
        role=role, specialist_report=specialist_report, debate_summary=summary_context,
    )

    if round_idx > 1 and previous_round_stance and is_plausible_diagnosis_label(previous_round_stance) and top3_cf_data:
        prev_norm = normalize_label(previous_round_stance)
        new_norm = normalize_label(final_dx)
        if prev_norm != new_norm:
            CPG_THRESHOLD = 0.20
            MIN_SUPPORTING = 3
            supporting = sum(
                1 for d in top3_cf_data
                if normalize_label(d["candidate"].get("tested_diagnosis", "")) == new_norm
                and d["metrics"]["scores"].get("CPG", 0.0) > CPG_THRESHOLD
            )
            if supporting < MIN_SUPPORTING:
                print(f"  [STANCE INERTIA] {role}: blocking switch '{previous_round_stance}' -> '{final_dx}'")
                final_dx = previous_round_stance

    confidence_tag = extract_between_tags(response, "confidence")
    stance_conf = _confidence_band(confidence_tag, final_dx_prob)

    tested_hypotheses = list({cf["candidate"].get("tested_diagnosis", "") for cf in top3_cf_data if cf["candidate"].get("tested_diagnosis")})
    cf_baseline_used = previous_round_stance if round_idx > 0 else final_dx

    return {
        "role": role,
        "round": round_idx,
        "response": response,
        "thinking": thinking,
        "stance": final_dx,
        "final_diagnosis": final_dx,
        "final_diagnosis_probability": final_dx_prob,
        "confidence": stance_conf,
        "cf_baseline_used": cf_baseline_used,
        "previous_round_stance": previous_round_stance,
        "cf_hypotheses_tested": {
            "current_stance": cf_baseline_used,
            "top3_tested": tested_hypotheses,
            "num_alternative_in_top3": sum(
                1 for h in tested_hypotheses if (cf_baseline_used or "").lower() != (h or "").lower()
            ),
        },
        "counterargument_question": extract_between_tags(response, "counterargument_question") or ("None" if round_idx == 0 else ""),
        "counterargument_answer": extract_between_tags(response, "counterargument_answer") or ("None" if round_idx <= 1 else ""),
        "top3_counterfactuals": top3_cf_data,
        "all_candidates_evaluated": [
            {
                "candidate": c,
                "metrics": {
                    "baseline_label": m.baseline_label,
                    "baseline_label_prob": m.baseline_label_prob,
                    "cf_label": m.cf_label,
                    "cf_baseline_label_prob": m.cf_baseline_label_prob,
                    "CPG": m.cpg_text,
                    "CFR": m.cfr_text,
                    "label_shift_score": m.label_shift_score,
                    "semantic_similarity": m.semantic_similarity,
                    "edit_similarity": m.edit_similarity,
                    "preservation_score": m.preservation_score,
                    "combined_score": m.combined_score,
                },
            }
            for c, m in zip(candidates, metrics_list)
        ],
    }


# ---------------------------------------------------------------------------
# Independent-clinician (audit) round
# ---------------------------------------------------------------------------

async def audit_round_with_counterfactual(
    case_presentation: str,
    all_reports_text: str,
    round_idx: int,
    summary_context: str,
    peers_latest_stances: Optional[Dict[str, str]] = None,
    previous_round_stance: Optional[str] = None,
    previous_round_diagnosis_prob: Optional[float] = None,
    assigned_specialist_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    role = "Independent Clinician"
    peers_text = (
        "None" if not peers_latest_stances
        else "\n".join(f"- {k}: {v}" for k, v in peers_latest_stances.items())
    )

    audit_scoreboard = "Not available"
    try:
        if round_idx > 0 and peers_latest_stances:
            ddx_labels = list({v for v in peers_latest_stances.values() if v})[:3]
            if ddx_labels:
                results = await asyncio.gather(*[probability_of_label(case_presentation, dx) for dx in ddx_labels])
                pairs = sorted(zip(ddx_labels, (float(p) for p, _ in results)), key=lambda x: x[1], reverse=True)
                audit_scoreboard = "\n".join(f"- {dx}: P={p:.3f}" for dx, p in pairs)
    except Exception:
        audit_scoreboard = "Not available"

    if round_idx == 0:
        round_instructions = (
            "Round 0: Focus on INITIAL symptom(s) and timeline. Do NOT ask counterargument questions. "
            "Set <counterargument_question> and <counterargument_answer> to 'None'. "
            "Provide your final stance and confidence."
        )
    elif round_idx == 1:
        round_instructions = (
            f"Round {round_idx}: Update your reasoning after reviewing the discussion history, emphasize initial symptom(s) and timeline parsimony. Propose up to 3 counterargument questions. Provide your final stance and confidence."
        )
    else:
        round_instructions = (
            f"Round {round_idx}: Update your reasoning after reviewing the discussion history. **YOU MUST answer ALL questions directed to you in the summarized discussion history above.** Look for questions with to=\"Independent Clinician\" in the <question> tags. Use format: A-TO-[AskerRole]: [answer]. Check the 'from' attribute in the question tags to identify who asked each question. For multi-word role names, use brackets: A-TO-[General Internal Medicine Doctor]: [answer]. Then propose up to 3 counterargument questions in <counterargument_question> tag. Emphasize primary cause vs downstream, and timeline consistency. Provide your final stance and confidence."
        )

    assigned_roles_text = _assigned_roles_block(role, assigned_specialist_roles, peers_latest_stances)

    if round_idx == 0:
        synthesis_user = f"""
        Original Case:
        {case_presentation}

        All Specialists' Reports:
        {all_reports_text}

        {assigned_roles_text}

        {round_instructions}

        Round 0 Instructions:
        - This is the initial round. Focus on proposing the most likely final diagnosis based on the case presentation and all specialists' domain reports.
        - Prioritize the patient's initial symptom(s) at the time of first hospital visit and the overall presentation timeline.
        - Push reverse thinking: reason backwards from the first symptom to likely primary causes.
        - Do NOT ask or answer counterarguments. Set <counterargument_question> and <counterargument_answer> to "None".
        - There are no other specialists' stances to critique yet. Set <critique> to "None".

        Guidance:
        - Focus on initial symptoms and timeline consistency when imaging/labs are ambiguous.
        - Consider whether findings could be SECONDARY to an underlying systemic or primary condition.
        """
    else:
        synthesis_user = f"""
        Original Case:
        {case_presentation}

        All Specialists' Reports:
        {all_reports_text}

        {assigned_roles_text}

        Other Specialists' Latest Stances:
        {peers_text}

        Probability Scoreboard (Independent Computation):
        {audit_scoreboard}

        Summarized Discussion So Far:
        {summary_context or 'None'}

        {round_instructions}

        Guidance:
        - Challenge consensus explicitly when scoreboard or evidence suggests an alternative.
        - Focus on initial symptoms and timeline consistency when imaging/labs are ambiguous.
        - Critique other specialists when they omit or downplay the initial symptom(s), timeline, or diagnostic criteria matching.
        """

    messages = [
        {"role": "system", "content": AUDIT_CF_SYSTEM_PROMPT},
        {"role": "user", "content": synthesis_user},
    ]
    thinking, response, choice = await llm.chat_with_logprobs(messages, max_new_tokens=MAX_NEW_TOKENS)

    final_dx_raw = extract_between_tags(response, "final_diagnosis") or (previous_round_stance or "Unknown")
    final_dx = clean_diagnosis_label(final_dx_raw)
    final_dx_prob = await _extract_final_dx_prob(
        response, choice, case_presentation, role=role, debate_summary=summary_context,
    )

    confidence_tag = extract_between_tags(response, "confidence")
    stance_conf = _confidence_band(confidence_tag, final_dx_prob if final_dx_prob is not None else previous_round_diagnosis_prob)

    return {
        "role": role,
        "round": round_idx,
        "response": response,
        "thinking": thinking,
        "stance": final_dx,
        "final_diagnosis": final_dx,
        "final_diagnosis_probability": final_dx_prob,
        "confidence": stance_conf,
        "previous_round_stance": previous_round_stance,
        "counterargument_question": extract_between_tags(response, "counterargument_question") or ("None" if round_idx == 0 else ""),
        "counterargument_answer": extract_between_tags(response, "counterargument_answer") or ("None" if round_idx < 2 else ""),
    }
