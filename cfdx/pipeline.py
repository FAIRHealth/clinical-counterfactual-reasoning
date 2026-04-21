from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional
from .agents import (
    assign_specialists_and_problems,
    audit_round_with_counterfactual,
    generate_specialist_report,
    generate_top_k_ddx,
    specialist_round_with_counterfactual,
)
from .config import MAX_ROUNDS, NUM_CF_CANDIDATES
from .prompts import format_operation_description
from .summarize import judge_final, summarize_final, summarize_round
from .utils import (
    consensus_from_stances,
    format_evidence_item,
    parse_questions_simplified,
)


def _build_probability_scoreboard(all_round_responses: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for response in all_round_responses:
        role = response.get("role", "")
        if role == "Independent Clinician":
            continue
        cf_data = response.get("top3_counterfactuals") or []
        if not cf_data:
            continue

        cf_blocks: List[str] = []
        for cf in cf_data:
            cand = cf["candidate"]
            scores = cf["metrics"]["scores"]
            baseline = cf["metrics"]["baseline"]
            counterfactual = cf["metrics"]["counterfactual"]
            baseline_label = baseline["diagnosis"]
            baseline_prob = baseline["probability_of_baseline_label"] or 0.0
            cf_label = counterfactual["diagnosis"]
            cf_prob = counterfactual["probability_of_baseline_label"] or 0.0

            change = (
                f" → {cf_label}"
                if cf_label.lower() != baseline_label.lower()
                else " (unchanged)"
            )

            tgt = cand.get("target_evidence_group")
            if isinstance(tgt, list):
                evidence = "\n  ".join(
                    format_evidence_item(it, include_feature_type=True, bullet=True)
                    for it in tgt
                )
                target_display = f"Target Evidence Group Modified:\n  {evidence}"
            else:
                target_display = (
                    f"Target Evidence Group Modified: "
                    f"{(cand.get('target_span', 'N/A') or '')[:100]}"
                )

            op_desc = format_operation_description(
                cand.get("operation", "N/A"),
                cand.get("hypothesis_tested", "N/A"),
                baseline_label,
            )
            cpg = scores["CPG"]
            if cpg > 0.2:
                cpg_interp = "PRIMARY evidence group (critical impact)"
            elif cpg > 0.1:
                cpg_interp = "SUPPORTING evidence group (moderate impact)"
            else:
                cpg_interp = "NON-DISCRIMINATING (minimal impact)"
            shift_note = " (diagnosis changed!)" if scores["label_shift_score"] > 0.3 else ""

            cf_blocks.append(
                f"Counterfactual #{cf['rank']} - Hypothesis Testing Results "
                f"(Combined Score: {scores['combined_score']:.3f})\n"
                f"- **Counterfactual Operation**: {op_desc}\n"
                f"- Original Diagnosis Used to Extract Target Evidence Group: {baseline_label}\n"
                f"- {target_display}\n"
                f"- Rationale: {cand.get('rationale', 'N/A')}\n"
                f"- **Diagnosis Impact**: {baseline_label} (P={baseline_prob:.3f})"
                f"{change} (P={cf_prob:.3f})\n"
                f"- **CPG (Probability Gap)**: {cpg:.3f}\n"
                f"  └─ Interpretation: {cpg_interp}\n"
                f"- Label Shift Semantic Score: {scores['label_shift_score']:.3f}{shift_note}\n"
            )

        round_idx = response.get("round", 0)
        blocks.append(
            f"=== {role} Counterfactual Hypothesis Testing Results (Round {round_idx}) ===\n"
            + "\n\n---\n\n".join(cf_blocks)
        )
    return "\n\n".join(blocks) if blocks else "No counterfactual results available"


async def run_counterfactual_role_based_debate(
    case_presentation: str,
    pmc_id: str,
    ground_truth: Optional[str] = None,
    max_rounds: int = MAX_ROUNDS,
    num_candidates: int = NUM_CF_CANDIDATES,
) -> Dict[str, Any]:
    debate_history: List[Dict[str, Any]] = [
        {"role": "Case Presentation", "content": case_presentation}
    ]

    # 1) Triage: choose specialists.
    print("Assigning specialists and identifying problems...")
    triage = await assign_specialists_and_problems(case_presentation)
    assigned = triage["assigned_specialists"]
    debate_history.append({
        "role": "Triage Agent",
        "content": json.dumps({k: v for k, v in triage.items() if k != "raw"}, indent=2),
    })

    # 2) Top-k differential diagnoses.
    print("Generating top 3 differential diagnoses as guidance...")
    top_ddx_list, case_summary, _ = await generate_top_k_ddx(case_presentation, k=3)
    debate_history.append({
        "role": "DDx Agent",
        "content": json.dumps({
            "case_summary": case_summary,
            "most_likely_diagnoses": top_ddx_list,
            "num_ddx": len(top_ddx_list),
        }, indent=2),
    })
    print(f"Top 3 DDx: {[d['diagnosis'] for d in top_ddx_list]}")

    # 3) Specialist domain reports.
    print("Generating domain-specific reports for assigned specialists...")
    report_tasks = [generate_specialist_report(case_presentation, spec["role"]) for spec in assigned]
    report_results = await asyncio.gather(*report_tasks)
    role_to_report: Dict[str, str] = {}
    for spec, (report, raw) in zip(assigned, report_results):
        role_to_report[spec["role"]] = report
        debate_history.append({"role": f"{spec['role']} Report", "content": raw})
    reports_text = "\n\n".join(f"- {role}:\n{rep}" for role, rep in role_to_report.items())
    assigned_role_names = [spec["role"] for spec in assigned]

    # 4) Round 0: specialists + audit clinician run CFs independently.
    print("Round 0: Specialists perform counterfactual edits and present stance...")
    round0_tasks = [
        specialist_round_with_counterfactual(
            role=role_name,
            case_presentation=case_presentation,
            all_reports_text=f"- {role_name}:\n{role_to_report.get(role_name, '')}",
            round_idx=0,
            summary_context="",
            top_ddx_guidance=top_ddx_list,
            peers_latest_stances={},
            incoming_questions_for_role=[],
            previous_round_stance=None,
            num_candidates=num_candidates,
            specialist_report=role_to_report.get(role_name, ""),
            assigned_specialist_roles=assigned_role_names,
        )
        for role_name in assigned_role_names
    ]
    round0_tasks.append(
        audit_round_with_counterfactual(
            case_presentation=case_presentation,
            all_reports_text=reports_text,
            round_idx=0,
            summary_context="",
            peers_latest_stances={},
            incoming_questions_for_role=[],
            previous_round_stance=None,
            num_candidates=max(1, num_candidates // 2),
            assigned_specialist_roles=assigned_role_names,
        )
    )
    round0_outputs = await asyncio.gather(*round0_tasks)

    specialist_names: List[str] = []
    role_to_current_stance: Dict[str, str] = {}
    role_to_previous_stance: Dict[str, str] = {}
    role_to_diagnosis_prob: Dict[str, float] = {}
    all_round_responses: List[Dict[str, Any]] = []

    for out in round0_outputs:
        role = out["role"]
        if role != "Independent Clinician":
            specialist_names.append(role)
        final_dx = out.get("final_diagnosis") or ""
        final_dx_prob = out.get("final_diagnosis_probability", 0.5)
        role_to_current_stance[role] = final_dx
        role_to_previous_stance[role] = final_dx
        role_to_diagnosis_prob[role] = final_dx_prob
        debate_history.append({"role": role, "round": 0, "content": out["response"]})
        debate_history.append({"role": f"{role} Thinking", "round": 0, "content": out["thinking"]})
        debate_history.append({
            "role": f"{role} Top3 CF Selection",
            "round": 0,
            "content": json.dumps(out.get("top3_counterfactuals", []), indent=2),
        })
        all_round_responses.append(out)

    for out in round0_outputs:
        print(f"{out['role']}: {out.get('final_diagnosis', '') or '-'} "
              f"(P={out.get('final_diagnosis_probability', 0.5):.3f})")

    print("Summarizing Round 0...")
    _, r0_summary = await summarize_round(all_round_responses, 0)
    debate_history.append({"role": "Round 0 Summary", "content": r0_summary})
    current_summary_context = r0_summary

    # 5) Subsequent rounds.
    reached_consensus, consensus_dx = consensus_from_stances(role_to_current_stance)
    consensus_round: Optional[int] = 0 if reached_consensus else None
    round_idx = 1

    specialist_only_stances = {
        k: v for k, v in role_to_current_stance.items() if k != "Independent Clinician"
    }
    all_specialists_agree = (
        len(set(specialist_only_stances.values())) == 1 if specialist_only_stances else False
    )

    while (not reached_consensus) and (round_idx <= max_rounds):
        print(f"Round {round_idx}/{max_rounds}...")
        peers_for_role: Dict[str, Dict[str, str]] = {
            role: {r: s for r, s in role_to_current_stance.items()
                   if r != role and r != "Independent Clinician"}
            for role in specialist_names
        }
        audit_peers = {
            r: s for r, s in role_to_current_stance.items() if r != "Independent Clinician"
        }

        incoming_questions: Dict[str, List[str]] = {r: [] for r in specialist_names}
        incoming_questions["Independent Clinician"] = []
        available_roles = specialist_names + ["Independent Clinician"]
        for prev in [r for r in all_round_responses if r.get("round") == round_idx - 1]:
            asker = prev.get("role", "")
            qblock = (prev.get("counterargument_question") or "").strip()
            routed = parse_questions_simplified(qblock, asker, available_roles)
            for target, questions in routed.items():
                if target in incoming_questions:
                    incoming_questions[target].extend(questions)

        tasks = [
            specialist_round_with_counterfactual(
                role=role,
                case_presentation=case_presentation,
                all_reports_text=reports_text,
                round_idx=round_idx,
                summary_context=current_summary_context,
                top_ddx_guidance=top_ddx_list,
                peers_latest_stances=peers_for_role.get(role, {}),
                incoming_questions_for_role=incoming_questions.get(role, []),
                previous_round_stance=role_to_previous_stance.get(role),
                previous_round_diagnosis_prob=role_to_diagnosis_prob.get(role, 0.5),
                num_candidates=num_candidates,
                groupthink_flag=all_specialists_agree,
                specialist_report=role_to_report.get(role, ""),
                assigned_specialist_roles=assigned_role_names,
            )
            for role in specialist_names
        ]
        tasks.append(
            audit_round_with_counterfactual(
                case_presentation=case_presentation,
                all_reports_text=reports_text,
                round_idx=round_idx,
                summary_context=current_summary_context,
                peers_latest_stances=audit_peers,
                incoming_questions_for_role=incoming_questions.get("Independent Clinician", []),
                previous_round_stance=role_to_previous_stance.get("Independent Clinician"),
                previous_round_diagnosis_prob=role_to_diagnosis_prob.get("Independent Clinician", 0.5),
                num_candidates=max(1, num_candidates // 2),
                assigned_specialist_roles=assigned_role_names,
            )
        )
        round_outputs = await asyncio.gather(*tasks)

        for out in round_outputs:
            role = out["role"]
            final_dx = out.get("final_diagnosis")
            final_dx_prob = out.get("final_diagnosis_probability", 0.5)
            if final_dx:
                role_to_current_stance[role] = final_dx
                role_to_previous_stance[role] = final_dx
                role_to_diagnosis_prob[role] = final_dx_prob
            debate_history.append({"role": role, "round": round_idx, "content": out["response"]})
            debate_history.append({
                "role": f"{role} Thinking", "round": round_idx, "content": out["thinking"],
            })
            debate_history.append({
                "role": f"{role} Top3 CF Selection",
                "round": round_idx,
                "content": json.dumps(out.get("top3_counterfactuals", []), indent=2),
            })
            all_round_responses.append(out)

        print(f"\n=== Round {round_idx} Final Diagnoses ===")
        for out in round_outputs:
            print(f"{out['role']}: {out.get('final_diagnosis', '') or '-'} "
                  f"(P={out.get('final_diagnosis_probability', 0.5):.3f})")

        print(f"Summarizing Round {round_idx}...")
        _, r_summary = await summarize_round(all_round_responses, round_idx)
        debate_history.append({"role": f"Round {round_idx} Summary", "content": r_summary})
        current_summary_context = r_summary

        reached_consensus, consensus_dx = consensus_from_stances(role_to_current_stance)
        specialist_only_stances = {
            k: v for k, v in role_to_current_stance.items() if k != "Independent Clinician"
        }
        all_specialists_agree = (
            len(set(specialist_only_stances.values())) == 1 if specialist_only_stances else False
        )

        if all_specialists_agree and round_idx < max_rounds and top_ddx_list:
            agreed = list(specialist_only_stances.values())[0]
            if any(d.get("diagnosis", "").lower() != agreed.lower() for d in top_ddx_list):
                debate_history.append({
                    "role": f"Diversity Warning Round {round_idx}",
                    "content": (
                        f"GROUPTHINK WARNING (Round {round_idx}): All specialists converged on "
                        f"\"{agreed}\" while alternative DDx remain: "
                        f"{[d['diagnosis'] for d in top_ddx_list]}."
                    ),
                })

        if reached_consensus:
            consensus_round = round_idx
            print(f"Consensus reached at Round {round_idx}: {consensus_dx}")
            break
        round_idx += 1

    # 6) Probability scoreboard for the judge.
    probability_scoreboard = _build_probability_scoreboard(all_round_responses)
    print("Probability scoreboard", probability_scoreboard)

    # 7) Final summary + judge.
    print("Producing final summary...")
    _, final_summary = await summarize_final(all_round_responses, consensus_round=consensus_round)
    debate_history.append({"role": "Final Summary", "content": final_summary})

    if reached_consensus:
        print(f"Consensus reached: {consensus_dx}. Using consensus directly without judge override.")
    else:
        print("No consensus; invoking judge to select final diagnosis...")

    judge_thinking, judge_response = await judge_final(
        case_presentation,
        final_summary,
        reached_consensus,
        consensus_dx,
        probability_scoreboard,
        top_ddx_list,
        all_round_responses,
    )
    debate_history.append({"role": "Judge Decision", "content": judge_response})
    if judge_thinking:
        debate_history.append({"role": "Judge Thinking", "content": judge_thinking})

    return {
        "pmc_id": pmc_id,
        "ground_truth": ground_truth,
        "assigned_specialists": assigned,
        "top_ddx_guidance": top_ddx_list,
        "final_role_to_stance": role_to_current_stance,
        "consensus": {"had_consensus": reached_consensus, "diagnosis": consensus_dx},
        "history": debate_history,
    }
