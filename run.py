from __future__ import annotations
import argparse
import asyncio
import os
import time
import traceback
from typing import Any, Dict, List
from cfdx import build_backend, run_counterfactual_role_based_debate
from cfdx.config import MAX_ROUNDS, NUM_CF_CANDIDATES
from cfdx.io_utils import load_cases, save_results
from cfdx.llm import set_backend, wait_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", "-i", required=False,
        default=os.environ.get("INPUT_CSV"),
        help="Path to a CSV (or JSON) with columns like pmc_id, case_presentation, final_diagnosis.",
    )
    parser.add_argument(
        "--output", "-o", required=False,
        default=os.environ.get("OUTPUT_JSON", "results.json"),
        help="Where to write the per-case JSON results.",
    )
    parser.add_argument(
        "--backend", choices=["vllm", "openai"],
        default=os.environ.get("CFDX_BACKEND", "vllm"),
        help="LLM backend to use.",
    )
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                        help="Maximum discussion rounds after Round 0.")
    parser.add_argument("--num-candidates", type=int, default=NUM_CF_CANDIDATES,
                        help="Counterfactual candidates to generate per DDx hypothesis.")
    parser.add_argument("--limit", type=int, default=None, help="Take the first N rows after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N rows.")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N rows.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for --sample.")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts per case.")
    parser.add_argument("--wait-ready-timeout", type=int, default=180,
                        help="Seconds to wait for the backend server to become ready.")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not args.input:
        raise SystemExit("--input (or INPUT_CSV env var) is required.")

    set_backend(build_backend(kind=args.backend))
    try:
        await wait_ready(timeout_seconds=args.wait_ready_timeout)
    except Exception as exc: 
        print(f"[WARN] backend readiness check failed: {exc}")

    cases = load_cases(
        args.input,
        limit=args.limit,
        offset=args.offset,
        sample=args.sample,
        random_state=args.random_state,
    )
    print(f"Loaded {len(cases)} case(s) from {args.input}")

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()
    print(f"Starting discussion ({args.backend}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for i, case in enumerate(cases):
        pmc_id = case["pmc_id"]
        case_text = case["case_presentation"]
        ground_truth = case.get("ground_truth", "")

        t_case = time.time()
        print(f"\nCase {i+1}/{len(cases)} - ID: {pmc_id}")

        case_result = None
        last_error = None
        for attempt in range(max(1, args.retries)):
            if attempt > 0:
                print(f"  Retry attempt {attempt}/{args.retries-1} for case {pmc_id}")
            try:
                result = await run_counterfactual_role_based_debate(
                    case_presentation=case_text,
                    pmc_id=pmc_id,
                    ground_truth=ground_truth,
                    max_rounds=args.max_rounds,
                    num_candidates=args.num_candidates,
                )
                if isinstance(result, dict):
                    case_result = {
                        "pmc_id": pmc_id,
                        "ground_truth_diagnosis": ground_truth,
                        "discussion_result": result,
                    }
                    print(f"Completed in {(time.time() - t_case)/60:.2f} minutes")
                    print(f"Consensus: {result.get('consensus', 'N/A')}")
                    break
                last_error = "Invalid result structure"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                print(f"  Attempt {attempt+1} error: {last_error}")
                if attempt == args.retries - 1:
                    print(traceback.format_exc())

        if case_result is None:
            case_result = {
                "pmc_id": pmc_id,
                "ground_truth_diagnosis": ground_truth,
                "error": last_error,
                "retry_exhausted": True,
            }
        all_results.append(case_result)
        save_results(all_results, args.output)
        print("-" * 50)

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Done in {elapsed/60:.2f} min ({elapsed/3600:.2f} h). "
          f"Average {elapsed/max(1, len(cases)):.1f}s/case.")
    print(f"Results saved to: {args.output}")


def main() -> None:
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
