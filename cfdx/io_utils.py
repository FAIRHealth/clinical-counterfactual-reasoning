from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd


CASE_ID_CANDIDATES = ("pmc_id", "pmcid", "key", "id")
CASE_TEXT_CANDIDATES = ("case_presentation", "case_prompt", "full_information", "case")
GROUND_TRUTH_CANDIDATES = ("final_diagnosis", "discharge_diagnosis", "ground_truth")


def _first_present(row: pd.Series, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    return None


def load_cases(
    input_path: str,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    sample: Optional[int] = None,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(input_path)

    if sample is not None and sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=random_state).reset_index(drop=True)

    if offset:
        df = df.iloc[offset:]
    if limit is not None and limit > 0:
        df = df.iloc[:limit]

    cases: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        pmc_id = _first_present(row, CASE_ID_CANDIDATES) or f"row_{idx}"
        case_text = _first_present(row, CASE_TEXT_CANDIDATES) or ""
        ground_truth = _first_present(row, GROUND_TRUTH_CANDIDATES) or ""
        if not case_text:
            continue
        cases.append({
            "pmc_id": pmc_id,
            "case_presentation": case_text,
            "ground_truth": ground_truth,
        })
    return cases


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
