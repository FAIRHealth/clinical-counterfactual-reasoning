from __future__ import annotations
import difflib
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False
    SentenceTransformer = None 
    import numpy as np 


_SBERT_MODEL: Optional["SentenceTransformer"] = None


def _ensure_sbert_model() -> Optional["SentenceTransformer"]:
    global _SBERT_MODEL
    if not SBERT_AVAILABLE:
        return None
    if _SBERT_MODEL is None:
        try:
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("Loaded SBERT model (all-MiniLM-L6-v2).")
        except Exception:
            _SBERT_MODEL = None
    return _SBERT_MODEL


def semantic_similarity_sbert(text_a: str, text_b: str) -> float:
    model = _ensure_sbert_model()
    if model is None:
        return 0.0
    try:
        emb = model.encode([text_a, text_b])
        a, b = emb[0], emb[1]
        num = float((a * b).sum())
        den = float((a ** 2).sum()) ** 0.5 * float((b ** 2).sum()) ** 0.5
        if den == 0:
            return 0.0
        cos = num / den
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))
    except Exception:
        return 0.0


def sequence_similarity_ratio(text_a: str, text_b: str) -> float:
    try:
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()
    except Exception:
        return 0.0


def compute_sip_score(original: str, edited: str) -> Dict[str, float]:
    sem = semantic_similarity_sbert(original, edited)
    edit = sequence_similarity_ratio(original, edited)
    return {
        "semantic_similarity": sem,
        "edit_similarity": edit,
        "preservation_score": 0.7 * sem + 0.3 * edit,
    }


def compute_label_shift_score(baseline_label: str, cf_label: str) -> float:
    if not baseline_label or not cf_label:
        return 0.0
    if baseline_label.strip().lower() == cf_label.strip().lower():
        return 0.0
    sim = semantic_similarity_sbert(baseline_label, cf_label)
    return max(0.0, min(1.0, 1.0 - sim))


@dataclass
class CounterfactualMetrics:
    baseline_label: str
    baseline_label_prob: float
    cf_label: str
    cf_baseline_label_prob: float
    cpg_text: float  
    cfr_text: float 
    label_shift_score: float
    semantic_similarity: float
    edit_similarity: float
    preservation_score: float
    combined_score: float


def combine_scores(cpg: float, label_shift: float, preservation: float, edit_sim: float) -> float:
    MIN_PRESERVATION = 0.50
    MIN_EDIT_SIM = 0.50
    if preservation < MIN_PRESERVATION or edit_sim < MIN_EDIT_SIM:
        return 0.0

    MIN_CPG = 0.01
    MIN_LABEL_SHIFT = 0.1
    if cpg < MIN_CPG and label_shift < MIN_LABEL_SHIFT:
        return 0.0

    signal = max(cpg, label_shift * 0.75)
    combined = 0.8 * signal + 0.2 * preservation
    return max(0.0, min(1.0, combined))
