from __future__ import annotations
import os


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Counterfactual preservation/quality gates
PRESERVATION_THRESHOLD: float = _get_env_float("PRESERVATION_THRESHOLD", 0.85)
EDIT_SIM_THRESHOLD: float = _get_env_float("EDIT_SIM_THRESHOLD", 0.80)
POOR_PRESERVATION_GATE: float = _get_env_float("POOR_PRESERVATION_GATE", 0.5)

# Default label-probability cache capacity
LABEL_PROB_CACHE_SIZE: int = max(1, _get_env_int("LABEL_PROB_CACHE_SIZE", 2048))

# Debate hyper-parameters (also accepted via CLI)
MAX_ROUNDS: int = _get_env_int("MAX_ROUNDS", 3)
NUM_CF_CANDIDATES: int = _get_env_int("NUM_CF_CANDIDATES", 2)

# Default token budget for generative calls (backend may override)
MAX_NEW_TOKENS: int = _get_env_int("MAX_NEW_TOKENS", 32768)
