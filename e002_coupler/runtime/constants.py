"""Frozen paths, conditions, and helpers for LatentPort E002."""

from __future__ import annotations

import json
from pathlib import Path


E002_ROOT = Path(__file__).resolve().parents[1]
E001_ROOT = E002_ROOT.parent / "e001_handoff"
ATTEMPTS_ROOT = E002_ROOT / "artifacts"
ATTEMPT_NAME = "attempt_001"
ATTEMPT_ROOT = ATTEMPTS_ROOT / ATTEMPT_NAME

FACTORIAL_CONDITIONS = ("DDD", "DDT", "DTD", "DTT", "TDD", "TDT", "TTD", "TTT")
PRIMARY_LOCKED_CONDITIONS = (
    "NATIVE_9B",
    "SOURCE_4B",
    "EMPTY_9B",
    *FACTORIAL_CONDITIONS,
    "BASE_STATE",
    "JOINT_CORRECTED",
    "JOINT_SHUFFLED",
)
STATE_REPAIR_CHECKPOINTS = (1, 4, 16, 64, 256)
TRAINABLE_PARAMETER_CAP = 2_000_000


def load_preregistration() -> dict:
    return json.loads((E002_ROOT / "PREREGISTRATION.json").read_text(encoding="utf-8"))


def ensure_attempt_layout() -> None:
    for relative in (
        "implementation",
        "factorial",
        "fit",
        "validation",
        "locked",
        "diagnostics",
        "long_context",
        "timing",
        "figures",
        "verdict",
    ):
        (ATTEMPT_ROOT / relative).mkdir(parents=True, exist_ok=True)

