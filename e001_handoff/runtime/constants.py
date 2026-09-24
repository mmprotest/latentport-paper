"""Frozen paths and model identities for LatentPort E001."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


E001_ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_ROOT = E001_ROOT / "artifacts" / "attempt_001"
IMPLEMENTATION_ROOT = ATTEMPT_ROOT / "implementation"
STATE_VALIDATION_ROOT = ATTEMPT_ROOT / "state_validation"


@dataclass(frozen=True)
class ModelSpec:
    role: str
    repository: str
    revision: str
    snapshot: Path
    expected_parameter_count: int


SOURCE = ModelSpec(
    role="source",
    repository="Qwen/Qwen3.5-4B-Base",
    revision="1001bb4d826a52d1f399e183466143f4da7b741b",
    snapshot=Path(
        r"C:\Users\Simon\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B-Base"
        r"\snapshots\1001bb4d826a52d1f399e183466143f4da7b741b"
    ),
    expected_parameter_count=4_205_751_296,
)

TARGET = ModelSpec(
    role="target",
    repository="Qwen/Qwen3.5-9B-Base",
    revision="68c46c4b3498877f3ef123c856ecfde50c39f404",
    snapshot=Path(
        r"C:\Users\Simon\.cache\huggingface\hub\models--Qwen--Qwen3.5-9B-Base"
        r"\snapshots\68c46c4b3498877f3ef123c856ecfde50c39f404"
    ),
    expected_parameter_count=8_953_803_264,
)

MODEL_SPECS = {"source": SOURCE, "target": TARGET, "4b": SOURCE, "9b": TARGET}


def load_preregistration() -> dict:
    with (E001_ROOT / "PREREGISTRATION.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)

