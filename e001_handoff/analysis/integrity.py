"""Raw-evidence and timing integrity checks for canonical E001 records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CANONICAL_CONDITIONS = (
    "NATIVE_9B",
    "SOURCE_4B",
    "EMPTY_9B",
    "KV_ONLY",
    "KV_GDN_DIRECT",
    "FULL_TRANSLATED",
    "FULL_SHUFFLED",
)


def translation_total_ms(timing: dict[str, Any]) -> float:
    compute = sum(
        float(timing[key])
        for key in (
            "kv_translation_ms",
            "gdn_recurrent_translation_ms",
            "gdn_convolution_translation_ms",
        )
    )
    recorded = float(timing["translation_compute_ms"])
    if not math.isclose(compute, recorded, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Translation compute timing does not equal its component sum")
    return recorded + float(timing["host_device_copy_ms"])


def validate_raw_evidence_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "document_id",
        "corpus",
        "split",
        "prefix_token_count",
        "bridge_token_id",
        "continuation_token_ids",
        "conditions",
        "state_checksums",
        "state_bytes",
        "timing",
        "state_convergence",
        "token_accounting",
        "shuffle",
        "translator_hash",
    ):
        if key not in record:
            errors.append(f"missing root field: {key}")
    if errors:
        return errors
    if record["split"] != "LOCKED" or record["prefix_token_count"] != 4096:
        errors.append("wrong canonical split or prefix length")
    if len(record["continuation_token_ids"]) != 64:
        errors.append("continuation token count is not 64")
    if set(record["conditions"]) != set(CANONICAL_CONDITIONS):
        errors.append("canonical condition inventory mismatch")
    for name in CANONICAL_CONDITIONS:
        condition = record["conditions"].get(name, {})
        if "nll" not in condition or len(condition.get("per_token_nll", [])) != 64:
            errors.append(f"{name} lacks complete NLL evidence")
    for name in ("EMPTY_9B", "KV_ONLY", "KV_GDN_DIRECT", "FULL_TRANSLATED", "FULL_SHUFFLED"):
        condition = record["conditions"].get(name, {})
        if condition.get("historical_prefix_tokens_processed_in_scoring_branch") != 0:
            errors.append(f"{name} replayed historical target tokens")
        if condition.get("continuation_schedule") != [1, 4, 16, 64]:
            errors.append(f"{name} used a different continuation schedule")
    if record["shuffle"].get("fixed_point") is not False:
        errors.append("shuffled-state control has a fixed point")
    if set(record["state_convergence"]) != {"1", "4", "16", "64"}:
        errors.append("state-convergence checkpoints are incomplete")
    try:
        translation_total_ms(record["timing"])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid timing accounting: {error}")
    return errors


def validate_raw_evidence_file(path: Path, *, expected_documents: int = 64) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    errors = []
    if len(rows) != expected_documents:
        errors.append(f"document count {len(rows)} != {expected_documents}")
    for index, row in enumerate(rows):
        errors.extend(f"document {index}: {message}" for message in validate_raw_evidence_record(row))
    document_ids = [row.get("document_id") for row in rows]
    if len(document_ids) != len(set(document_ids)):
        errors.append("duplicate document IDs in raw evidence")
    return {"pass": not errors, "documents": len(rows), "errors": errors}
