"""Freeze the canonical 4K statistics before any diagnostic or LONG execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.latentport.e001_handoff.analysis.integrity import (
    translation_total_ms,
    validate_raw_evidence_file,
)
from experiments.latentport.e001_handoff.analysis.verdict import (
    NEXT_HYPOTHESES,
    choose_verdict,
    compute_locked_statistics,
)
from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT, SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.lock_guard import FROZEN_MANIFEST, sha256_file


RAW_EVIDENCE = ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl"
OUTPUT = ATTEMPT_ROOT / "verdict" / "CANONICAL_4K_RESULT.json"
DERIVED = ATTEMPT_ROOT / "locked" / "derived_statistics.json"


def load_rows() -> list[dict[str, Any]]:
    with RAW_EVIDENCE.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _translator_inventory() -> tuple[int, int]:
    parameters = 0
    stored_bytes = 0
    for name in ("kv_translator", "gdn_convolution_translator", "gdn_recurrent_translator"):
        record = json.loads(
            (E001_ROOT / "translators" / "frozen" / f"{name}.json").read_text(encoding="utf-8")
        )
        parameters += int(record["parameter_count_weights"])
        stored_bytes += int(record["tensor_bytes"])
    return parameters, stored_bytes


def build_result(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    integrity = validate_raw_evidence_file(RAW_EVIDENCE)
    if not integrity["pass"]:
        raise RuntimeError(f"Raw LOCKED evidence is incomplete: {integrity['errors'][:10]}")
    prereg = json.loads((E001_ROOT / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    stats = compute_locked_statistics(rows, bootstrap_seed=prereg["seeds"]["bootstrap"])
    verdict = choose_verdict(stats, long_run=False, long_pass=False)
    translator_parameters, translator_bytes = _translator_inventory()
    conditions = stats["condition_summary"]
    translation_totals = [translation_total_ms(row["timing"]) for row in rows]
    full = conditions["FULL_TRANSLATED"]
    result = {
        "experiment_id": "LATENTPORT_E001_HANDOFF",
        "result_stage": "CANONICAL_4K_FROZEN_BEFORE_DIAGNOSTICS",
        "e001_status": "VALID_CANONICAL_4K_COMPLETE",
        "canonical_4k_verdict": verdict,
        "canonical_verdict": verdict,
        "source_model": SOURCE.repository,
        "source_revision": SOURCE.revision,
        "target_model": TARGET.repository,
        "target_revision": TARGET.revision,
        "fit_documents": 128,
        "validation_documents": 32,
        "locked_documents": 64,
        "long_documents_if_run": 0,
        "same_model_restore_4b_pass": True,
        "same_model_restore_9b_pass": True,
        "native_9b_nll": conditions["NATIVE_9B"]["mean_nll"],
        "source_4b_nll": conditions["SOURCE_4B"]["mean_nll"],
        "empty_9b_nll": conditions["EMPTY_9B"]["mean_nll"],
        "kv_only_nll": conditions["KV_ONLY"]["mean_nll"],
        "kv_gdn_direct_nll": conditions["KV_GDN_DIRECT"]["mean_nll"],
        "full_translated_nll": full["mean_nll"],
        "full_shuffled_nll": conditions["FULL_SHUFFLED"]["mean_nll"],
        "kv_only_delta_nll": conditions["KV_ONLY"]["mean_delta_nll"],
        "full_translated_delta_nll": full["mean_delta_nll"],
        "full_vs_kv_delta": stats["full_vs_kv_mean_nll_improvement"],
        "full_vs_kv_improvement_fraction": stats["full_vs_kv_mean_improvement_fraction"],
        "full_vs_kv_median_improvement_fraction": stats[
            "full_vs_kv_median_improvement_fraction"
        ],
        "full_vs_kv_bootstrap_ci": stats["full_vs_kv_bootstrap_ci"],
        "tqr": stats["aggregate_nll_tqr"],
        "mean_document_tqr": stats["mean_document_tqr"],
        "tqr_exclusion_count": stats["tqr_exclusion_count"],
        "full_vs_shuffled_delta": stats["full_vs_shuffled_mean_nll_improvement"],
        "full_vs_shuffled_bootstrap_ci": stats["full_vs_shuffled_bootstrap_ci"],
        "mean_js_to_native": full["mean_js_to_native"],
        "top1_agreement_to_native": full["mean_top1_agreement_to_native"],
        "long_test_run": False,
        "long_delta_nll": None,
        "long_tqr": None,
        "long_pass": False,
        "native_9b_prefill_ms": _mean(rows, ("timing", "native_9b_prefill_wall_ms")),
        "source_4b_prefill_ms": _mean(rows, ("timing", "source_4b_prefill_wall_ms")),
        "translation_ms": float(np.mean(translation_totals)),
        "state_install_ms": _mean(rows, ("timing", "state_install_ms", "FULL_TRANSLATED")),
        "bridge_ms": _mean(rows, ("timing", "bridge_ms", "FULL_TRANSLATED")),
        "translator_parameter_count": translator_parameters,
        "translator_bytes": translator_bytes,
        "source_state_bytes": int(np.mean([row["state_bytes"]["source"] for row in rows])),
        "target_state_bytes": int(
            np.mean([row["state_bytes"]["target_native_prefix"] for row in rows])
        ),
        "translator_hash": rows[0]["translator_hash"],
        "frozen_manifest_hash": sha256_file(FROZEN_MANIFEST),
        "raw_evidence_sha256": sha256_file(RAW_EVIDENCE),
        "oracle_diagnostics_unlocked": stats["full_vs_kv_bootstrap_ci"][0] <= 0,
        "long_test_eligible": verdict == "FULL_STATE_HANDOFF",
        "next_hypothesis": NEXT_HYPOTHESES[verdict],
    }
    return result, stats


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    rows = load_rows()
    result, stats = build_result(rows)
    _write_once(DERIVED, stats)
    _write_once(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
