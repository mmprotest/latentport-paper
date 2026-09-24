"""Conditionally execute frozen 16K generalization after a NEAR_NATIVE 4K verdict."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import json_safe_score, position_ids
from experiments.latentport.e001_handoff.state.cache_state import capture_cache, state_checksum
from experiments.latentport.e001_handoff.translators.runtime_translation import (
    FrozenTranslators,
    translate_full_state,
)
from experiments.latentport.e002_coupler.analysis.metrics import native_context_recovery
from experiments.latentport.e002_coupler.analysis.statistics import bootstrap_mean_ci
from experiments.latentport.e002_coupler.correction.serialization import load_correction
from experiments.latentport.e002_coupler.factorial.composer import compose_condition
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT, load_preregistration
from experiments.latentport.e002_coupler.runtime.install import install_for_inference
from experiments.latentport.e002_coupler.runtime.lock_guard import sha256_file, verify_frozen_manifest
from experiments.latentport.e002_coupler.runtime.scoring import score_primary
from experiments.latentport.e002_coupler.runtime.state_io import load_state, save_state_once, write_json_once


ROOT = ATTEMPT_ROOT / "long_context"
START = ROOT / "LONG_RUN_STARTED.json"
RAW = ROOT / "raw_evidence.jsonl"
RESULT = ROOT / "long_result.json"
CONDITIONS = ("NATIVE_9B", "SOURCE_4B", "EMPTY_9B", "BASE_STATE", "JOINT_CORRECTED", "JOINT_SHUFFLED")


def _rows() -> list[dict[str, Any]]:
    verify_frozen_manifest()
    path = E002_ROOT / "data" / "long" / "manifest.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 16:
        raise RuntimeError("frozen LONG manifest must contain 16 documents")
    return rows


def _tokens(row: dict[str, Any]) -> tuple[list[int], int, list[int]]:
    ids = row["token_ids"]
    prefix, bridge, continuation = ids[:16384], int(ids[16384]), ids[16385:16449]
    if len(prefix) != 16384 or len(continuation) != 64:
        raise RuntimeError("LONG token accounting failure")
    return prefix, bridge, [int(value) for value in continuation]


def _append(value: object) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with RAW.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _score(model, cache, bridge: int, continuation: list[int]) -> dict[str, Any]:
    return score_primary(
        model,
        cache,
        prefix_length=16384,
        bridge_token_id=bridge,
        continuation_token_ids=continuation,
        retain_logits=False,
    )


def _prefill(model, config, prefix: list[int]) -> Any:
    device = next(model.parameters()).device
    cache = DynamicCache(config=config)
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([prefix], dtype=torch.long, device=device),
            position_ids=position_ids(0, 16384, device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    return cache


def _prepare_states(rows: list[dict[str, Any]], frozen: dict[str, Any]) -> list[dict[str, Any]]:
    source_dir = ROOT / "states" / "source"
    translated_dir = ROOT / "states" / "translated"
    corrected_dir = ROOT / "states" / "corrected"
    source_model = load_language_model(SOURCE)
    source_config = load_text_config(SOURCE)
    prepared = []
    for index, row in enumerate(rows):
        prefix, bridge, continuation = _tokens(row)
        cache = _prefill(source_model, source_config, prefix)
        state = capture_cache(cache, source_config, device="cpu")
        source_path = source_dir / f"source_{index:03d}.pt"
        save_state_once(source_path, state)
        score = _score(source_model, cache, bridge, continuation)
        prepared.append(
            {
                "document_id": row["document_id"],
                "source_file": str(source_path.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
                "source_checksum": state_checksum(state),
                "source_score": json_safe_score(score),
            }
        )
        print(f"E002 LONG source {index + 1}/16", flush=True)
        del cache, state, score
    release_model(source_model)
    translators = FrozenTranslators()
    coupler = load_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
        device="cuda:0",
    )
    for index, record in enumerate(prepared):
        source = load_state(ATTEMPT_ROOT / record["source_file"])
        translated, _ = translate_full_state(source, translators)
        translated_path = translated_dir / f"translated_{index:03d}.pt"
        save_state_once(translated_path, translated)
        base = compose_condition(source, translated, frozen["base_state"])
        corrected = coupler.apply_to_state(base, detach_to_cpu=True).state
        corrected_path = corrected_dir / f"corrected_{index:03d}.pt"
        save_state_once(corrected_path, corrected)
        record.update(
            {
                "translated_file": str(translated_path.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
                "corrected_file": str(corrected_path.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
                "base_checksum": state_checksum(base),
                "corrected_checksum": state_checksum(corrected),
            }
        )
        print(f"E002 LONG translation/correction {index + 1}/16", flush=True)
        del source, translated, base, corrected
        torch.cuda.empty_cache()
    del translators, coupler
    release_model(None)
    return prepared


def _target(rows: list[dict[str, Any]], prepared: list[dict[str, Any]], frozen: dict[str, Any]) -> None:
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    order = sorted(range(16), key=lambda item: rows[item]["selection_sha256"])
    donor = {order[position]: order[(position + 1) % 16] for position in range(16)}
    if any(left == right for left, right in donor.items()):
        raise RuntimeError("LONG shuffle rotation contains a fixed point")
    for index, row in enumerate(rows):
        prefix, bridge, continuation = _tokens(row)
        source = load_state(ATTEMPT_ROOT / prepared[index]["source_file"])
        translated = load_state(ATTEMPT_ROOT / prepared[index]["translated_file"])
        base = compose_condition(source, translated, frozen["base_state"])
        corrected = load_state(ATTEMPT_ROOT / prepared[index]["corrected_file"])
        shuffled = load_state(ATTEMPT_ROOT / prepared[donor[index]]["corrected_file"])
        native_cache = _prefill(model, config, prefix)
        conditions = {"NATIVE_9B": json_safe_score(_score(model, native_cache, bridge, continuation))}
        conditions["SOURCE_4B"] = prepared[index]["source_score"]
        for name, state in (
            ("EMPTY_9B", None),
            ("BASE_STATE", base),
            ("JOINT_CORRECTED", corrected),
            ("JOINT_SHUFFLED", shuffled),
        ):
            cache = DynamicCache(config=config) if state is None else install_for_inference(state, config)
            conditions[name] = json_safe_score(_score(model, cache, bridge, continuation))
            del cache
        _append(
            {
                "document_index": index,
                "document_id": row["document_id"],
                "prefix_tokens": 16384,
                "bridge_token_id": bridge,
                "continuation_token_ids": continuation,
                "conditions": conditions,
                "shuffle_donor_index": donor[index],
                "shuffle_fixed_point": False,
                "correction_hash": frozen["correction_hash"],
                "no_retraining_or_recalibration": True,
            }
        )
        print(f"E002 LONG target {index + 1}/16", flush=True)
        del source, translated, base, corrected, shuffled, native_cache
        torch.cuda.empty_cache()
    release_model(model)


def _aggregate() -> dict[str, Any]:
    with RAW.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    nll = {
        condition: np.asarray([row["conditions"][condition]["nll"] for row in rows], dtype=np.float64)
        for condition in CONDITIONS
    }
    means = {condition: float(values.mean()) for condition, values in nll.items()}
    delta = means["JOINT_CORRECTED"] - means["NATIVE_9B"]
    ncr = native_context_recovery(means["EMPTY_9B"], means["JOINT_CORRECTED"], means["NATIVE_9B"])
    base_ci = list(bootstrap_mean_ci(nll["BASE_STATE"] - nll["JOINT_CORRECTED"]))
    source_ci = list(bootstrap_mean_ci(nll["JOINT_CORRECTED"] - nll["SOURCE_4B"]))
    shuffled_ci = list(bootstrap_mean_ci(nll["JOINT_CORRECTED"] - nll["JOINT_SHUFFLED"]))
    passed = delta <= 0.10 and ncr >= 0.90 and base_ci[0] > 0 and source_ci[1] < 0 and shuffled_ci[1] < 0
    return {
        "documents": 16,
        "condition_nll": means,
        "long_delta_nll": delta,
        "long_ncr": ncr,
        "base_minus_corrected_bootstrap_ci": base_ci,
        "corrected_minus_source_bootstrap_ci": source_ci,
        "corrected_minus_shuffled_bootstrap_ci": shuffled_ci,
        "long_source_advantage": source_ci[1] < 0,
        "long_pass": passed,
        "raw_evidence_sha256": sha256_file(RAW),
    }


def main() -> None:
    verdict = json.loads(
        (ATTEMPT_ROOT / "verdict" / "4k_verdict.json").read_text(encoding="utf-8")
    )
    if verdict["canonical_4k_verdict"] != "NEAR_NATIVE_HANDOFF":
        raise PermissionError("LONG is locked unless the frozen 4K verdict is NEAR_NATIVE_HANDOFF")
    if START.exists() or RAW.exists() or RESULT.exists():
        raise FileExistsError("LONG execution has already started")
    frozen = verify_frozen_manifest()
    write_json_once(
        START,
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_4k_verdict": verdict["canonical_4k_verdict"],
            "correction_hash": frozen["correction_hash"],
            "retraining": False,
            "recalibration": False,
        },
    )
    seed_everything(load_preregistration()["seeds"]["global"])
    rows = _rows()
    prepared = _prepare_states(rows, frozen)
    _target(rows, prepared, frozen)
    result = _aggregate()
    write_json_once(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
