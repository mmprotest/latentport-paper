"""Execute the single authorized canonical 4K LOCKED run."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, SOURCE, TARGET, load_preregistration
from experiments.latentport.e001_handoff.runtime.lock_guard import (
    begin_locked_run,
    load_locked_rows,
    sha256_file,
    verify_frozen_manifest,
)
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import (
    compare_full_distributions,
    json_safe_score,
    position_ids,
    score_from_cache_segmented,
)
from experiments.latentport.e001_handoff.state.cache_state import (
    capture_cache,
    component_nbytes,
    compose_component_state,
    install_components,
    state_checksum,
    state_nbytes,
)
from experiments.latentport.e001_handoff.state.convergence import compare_convergence_states
from experiments.latentport.e001_handoff.translators.runtime_translation import (
    FrozenTranslators,
    translate_full_state,
)


LOCKED_ROOT = ATTEMPT_ROOT / "locked"
SOURCE_STATE_ROOT = LOCKED_ROOT / "source_states"
TRANSLATED_STATE_ROOT = LOCKED_ROOT / "translated_states"
NATIVE_STATE_ROOT = LOCKED_ROOT / "native_states"
SOURCE_RAW = LOCKED_ROOT / "source_pass.jsonl"
TRANSLATION_RAW = LOCKED_ROOT / "translation_pass.jsonl"
RAW_EVIDENCE = LOCKED_ROOT / "raw_evidence.jsonl"
EXECUTION_RECORD = LOCKED_ROOT / "locked_execution.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _token_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in token_ids).encode("ascii")).hexdigest()


def _save_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite LOCKED state evidence: {path}")
    start = time.perf_counter()
    torch.save(state, path)
    serialization_ms = (time.perf_counter() - start) * 1000.0
    return {
        "path": str(path.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
        "serialized_bytes": path.stat().st_size,
        "serialized_sha256": sha256_file(path),
        "serialization_ms": serialization_ms,
    }


def _load_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _timed_prefill(
    model, input_ids: torch.Tensor, cache, prefix_length: int
) -> tuple[float, float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    start_event.record()
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            position_ids=position_ids(0, prefix_length, input_ids.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    end_event.record()
    torch.cuda.synchronize()
    return (
        (time.perf_counter() - wall_start) * 1000.0,
        float(start_event.elapsed_time(end_event)),
        (time.process_time() - cpu_start) * 1000.0,
    )


def _capture_timed(cache, config) -> tuple[dict[str, Any], float, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    cpu_start = time.process_time()
    captured = capture_cache(cache, config, device="cpu")
    torch.cuda.synchronize()
    return (
        captured,
        (time.perf_counter() - start) * 1000.0,
        (time.process_time() - cpu_start) * 1000.0,
    )


def _validate_finite_state(state: dict[str, Any]) -> bool:
    for layer in state["layers"]:
        tensors = (
            (layer["keys"], layer["values"])
            if layer["layer_type"] == "full_attention"
            else (layer["conv_states"][0], layer["recurrent_states"][0])
        )
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            return False
    return True


def _source_pass(rows: list[dict[str, Any]], prereg: dict) -> list[dict[str, Any]]:
    if SOURCE_RAW.exists():
        raise FileExistsError("Source LOCKED evidence already exists")
    SOURCE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    config = load_text_config(SOURCE)
    model = load_language_model(SOURCE)
    device = next(model.parameters()).device
    results = []
    for index, row in enumerate(rows):
        torch.cuda.reset_peak_memory_stats()
        token_ids = row["token_ids"]
        prefix = token_ids[:4096]
        bridge = token_ids[4096]
        continuation = token_ids[4097:4161]
        inputs = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        cache = DynamicCache(config=config)
        prefill_wall_ms, prefill_gpu_ms, prefill_cpu_ms = _timed_prefill(model, inputs, cache, 4096)
        captured, extraction_ms, extraction_cpu_ms = _capture_timed(cache, config)
        checksum = state_checksum(captured)
        state_path = SOURCE_STATE_ROOT / f"source_{index:03d}.pt"
        serialization = _save_state(state_path, captured)
        score = score_from_cache_segmented(
            model,
            cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
        )
        score["historical_prefix_tokens_processed_in_scoring_branch"] = 4096
        record = {
            "document_index": index,
            "document_id": row["document_id"],
            "corpus": row["corpus"],
            "selection_sha256": row["selection_sha256"],
            "prefix_token_count": 4096,
            "bridge_token_id": bridge,
            "continuation_token_ids": continuation,
            "prefix_token_ids_sha256": _token_sha256(prefix),
            "continuation_token_ids_sha256": _token_sha256(continuation),
            "source_model_id": SOURCE.repository,
            "source_model_revision": SOURCE.revision,
            "source_state_sha256": checksum,
            "source_state_bytes": state_nbytes(captured),
            "state_file": serialization,
            "source_prefill_wall_ms": prefill_wall_ms,
            "source_prefill_gpu_ms": prefill_gpu_ms,
            "source_prefill_cpu_ms": prefill_cpu_ms,
            "source_state_extraction_ms": extraction_ms,
            "source_state_extraction_cpu_ms": extraction_cpu_ms,
            "source_peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "SOURCE_4B": json_safe_score(score),
            "source_tokens_processed": 4096 + 64,
            "continuation_schedule": [1, 4, 16, 64],
        }
        _append_jsonl(SOURCE_RAW, record)
        results.append(record)
        print(f"LOCKED source {index + 1}/64", flush=True)
        del captured, cache, inputs
    del model
    release_model(None)
    return results


def _translation_pass(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if TRANSLATION_RAW.exists():
        raise FileExistsError("Translation LOCKED evidence already exists")
    TRANSLATED_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    translators = FrozenTranslators()
    frozen_manifest = verify_frozen_manifest()
    results = []
    for index, source_record in enumerate(source_rows):
        torch.cuda.reset_peak_memory_stats()
        source_path = ATTEMPT_ROOT / source_record["state_file"]["path"]
        source_state = _load_state(source_path)
        if state_checksum(source_state) != source_record["source_state_sha256"]:
            raise RuntimeError(f"Source state checksum mismatch before translation at document {index}")
        translated, timing = translate_full_state(source_state, translators)
        finite = _validate_finite_state(translated)
        translated_checksum = state_checksum(translated)
        translated_path = TRANSLATED_STATE_ROOT / f"translated_{index:03d}.pt"
        serialization = _save_state(translated_path, translated)
        record = {
            "document_index": index,
            "document_id": source_record["document_id"],
            "source_state_sha256": source_record["source_state_sha256"],
            "translated_state_sha256": translated_checksum,
            "translated_state_bytes": state_nbytes(translated),
            "translated_state_finite": finite,
            "state_file": serialization,
            "timing": timing,
            "translation_peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "translator_tokens_processed": 0,
            "translator_hash": frozen_manifest["translator_hash"],
        }
        _append_jsonl(TRANSLATION_RAW, record)
        results.append(record)
        print(f"LOCKED translation {index + 1}/64", flush=True)
        del source_state, translated
        torch.cuda.empty_cache()
    del translators
    release_model(None)
    return results


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _native_metrics(score: dict[str, Any]) -> dict[str, Any]:
    result = json_safe_score(score)
    result.update(
        {
            "top1_agreement_to_native": 1.0,
            "per_token_top1_agreement_to_native": [True] * 64,
            "mean_top5_overlap_to_native": 1.0,
            "per_token_top5_overlap_to_native": [1.0] * 64,
            "mean_js_to_native": 0.0,
            "per_token_js_to_native": [0.0] * 64,
            "mean_kl_native_to_condition": 0.0,
            "per_token_kl_native_to_condition": [0.0] * 64,
            "mean_logit_cosine_to_native": 1.0,
            "per_token_logit_cosine_to_native": [1.0] * 64,
        }
    )
    return result


def _target_pass(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    translation_rows: list[dict[str, Any]],
) -> None:
    if RAW_EVIDENCE.exists():
        raise FileExistsError("Canonical target evidence already exists")
    NATIVE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    device = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda index: rows[index]["selection_sha256"])
    donor = {order[position]: order[(position + 1) % len(order)] for position in range(len(order))}
    if any(recipient == giver for recipient, giver in donor.items()):
        raise RuntimeError("FULL_SHUFFLED rotation has a fixed point")
    for index, row in enumerate(rows):
        token_ids = row["token_ids"]
        prefix = token_ids[:4096]
        bridge = token_ids[4096]
        continuation = token_ids[4097:4161]
        source_record = source_rows[index]
        translation_record = translation_rows[index]
        if source_record["document_id"] != row["document_id"]:
            raise RuntimeError(f"Source/LOCKED row identity mismatch at document {index}")
        if translation_record["document_id"] != row["document_id"]:
            raise RuntimeError(f"Translation/LOCKED row identity mismatch at document {index}")
        if not translation_record["translated_state_finite"]:
            raise RuntimeError(f"Non-finite translated state at document {index}")
        source_state = _load_state(ATTEMPT_ROOT / source_record["state_file"]["path"])
        translated_state = _load_state(ATTEMPT_ROOT / translation_record["state_file"]["path"])
        donor_index = donor[index]
        shuffled_state = _load_state(
            ATTEMPT_ROOT / translation_rows[donor_index]["state_file"]["path"]
        )
        conditions: dict[str, Any] = {"SOURCE_4B": source_record["SOURCE_4B"]}

        torch.cuda.reset_peak_memory_stats()
        native_cache = DynamicCache(config=config)
        native_inputs = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        native_prefill_wall_ms, native_prefill_gpu_ms, native_prefill_cpu_ms = _timed_prefill(
            model, native_inputs, native_cache, 4096
        )
        native_prefix_state, native_extraction_ms, native_extraction_cpu_ms = _capture_timed(
            native_cache, config
        )
        native_state_checksum = state_checksum(native_prefix_state)
        native_path = NATIVE_STATE_ROOT / f"native_{index:03d}.pt"
        native_serialization = _save_state(native_path, native_prefix_state)
        native_score = score_from_cache_segmented(
            model,
            native_cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
            retain_logits=True,
            capture_convergence=True,
        )
        native_score["historical_prefix_tokens_processed_in_scoring_branch"] = 4096
        conditions["NATIVE_9B"] = _native_metrics(native_score)

        convergence = None
        condition_specs = (
            ("EMPTY_9B", None, False, False),
            ("KV_ONLY", translated_state, True, False),
            (
                "KV_GDN_DIRECT",
                compose_component_state(kv_state=translated_state, gdn_state=source_state),
                True,
                True,
            ),
            ("FULL_TRANSLATED", translated_state, True, True),
            ("FULL_SHUFFLED", shuffled_state, True, True),
        )
        install_timings = {}
        installed_bytes = {}
        for condition, state, include_kv, include_gdn in condition_specs:
            torch.cuda.synchronize()
            install_start = time.perf_counter()
            if state is None:
                condition_cache = DynamicCache(config=config)
                bytes_installed = 0
            else:
                condition_cache = install_components(
                    state,
                    config,
                    device=device,
                    include_kv=include_kv,
                    include_gdn_recurrent=include_gdn,
                    include_gdn_convolution=include_gdn,
                )
                bytes_installed = component_nbytes(
                    state, include_kv=include_kv, include_gdn=include_gdn
                )
            torch.cuda.synchronize()
            install_timings[condition] = (time.perf_counter() - install_start) * 1000.0
            installed_bytes[condition] = bytes_installed
            score = score_from_cache_segmented(
                model,
                condition_cache,
                prefix_length=4096,
                bridge_token_id=bridge,
                continuation_token_ids=continuation,
                retain_logits=True,
                capture_convergence=condition == "FULL_TRANSLATED",
            )
            distribution_metrics = compare_full_distributions(score, native_score)
            score.update(distribution_metrics)
            score["state_install_ms"] = install_timings[condition]
            score["bytes_installed"] = bytes_installed
            score["no_target_historical_prefill"] = True
            conditions[condition] = json_safe_score(score)
            if condition == "FULL_TRANSLATED":
                convergence = {
                    str(checkpoint): compare_convergence_states(
                        native_score["convergence_states"][checkpoint],
                        score["convergence_states"][checkpoint],
                    )
                    for checkpoint in (1, 4, 16, 64)
                }
            del condition_cache, score
            torch.cuda.empty_cache()

        if convergence is None:
            raise RuntimeError("Missing FULL_TRANSLATED convergence evidence")
        raw_record = {
            "document_index": index,
            "document_id": row["document_id"],
            "corpus": row["corpus"],
            "split": "LOCKED",
            "prefix_token_count": 4096,
            "bridge_token_id": bridge,
            "continuation_token_ids": continuation,
            "continuation_token_ids_sha256": _token_sha256(continuation),
            "prefix_token_ids_sha256": _token_sha256(prefix),
            "source_model_id": SOURCE.repository,
            "source_model_revision": SOURCE.revision,
            "target_model_id": TARGET.repository,
            "target_model_revision": TARGET.revision,
            "conditions": conditions,
            "state_checksums": {
                "source": source_record["source_state_sha256"],
                "translated": translation_record["translated_state_sha256"],
                "target_native_prefix": native_state_checksum,
                "shuffled_donor_translated": translation_rows[donor_index]["translated_state_sha256"],
            },
            "translator_hash": translation_record["translator_hash"],
            "state_bytes": {
                "source": source_record["source_state_bytes"],
                "translated": translation_record["translated_state_bytes"],
                "target_native_prefix": state_nbytes(native_prefix_state),
            },
            "shuffle": {
                "donor_document_index": donor_index,
                "donor_document_id": rows[donor_index]["document_id"],
                "fixed_point": False,
                "recipient_bridge_and_future_preserved": True,
            },
            "timing": {
                "native_9b_prefill_wall_ms": native_prefill_wall_ms,
                "native_9b_prefill_gpu_ms": native_prefill_gpu_ms,
                "native_9b_prefill_cpu_ms": native_prefill_cpu_ms,
                "source_4b_prefill_wall_ms": source_record["source_prefill_wall_ms"],
                "source_4b_prefill_gpu_ms": source_record["source_prefill_gpu_ms"],
                "source_4b_prefill_cpu_ms": source_record["source_prefill_cpu_ms"],
                "source_state_extraction_ms": source_record["source_state_extraction_ms"],
                "source_state_extraction_cpu_ms": source_record[
                    "source_state_extraction_cpu_ms"
                ],
                **translation_record["timing"],
                "state_install_ms": install_timings,
                "native_state_extraction_ms": native_extraction_ms,
                "native_state_extraction_cpu_ms": native_extraction_cpu_ms,
                "bridge_ms": {
                    condition: conditions[condition]["bridge_ms"]
                    for condition in conditions
                    if "bridge_ms" in conditions[condition]
                },
            },
            "bytes_moved": {
                "translation_host_to_device": translation_record["timing"]["bytes_host_to_device"],
                "translation_device_to_host": translation_record["timing"]["bytes_device_to_host"],
                "condition_state_install": installed_bytes,
            },
            "peak_vram_bytes": {
                "source_pass": source_record["source_peak_vram_bytes"],
                "translation": translation_record["translation_peak_vram_bytes"],
                "target_pass": torch.cuda.max_memory_allocated(),
            },
            "native_state_file": native_serialization,
            "state_convergence": convergence,
            "token_accounting": {
                "source_prefix": 4096,
                "translator": 0,
                "target_native_prefix": 4096,
                "target_handoff_prefix": 0,
                "bridge_and_teacher_forced_inputs_per_condition": 64,
                "continuation_schedule": [1, 4, 16, 64],
            },
        }
        _append_jsonl(RAW_EVIDENCE, raw_record)
        print(f"LOCKED target {index + 1}/64", flush=True)
        del (
            source_state,
            translated_state,
            shuffled_state,
            native_prefix_state,
            native_cache,
            native_inputs,
            native_score,
        )
        gc.collect()
        torch.cuda.empty_cache()
    del model
    release_model(None)


def main() -> None:
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["global"])
    started = begin_locked_run()
    rows = load_locked_rows(authorization=True)
    if [row["selection_sha256"] for row in rows] != sorted(
        row["selection_sha256"] for row in rows
    ):
        raise RuntimeError("LOCKED manifest is not in frozen hash order")
    run_start = time.perf_counter()
    source_rows = _source_pass(rows, prereg)
    translation_rows = _translation_pass(source_rows)
    _target_pass(rows, source_rows, translation_rows)
    finished = {
        **started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_COMPLETE",
        "documents": 64,
        "conditions": prereg["locked_conditions"],
        "wall_seconds": time.perf_counter() - run_start,
        "raw_evidence": str(RAW_EVIDENCE.relative_to(ATTEMPT_ROOT)).replace("\\", "/"),
        "raw_evidence_sha256": sha256_file(RAW_EVIDENCE),
        "source_pass_sha256": sha256_file(SOURCE_RAW),
        "translation_pass_sha256": sha256_file(TRANSLATION_RAW),
    }
    _write_json(EXECUTION_RECORD, finished)
    print(json.dumps(finished, indent=2))


if __name__ == "__main__":
    main()
