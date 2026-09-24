"""Execute the single authorized canonical E002 4K LOCKED run."""

from __future__ import annotations

import gc
import hashlib
import json
import time
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
from experiments.latentport.e001_handoff.state.cache_state import (
    capture_cache,
    state_checksum,
    state_nbytes,
)
from experiments.latentport.e001_handoff.state.convergence import compare_convergence_states
from experiments.latentport.e001_handoff.translators.runtime_translation import (
    FrozenTranslators,
    translate_full_state,
)
from experiments.latentport.e002_coupler.correction.serialization import (
    load_correction,
    verify_correction,
)
from experiments.latentport.e002_coupler.factorial.composer import compose_condition
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    FACTORIAL_CONDITIONS,
    PRIMARY_LOCKED_CONDITIONS,
    load_preregistration,
)
from experiments.latentport.e002_coupler.runtime.install import install_for_inference
from experiments.latentport.e002_coupler.runtime.lock_guard import (
    begin_locked_run,
    load_locked_rows,
    sha256_file,
    verify_frozen_manifest,
)
from experiments.latentport.e002_coupler.runtime.scoring import (
    capture_repair_trajectory,
    compare_primary,
    score_primary,
)
from experiments.latentport.e002_coupler.runtime.state_io import (
    load_logits,
    load_state,
    save_logits_once,
    save_state_once,
)


LOCKED_ROOT = ATTEMPT_ROOT / "locked"
SOURCE_ROOT = LOCKED_ROOT / "states" / "source"
TRANSLATED_ROOT = LOCKED_ROOT / "states" / "translated"
CORRECTED_ROOT = LOCKED_ROOT / "states" / "corrected"
LOGITS_ROOT = LOCKED_ROOT / "logits"
SOURCE_RAW = LOCKED_ROOT / "source_pass.jsonl"
TRANSLATION_RAW = LOCKED_ROOT / "translation_pass.jsonl"
CORRECTION_RAW = LOCKED_ROOT / "correction_pass.jsonl"
RAW_EVIDENCE = LOCKED_ROOT / "raw_evidence.jsonl"
EXECUTION_RECORD = LOCKED_ROOT / "locked_execution.json"


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in token_ids).encode("ascii")).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ATTEMPT_ROOT).as_posix()


def _state_file(root: Path, index: int, label: str) -> Path:
    return root / f"{label}_{index:03d}.pt"


def _logit_file(index: int, condition: str) -> Path:
    return LOGITS_ROOT / f"doc_{index:03d}" / f"{condition}.npy"


def _save_logits(index: int, condition: str, score: dict[str, Any]) -> dict[str, Any]:
    record = save_logits_once(_logit_file(index, condition), score["logits"])
    record["path"] = _relative(Path(record["path"]))
    return record


def _timed_prefill(model, input_ids: torch.Tensor, cache, prefix_length: int) -> dict[str, float]:
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
    return {
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "gpu_ms": float(start_event.elapsed_time(end_event)),
        "cpu_ms": (time.process_time() - cpu_start) * 1000.0,
    }


def _timed_capture(cache, config) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = capture_cache(cache, config, device="cpu")
    torch.cuda.synchronize()
    return state, (time.perf_counter() - started) * 1000.0


def _timed_install(state: dict[str, Any], config) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    cache = install_for_inference(state, config)
    torch.cuda.synchronize()
    return cache, (time.perf_counter() - started) * 1000.0


def _finite_state(state: dict[str, Any]) -> bool:
    for layer in state["layers"]:
        tensors = (
            (layer["keys"], layer["values"])
            if layer["layer_type"] == "full_attention"
            else (layer["recurrent_states"][0], layer["conv_states"][0])
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            return False
    return True


def _condition_tokens(row: dict[str, Any]) -> tuple[list[int], int, list[int], list[int]]:
    ids = [int(value) for value in row["token_ids"]]
    prefix = ids[:4096]
    bridge = ids[4096]
    continuation = ids[4097:4161]
    repair_inputs = ids[4097:4352]
    if not (len(prefix) == 4096 and len(continuation) == 64 and len(repair_inputs) == 255):
        raise RuntimeError("LOCKED token accounting failure")
    return prefix, bridge, continuation, repair_inputs


def _source_pass(rows: list[dict[str, Any]]) -> None:
    if SOURCE_RAW.exists():
        raise FileExistsError("E002 source LOCKED evidence already exists")
    config = load_text_config(SOURCE)
    model = load_language_model(SOURCE)
    device = next(model.parameters()).device
    for index, row in enumerate(rows):
        prefix, bridge, continuation, _ = _condition_tokens(row)
        inputs = torch.tensor([prefix], dtype=torch.long, device=device)
        cache = DynamicCache(config=config)
        timing = _timed_prefill(model, inputs, cache, 4096)
        state, extraction_ms = _timed_capture(cache, config)
        state_path = _state_file(SOURCE_ROOT, index, "source")
        state_record = save_state_once(state_path, state)
        score = score_primary(
            model,
            cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
        )
        score["historical_prefix_tokens_processed_in_scoring_branch"] = 4096
        logits = _save_logits(index, "SOURCE_4B", score)
        _append_jsonl(
            SOURCE_RAW,
            {
                "document_index": index,
                "document_id": row["document_id"],
                "state_file": _relative(state_path),
                "state_checksum": state_record["state_checksum"],
                "state_tensor_bytes": state_record["state_tensor_bytes"],
                "prefill": timing,
                "state_extraction_ms": extraction_ms,
                "score": json_safe_score(score),
                "logits": logits,
                "historical_prefix_tokens_processed": 4096,
            },
        )
        print(f"E002 LOCKED source {index + 1}/64", flush=True)
        del inputs, cache, state, score
    release_model(model)


def _translation_pass(source_rows: list[dict[str, Any]]) -> None:
    if TRANSLATION_RAW.exists():
        raise FileExistsError("E002 translation LOCKED evidence already exists")
    translators = FrozenTranslators()
    e001_manifest = json.loads(
        (Path(__file__).resolve().parents[2] / "e001_handoff" / "FROZEN_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    for index, source_record in enumerate(source_rows):
        source = load_state(ATTEMPT_ROOT / source_record["state_file"])
        if state_checksum(source) != source_record["state_checksum"]:
            raise RuntimeError(f"LOCKED source checksum mismatch at {index}")
        translated, timing = translate_full_state(source, translators)
        if not _finite_state(translated):
            raise RuntimeError(f"non-finite translated LOCKED state at {index}")
        path = _state_file(TRANSLATED_ROOT, index, "translated")
        record = save_state_once(path, translated)
        _append_jsonl(
            TRANSLATION_RAW,
            {
                "document_index": index,
                "document_id": source_record["document_id"],
                "state_file": _relative(path),
                "state_checksum": record["state_checksum"],
                "state_tensor_bytes": record["state_tensor_bytes"],
                "timing": timing,
                "translator_hash": e001_manifest["translator_hash"],
                "tokens_processed": 0,
            },
        )
        print(f"E002 LOCKED translation {index + 1}/64", flush=True)
        del source, translated
        torch.cuda.empty_cache()
    del translators
    release_model(None)


def _correction_pass(
    source_rows: list[dict[str, Any]], translation_rows: list[dict[str, Any]], base_condition: str
) -> None:
    if CORRECTION_RAW.exists():
        raise FileExistsError("E002 correction LOCKED evidence already exists")
    tensor_path = ATTEMPT_ROOT / "fit" / "correction.safetensors"
    metadata_path = ATTEMPT_ROOT / "fit" / "correction.json"
    metadata = verify_correction(tensor_path, metadata_path)
    coupler = load_correction(tensor_path, metadata_path, device="cuda:0")
    for index, (source_record, translation_record) in enumerate(
        zip(source_rows, translation_rows, strict=True)
    ):
        source = load_state(ATTEMPT_ROOT / source_record["state_file"])
        translated = load_state(ATTEMPT_ROOT / translation_record["state_file"])
        started = time.perf_counter()
        base = compose_condition(source, translated, base_condition)
        base_construction_ms = (time.perf_counter() - started) * 1000.0
        base_checksum = state_checksum(base)
        torch.cuda.synchronize()
        started = time.perf_counter()
        application = coupler.apply_to_state(base, detach_to_cpu=True)
        torch.cuda.synchronize()
        correction_ms = (time.perf_counter() - started) * 1000.0
        corrected = application.state
        if not _finite_state(corrected):
            raise RuntimeError(f"non-finite corrected LOCKED state at {index}")
        path = _state_file(CORRECTED_ROOT, index, "corrected")
        saved = save_state_once(path, corrected)
        layer_magnitudes = [
            {
                **{key: value for key, value in record.items() if key != "relative_norm"},
                "relative_norm": float(record["relative_norm"].detach().item()),
            }
            for record in application.layer_relative_magnitudes
        ]
        _append_jsonl(
            CORRECTION_RAW,
            {
                "document_index": index,
                "document_id": source_record["document_id"],
                "base_state": base_condition,
                "base_state_checksum": base_checksum,
                "corrected_state_file": _relative(path),
                "corrected_state_checksum": saved["state_checksum"],
                "corrected_state_tensor_bytes": saved["state_tensor_bytes"],
                "correction_hash": metadata["correction_hash"],
                "base_construction_ms": base_construction_ms,
                "joint_correction_ms": correction_ms,
                "component_relative_norm": {
                    name: float(value.detach().sqrt().item())
                    for name, value in application.component_relative_squared.items()
                },
                "layer_relative_magnitudes": layer_magnitudes,
            },
        )
        print(f"E002 LOCKED correction {index + 1}/64", flush=True)
        del source, translated, base, application, corrected
        torch.cuda.empty_cache()
    del coupler
    release_model(None)


def _native_metrics(score: dict[str, Any]) -> dict[str, Any]:
    result = json_safe_score(score)
    result.update(
        {
            "delta_nll_to_native": 0.0,
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


def _score_target_condition(
    model,
    config,
    *,
    state: dict[str, Any] | None,
    index: int,
    condition: str,
    bridge: int,
    continuation: list[int],
    native_score: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    if state is None:
        cache = DynamicCache(config=config)
        install_ms = 0.0
    else:
        cache, install_ms = _timed_install(state, config)
    score = score_primary(
        model,
        cache,
        prefix_length=4096,
        bridge_token_id=bridge,
        continuation_token_ids=continuation,
    )
    score.update(compare_primary(score, native_score))
    score["delta_nll_to_native"] = score["nll"] - native_score["nll"]
    score["state_install_ms"] = install_ms
    score["no_target_historical_prefill"] = True
    logits = _save_logits(index, condition, score)
    result = json_safe_score(score)
    result["logits"] = logits
    del cache, score
    return result, install_ms


def _target_pass(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    translation_rows: list[dict[str, Any]],
    correction_rows: list[dict[str, Any]],
    base_condition: str,
    correction_hash: str,
) -> None:
    if RAW_EVIDENCE.exists():
        raise FileExistsError("Canonical E002 target evidence already exists")
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    device = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda item: rows[item]["selection_sha256"])
    donor = {order[position]: order[(position + 1) % len(order)] for position in range(len(order))}
    if any(recipient == giver for recipient, giver in donor.items()):
        raise RuntimeError("JOINT_SHUFFLED rotation contains a fixed point")

    for index, row in enumerate(rows):
        prefix, bridge, continuation, repair_inputs = _condition_tokens(row)
        source_record = source_rows[index]
        translation_record = translation_rows[index]
        correction_record = correction_rows[index]
        if len({row["document_id"], source_record["document_id"], translation_record["document_id"], correction_record["document_id"]}) != 1:
            raise RuntimeError(f"LOCKED pass identity mismatch at document {index}")
        source = load_state(ATTEMPT_ROOT / source_record["state_file"])
        translated = load_state(ATTEMPT_ROOT / translation_record["state_file"])
        corrected = load_state(ATTEMPT_ROOT / correction_record["corrected_state_file"])
        donor_index = donor[index]
        shuffled = load_state(ATTEMPT_ROOT / correction_rows[donor_index]["corrected_state_file"])
        base = compose_condition(source, translated, base_condition)
        if state_checksum(base) != correction_record["base_state_checksum"]:
            raise RuntimeError(f"base state changed after correction freeze at {index}")

        native_inputs = torch.tensor([prefix], dtype=torch.long, device=device)
        native_cache = DynamicCache(config=config)
        native_prefill = _timed_prefill(model, native_inputs, native_cache, 4096)
        native_prefix_state, native_capture_ms = _timed_capture(native_cache, config)
        native_score = score_primary(
            model,
            native_cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
        )
        native_logits = _save_logits(index, "NATIVE_9B", native_score)
        conditions: dict[str, Any] = {"NATIVE_9B": _native_metrics(native_score)}
        conditions["NATIVE_9B"]["logits"] = native_logits

        source_score = dict(source_record["score"])
        source_score["logits"] = load_logits(
            ATTEMPT_ROOT / source_record["logits"]["path"], device="cpu"
        )
        source_score.update(compare_primary(source_score, native_score))
        source_score["delta_nll_to_native"] = source_score["nll"] - native_score["nll"]
        source_score.pop("logits")
        source_score["logits"] = source_record["logits"]
        conditions["SOURCE_4B"] = source_score

        install_timings: dict[str, float] = {}
        empty_result, install_timings["EMPTY_9B"] = _score_target_condition(
            model,
            config,
            state=None,
            index=index,
            condition="EMPTY_9B",
            bridge=bridge,
            continuation=continuation,
            native_score=native_score,
        )
        conditions["EMPTY_9B"] = empty_result
        for condition in FACTORIAL_CONDITIONS:
            condition_state = compose_condition(source, translated, condition)
            result, install_timings[condition] = _score_target_condition(
                model,
                config,
                state=condition_state,
                index=index,
                condition=condition,
                bridge=bridge,
                continuation=continuation,
                native_score=native_score,
            )
            conditions[condition] = result
            del condition_state
        for condition, state in (
            ("BASE_STATE", base),
            ("JOINT_CORRECTED", corrected),
            ("JOINT_SHUFFLED", shuffled),
        ):
            result, install_timings[condition] = _score_target_condition(
                model,
                config,
                state=state,
                index=index,
                condition=condition,
                bridge=bridge,
                continuation=continuation,
                native_score=native_score,
            )
            conditions[condition] = result
        if conditions[base_condition]["logits"]["content_sha256"] != conditions["BASE_STATE"]["logits"]["content_sha256"]:
            raise RuntimeError(f"explicit BASE_STATE is not bit-identical to {base_condition} at {index}")

        native_repair_cache, _ = _timed_install(native_prefix_state, config)
        native_trajectory = capture_repair_trajectory(
            model,
            native_repair_cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=repair_inputs,
        )
        repair: dict[str, Any] = {}
        for condition, state in (("BASE_STATE", base), ("JOINT_CORRECTED", corrected)):
            repair_cache, _ = _timed_install(state, config)
            trajectory = capture_repair_trajectory(
                model,
                repair_cache,
                prefix_length=4096,
                bridge_token_id=bridge,
                continuation_token_ids=repair_inputs,
            )
            repair[condition] = {
                str(checkpoint): compare_convergence_states(
                    native_trajectory[checkpoint], trajectory[checkpoint]
                )
                for checkpoint in (1, 4, 16, 64, 256)
            }
            del repair_cache, trajectory

        native_nll = conditions["NATIVE_9B"]["nll"]
        empty_nll = conditions["EMPTY_9B"]["nll"]
        source_nll = conditions["SOURCE_4B"]["nll"]
        base_nll = conditions["BASE_STATE"]["nll"]
        corrected_nll = conditions["JOINT_CORRECTED"]["nll"]
        denominator_ncr = empty_nll - native_nll
        denominator_tqr = source_nll - native_nll
        delta_base = base_nll - native_nll
        if denominator_ncr == 0 or denominator_tqr == 0:
            raise RuntimeError(f"undefined normalized metric denominator at document {index}")
        for value in conditions.values():
            value["delta_nll_to_native"] = value["nll"] - native_nll
            value["ncr"] = (empty_nll - value["nll"]) / denominator_ncr
            value["tqr"] = (source_nll - value["nll"]) / denominator_tqr
        derived = {
            "NCR": (empty_nll - corrected_nll) / denominator_ncr,
            "TQR": (source_nll - corrected_nll) / denominator_tqr,
            "RGR": (delta_base - (corrected_nll - native_nll)) / delta_base
            if delta_base > 0
            else None,
        }
        raw = {
            "document_index": index,
            "document_id": row["document_id"],
            "corpus": row["corpus"],
            "split": "LOCKED",
            "prefix_token_ids": prefix,
            "bridge_token_id": bridge,
            "continuation_token_ids": continuation,
            "repair_continuation_input_ids": repair_inputs,
            "prefix_token_ids_sha256": _token_hash(prefix),
            "continuation_token_ids_sha256": _token_hash(continuation),
            "source_model": {"repository": SOURCE.repository, "revision": SOURCE.revision},
            "target_model": {"repository": TARGET.repository, "revision": TARGET.revision},
            "base_state": base_condition,
            "correction_hash": correction_hash,
            "conditions": conditions,
            "derived": derived,
            "state_checksums": {
                "source": source_record["state_checksum"],
                "translated": translation_record["state_checksum"],
                "base": correction_record["base_state_checksum"],
                "corrected": correction_record["corrected_state_checksum"],
                "joint_shuffled_donor": correction_rows[donor_index]["corrected_state_checksum"],
                "native_prefix": state_checksum(native_prefix_state),
            },
            "correction_magnitude": {
                "component_relative_norm": correction_record["component_relative_norm"],
                "layer_relative_magnitudes": correction_record["layer_relative_magnitudes"],
            },
            "state_repair": repair,
            "shuffle": {
                "donor_document_index": donor_index,
                "donor_document_id": rows[donor_index]["document_id"],
                "fixed_point": False,
                "recipient_bridge_and_continuation_preserved": True,
                "identical_prefix_length": True,
            },
            "timing": {
                "native_9b_prefill": native_prefill,
                "source_4b_prefill": source_record["prefill"],
                "base_construction_ms": correction_record["base_construction_ms"],
                "joint_correction_ms": correction_record["joint_correction_ms"],
                "state_install_ms": install_timings,
                "bridge_ms": {
                    name: value["bridge_ms"]
                    for name, value in conditions.items()
                    if "bridge_ms" in value
                },
                "source_state_extraction_ms": source_record["state_extraction_ms"],
                "native_state_extraction_ms": native_capture_ms,
                "translation": translation_record["timing"],
            },
            "token_accounting": {
                "source_historical_prefix": 4096,
                "native_target_historical_prefix": 4096,
                "handoff_target_historical_prefix": 0,
                "primary_continuation_targets": 64,
                "repair_checkpoints": [1, 4, 16, 64, 256],
            },
        }
        _append_jsonl(RAW_EVIDENCE, raw)
        print(f"E002 LOCKED target {index + 1}/64", flush=True)
        del (
            source,
            translated,
            corrected,
            shuffled,
            base,
            native_inputs,
            native_cache,
            native_prefix_state,
            native_score,
            native_repair_cache,
            native_trajectory,
        )
        gc.collect()
        torch.cuda.empty_cache()
    release_model(model)


def main() -> None:
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["global"])
    started = begin_locked_run()
    frozen = verify_frozen_manifest()
    rows = load_locked_rows(authorization=True)
    if [row["selection_sha256"] for row in rows] != sorted(
        row["selection_sha256"] for row in rows
    ):
        raise RuntimeError("E002 LOCKED manifest is not frozen hash order")
    run_started = time.perf_counter()
    _source_pass(rows)
    source_rows = _load_jsonl(SOURCE_RAW)
    _translation_pass(source_rows)
    translation_rows = _load_jsonl(TRANSLATION_RAW)
    _correction_pass(source_rows, translation_rows, frozen["base_state"])
    correction_rows = _load_jsonl(CORRECTION_RAW)
    _target_pass(
        rows,
        source_rows,
        translation_rows,
        correction_rows,
        frozen["base_state"],
        frozen["correction_hash"],
    )
    finished = {
        **started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_COMPLETE",
        "documents": 64,
        "conditions": list(PRIMARY_LOCKED_CONDITIONS),
        "wall_seconds": time.perf_counter() - run_started,
        "source_pass_sha256": sha256_file(SOURCE_RAW),
        "translation_pass_sha256": sha256_file(TRANSLATION_RAW),
        "correction_pass_sha256": sha256_file(CORRECTION_RAW),
        "raw_evidence_sha256": sha256_file(RAW_EVIDENCE),
    }
    _write_json_once(EXECUTION_RECORD, finished)
    print(json.dumps(finished, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
