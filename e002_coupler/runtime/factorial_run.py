"""Prepare and evaluate the complete fresh E002 validation factorial."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
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
from experiments.latentport.e001_handoff.state.cache_state import capture_cache
from experiments.latentport.e001_handoff.translators.runtime_translation import (
    FrozenTranslators,
    translate_full_state,
)
from experiments.latentport.e002_coupler.analysis.statistics import summarize_factorial
from experiments.latentport.e002_coupler.factorial.composer import (
    compose_condition,
    condition_complexity,
)
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    E002_ROOT,
    FACTORIAL_CONDITIONS,
    ensure_attempt_layout,
    load_preregistration,
)
from experiments.latentport.e002_coupler.runtime.install import install_for_inference
from experiments.latentport.e002_coupler.runtime.scoring import compare_primary, score_primary
from experiments.latentport.e002_coupler.runtime.state_io import (
    load_state,
    save_state_once,
    write_json_once,
)


def _rows() -> list[dict[str, Any]]:
    path = E002_ROOT / "data" / "validation_factorial" / "manifest.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 32:
        raise RuntimeError("factorial manifest does not contain 32 documents")
    return rows


def _prefill(model, config, prefix: list[int]) -> tuple[DynamicCache, float]:
    device = next(model.parameters()).device
    cache = DynamicCache(config=config)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([prefix], dtype=torch.long, device=device),
            position_ids=position_ids(0, len(prefix), device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    return cache, (time.perf_counter() - started) * 1000.0


def _state_path(kind: str, index: int) -> Path:
    return ATTEMPT_ROOT / "factorial" / "states" / kind / f"{kind}_{index:03d}.pt"


def prepare_source(rows: list[dict[str, Any]]) -> None:
    config = load_text_config(SOURCE)
    model = load_language_model(SOURCE)
    for index, row in enumerate(rows):
        state_path = _state_path("source", index)
        metadata_path = ATTEMPT_ROOT / "factorial" / "source" / f"source_{index:03d}.json"
        if state_path.exists() and metadata_path.exists():
            continue
        token_ids = row["token_ids"]
        cache, prefill_ms = _prefill(model, config, token_ids[:4096])
        state = capture_cache(cache, config, device="cpu")
        state_record = save_state_once(state_path, state)
        write_json_once(
            metadata_path,
            {
                "document_id": row["document_id"],
                "split_index": index,
                "prefix_tokens": 4096,
                "source_prefill_ms": prefill_ms,
                "state": state_record,
            },
        )
        print(f"factorial source {index + 1}/{len(rows)}", flush=True)
    release_model(model)


def prepare_translated(rows: list[dict[str, Any]]) -> None:
    translators = FrozenTranslators(device="cuda:0")
    for index, row in enumerate(rows):
        state_path = _state_path("translated", index)
        metadata_path = ATTEMPT_ROOT / "factorial" / "translation" / f"translated_{index:03d}.json"
        if state_path.exists() and metadata_path.exists():
            continue
        source = load_state(_state_path("source", index))
        translated, timing = translate_full_state(source, translators)
        state_record = save_state_once(state_path, translated)
        write_json_once(
            metadata_path,
            {
                "document_id": row["document_id"],
                "split_index": index,
                "translation_timing": timing,
                "state": state_record,
            },
        )
        print(f"factorial translation {index + 1}/{len(rows)}", flush=True)
    del translators
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_target(rows: list[dict[str, Any]]) -> None:
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    for index, row in enumerate(rows):
        output_path = ATTEMPT_ROOT / "factorial" / "documents" / f"document_{index:03d}.json"
        if output_path.exists():
            continue
        token_ids = row["token_ids"]
        bridge = int(token_ids[4096])
        continuation = [int(value) for value in token_ids[4097:4161]]
        native_cache, native_prefill_ms = _prefill(model, config, token_ids[:4096])
        native = score_primary(
            model,
            native_cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
            retain_logits=True,
        )
        empty = score_primary(
            model,
            DynamicCache(config=config),
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
            retain_logits=True,
        )
        empty_comparison = compare_primary(empty, native)
        source = load_state(_state_path("source", index))
        translated = load_state(_state_path("translated", index))
        conditions = {}
        for condition in FACTORIAL_CONDITIONS:
            composed = compose_condition(source, translated, condition)
            torch.cuda.synchronize()
            install_started = time.perf_counter()
            cache = install_for_inference(composed, config, device="cuda:0")
            torch.cuda.synchronize()
            install_ms = (time.perf_counter() - install_started) * 1000.0
            score = score_primary(
                model,
                cache,
                prefix_length=4096,
                bridge_token_id=bridge,
                continuation_token_ids=continuation,
                retain_logits=True,
            )
            comparison = compare_primary(score, native)
            conditions[condition] = {
                **json_safe_score(score),
                **comparison,
                "state_install_ms": install_ms,
                "components": {"K": condition[0], "R": condition[1], "C": condition[2]},
            }
            del cache, score
        record = {
            "document_id": row["document_id"],
            "corpus": row["corpus"],
            "split": row["split"],
            "split_index": index,
            "prefix_token_count": 4096,
            "bridge_token_id": bridge,
            "continuation_token_ids": continuation,
            "native_9b": json_safe_score(native),
            "empty_9b": {**json_safe_score(empty), **empty_comparison},
            "conditions": conditions,
            "native_9b_prefill_ms": native_prefill_ms,
            "source_state_checksum": json.loads(
                (ATTEMPT_ROOT / "factorial" / "source" / f"source_{index:03d}.json").read_text(encoding="utf-8")
            )["state"]["state_checksum"],
            "translated_state_checksum": json.loads(
                (ATTEMPT_ROOT / "factorial" / "translation" / f"translated_{index:03d}.json").read_text(encoding="utf-8")
            )["state"]["state_checksum"],
            "target_historical_prefix_replay_in_handoff": 0,
        }
        write_json_once(output_path, record)
        del native, empty, source, translated
        torch.cuda.empty_cache()
        print(f"factorial target {index + 1}/{len(rows)}", flush=True)
    release_model(model)


def _selection_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        json.loads(
            (ATTEMPT_ROOT / "factorial" / "documents" / f"document_{index:03d}.json").read_text(
                encoding="utf-8"
            )
        )
        for index in range(len(rows))
    ]
    native = np.asarray([record["native_9b"]["nll"] for record in records], dtype=np.float64)
    document_deltas = []
    summary = {}
    for condition in FACTORIAL_CONDITIONS:
        nll = np.asarray([record["conditions"][condition]["nll"] for record in records])
        delta = nll - native
        summary[condition] = {
            "mean_nll": float(nll.mean()),
            "mean_delta_nll": float(delta.mean()),
            "mean_js_to_native": float(
                np.mean([record["conditions"][condition]["mean_js_to_native"] for record in records])
            ),
            "top1_agreement_to_native": float(
                np.mean(
                    [record["conditions"][condition]["top1_agreement_to_native"] for record in records]
                )
            ),
        }
    for document_index, record in enumerate(records):
        document_deltas.append(
            {
                condition: float(record["conditions"][condition]["nll"] - native[document_index])
                for condition in FACTORIAL_CONDITIONS
            }
        )
    effects = summarize_factorial(document_deltas)
    minimum = min(value["mean_delta_nll"] for value in summary.values())
    eligible = [
        condition
        for condition in FACTORIAL_CONDITIONS
        if summary[condition]["mean_delta_nll"] <= minimum + 0.01
    ]
    selected = min(eligible, key=lambda condition: (condition_complexity(condition), condition))
    table = [
        {
            "candidate": condition,
            "validation_nll": summary[condition]["mean_nll"],
            "validation_delta_nll": summary[condition]["mean_delta_nll"],
            "complexity_rank": condition_complexity(condition),
            "within_tie_tolerance": condition in eligible,
            "selected": condition == selected,
        }
        for condition in FACTORIAL_CONDITIONS
    ]
    payload = {
        "selected_state": selected,
        "selection_metric": "mean document-level DeltaNLL",
        "tie_tolerance_nats": 0.01,
        "selection_rule": "minimum; within 0.01 choose fewer translated components then lexical order",
        "documents": len(records),
        "condition_metrics": summary,
        "interaction_metrics": effects,
        "candidate_table": table,
    }
    payload["selection_payload_sha256"] = _selection_hash(payload)
    write_json_once(ATTEMPT_ROOT / "factorial" / "factorial_validation_statistics.json", payload)
    write_json_once(ATTEMPT_ROOT / "factorial" / "base_state_selection.json", payload)
    raw_path = ATTEMPT_ROOT / "factorial" / "raw_evidence.jsonl"
    with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("source", "translate", "target", "aggregate", "all"), default="all", nargs="?"
    )
    args = parser.parse_args()
    ensure_attempt_layout()
    seed_everything(load_preregistration()["seeds"]["global"])
    rows = _rows()
    if args.phase in ("source", "all"):
        prepare_source(rows)
    if args.phase in ("translate", "all"):
        prepare_translated(rows)
    if args.phase in ("target", "all"):
        evaluate_target(rows)
    if args.phase in ("aggregate", "all"):
        result = aggregate(rows)
        print(json.dumps({"selected_state": result["selected_state"], "condition_metrics": result["condition_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

