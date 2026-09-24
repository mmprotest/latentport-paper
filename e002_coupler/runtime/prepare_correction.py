"""Prepare source/base states and fresh native logits for correction fitting."""

from __future__ import annotations

import argparse
import gc
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
from experiments.latentport.e001_handoff.runtime.scoring import position_ids
from experiments.latentport.e001_handoff.state.cache_state import capture_cache
from experiments.latentport.e001_handoff.translators.runtime_translation import (
    FrozenTranslators,
    translate_full_state,
)
from experiments.latentport.e002_coupler.factorial.composer import compose_condition
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    E001_ROOT,
    ensure_attempt_layout,
    load_preregistration,
)
from experiments.latentport.e002_coupler.runtime.data_manifests import read_training_rows
from experiments.latentport.e002_coupler.runtime.scoring import training_logits
from experiments.latentport.e002_coupler.runtime.state_io import (
    load_state,
    save_logits_once,
    save_state_once,
    write_json_once,
)


SPLIT_MAP = {"fit_correction": "fit", "validation_correction": "validation"}


def _base_condition() -> str:
    path = ATTEMPT_ROOT / "factorial" / "base_state_selection.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected_state"]


def _paths(split: str, index: int) -> dict[str, Path]:
    short = SPLIT_MAP[split]
    root = ATTEMPT_ROOT / short / "correction_data"
    return {
        "source_state": root / "source_states" / f"source_{index:03d}.pt",
        "base_state": root / "base_states" / f"base_{index:03d}.pt",
        "native_logits": root / "native_logits" / f"native_{index:03d}.npy",
        "source_record": root / "source_records" / f"source_{index:03d}.json",
        "base_record": root / "base_records" / f"base_{index:03d}.json",
        "native_record": root / "native_records" / f"native_{index:03d}.json",
    }


def _e001_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    short = SPLIT_MAP[split]
    root = E001_ROOT / "artifacts" / "attempt_001" / short / "paired_states"
    recurrent = np.load(root / "source_recurrent.npy", mmap_mode="r")
    convolution = np.load(root / "source_convolution_bf16_u16.npy", mmap_mode="r")
    return recurrent, convolution


def _replace_with_e001_last_checkpoint(
    state: dict[str, Any], recurrent: np.ndarray, convolution: np.ndarray, document_index: int
) -> dict[str, Any]:
    flat_index = document_index * 8 + 7
    gdn_position = 0
    for layer in state["layers"]:
        if layer["layer_type"] != "linear_attention":
            continue
        sealed_recurrent = torch.from_numpy(np.array(recurrent[flat_index, gdn_position], copy=True)).unsqueeze(0)
        sealed_conv_bits = torch.from_numpy(np.array(convolution[flat_index, gdn_position], copy=True))
        sealed_convolution = sealed_conv_bits.view(torch.bfloat16).unsqueeze(0)
        if not torch.equal(layer["recurrent_states"][0], sealed_recurrent):
            raise RuntimeError(
                f"E001 recurrent reuse mismatch at document {document_index}, GDN {gdn_position}"
            )
        if not torch.equal(layer["conv_states"][0], sealed_convolution):
            raise RuntimeError(
                f"E001 convolution reuse mismatch at document {document_index}, GDN {gdn_position}"
            )
        # Use the sealed memmap values, not merely the recaptured equality copy.
        layer["recurrent_states"][0] = sealed_recurrent
        layer["conv_states"][0] = sealed_convolution
        gdn_position += 1
    if gdn_position != 24:
        raise RuntimeError("training state did not contain 24 GDN layers")
    state["e002_e001_paired_state_reused"] = True
    state["e002_e001_checkpoint_index"] = flat_index
    return state


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


def prepare_source(split: str) -> None:
    rows = read_training_rows(split)
    recurrent, convolution = _e001_arrays(split)
    config = load_text_config(SOURCE)
    model = load_language_model(SOURCE)
    for index, row in enumerate(rows):
        paths = _paths(split, index)
        if paths["source_state"].exists() and paths["source_record"].exists():
            continue
        cache, prefill_ms = _prefill(model, config, row["token_ids"][:1024])
        state = capture_cache(cache, config, device="cpu")
        state = _replace_with_e001_last_checkpoint(state, recurrent, convolution, index)
        state_record = save_state_once(paths["source_state"], state)
        write_json_once(
            paths["source_record"],
            {
                "document_id": row["document_id"],
                "split": split,
                "split_index": index,
                "prefix_tokens": 1024,
                "source_prefill_ms": prefill_ms,
                "e001_last_checkpoint_recurrent_equal": True,
                "e001_last_checkpoint_convolution_equal": True,
                "e001_flat_checkpoint_index": index * 8 + 7,
                "state": state_record,
            },
        )
        print(f"correction source {split} {index + 1}/{len(rows)}", flush=True)
    release_model(model)


def prepare_base(split: str) -> None:
    rows = read_training_rows(split)
    base_condition = _base_condition()
    translators = FrozenTranslators(device="cuda:0")
    for index, row in enumerate(rows):
        paths = _paths(split, index)
        if paths["base_state"].exists() and paths["base_record"].exists():
            continue
        source = load_state(paths["source_state"])
        translated, timing = translate_full_state(source, translators)
        base = compose_condition(source, translated, base_condition)
        base["e002_identity_anchor"] = base_condition
        state_record = save_state_once(paths["base_state"], base)
        write_json_once(
            paths["base_record"],
            {
                "document_id": row["document_id"],
                "split": split,
                "split_index": index,
                "base_state": base_condition,
                "translation_timing": timing,
                "state": state_record,
            },
        )
        print(f"correction base {split} {index + 1}/{len(rows)}", flush=True)
    del translators
    gc.collect()
    torch.cuda.empty_cache()


def prepare_native(split: str) -> None:
    rows = read_training_rows(split)
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    for index, row in enumerate(rows):
        paths = _paths(split, index)
        if paths["native_logits"].exists() and paths["native_record"].exists():
            continue
        token_ids = row["token_ids"]
        cache, prefill_ms = _prefill(model, config, token_ids[:1024])
        with torch.inference_mode():
            logits = training_logits(
                model,
                cache,
                prefix_length=1024,
                bridge_token_id=int(token_ids[1024]),
                continuation_token_ids=[int(value) for value in token_ids[1025:1034]],
            )
        logits_record = save_logits_once(paths["native_logits"], logits)
        write_json_once(
            paths["native_record"],
            {
                "document_id": row["document_id"],
                "split": split,
                "split_index": index,
                "prefix_tokens": 1024,
                "bridge_token_id": int(token_ids[1024]),
                "continuation_token_ids": [int(value) for value in token_ids[1025:1034]],
                "native_9b_prefill_ms": prefill_ms,
                "logits": logits_record,
            },
        )
        print(f"native logits {split} {index + 1}/{len(rows)}", flush=True)
    release_model(model)


def write_inventory() -> None:
    inventory = {"base_state": _base_condition(), "splits": {}}
    for split, expected in (("fit_correction", 128), ("validation_correction", 32)):
        rows = read_training_rows(split)
        if len(rows) != expected:
            raise RuntimeError("correction split count mismatch")
        complete = []
        for index, row in enumerate(rows):
            paths = _paths(split, index)
            if not all(paths[name].is_file() for name in paths):
                raise RuntimeError(f"incomplete correction preparation for {split} document {index}")
            complete.append(row["document_id"])
        inventory["splits"][split] = {
            "documents": len(rows),
            "document_ids": complete,
            "e001_locked_documents": 0,
            "target_historical_prefix_replay_in_handoff": 0,
        }
    write_json_once(ATTEMPT_ROOT / "fit" / "correction_data_inventory.json", inventory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "base", "native", "inventory", "all"), nargs="?", default="all")
    parser.add_argument("--split", choices=tuple(SPLIT_MAP), default=None)
    args = parser.parse_args()
    ensure_attempt_layout()
    seed_everything(load_preregistration()["seeds"]["global"])
    splits = [args.split] if args.split else list(SPLIT_MAP)
    for split in splits:
        if args.phase in ("source", "all"):
            prepare_source(split)
        if args.phase in ("base", "all"):
            prepare_base(split)
        if args.phase in ("native", "all"):
            prepare_native(split)
    if args.phase in ("inventory", "all"):
        write_inventory()


if __name__ == "__main__":
    main()

