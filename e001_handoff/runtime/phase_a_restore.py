"""Execute the exact same-model state capture/restore implementation gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.analysis.metrics import compare_logits
from experiments.latentport.e001_handoff.runtime.constants import (
    MODEL_SPECS,
    STATE_VALIDATION_ROOT,
    load_preregistration,
)
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    load_tokenizer,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.validation_contexts import (
    build_restore_context,
    token_stream_sha256,
)
from experiments.latentport.e001_handoff.state.cache_state import (
    caches_share_storage,
    capture_cache,
    restore_cache,
    state_checksum,
    state_nbytes,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _position_ids(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


@torch.inference_mode()
def validate_one(
    model,
    config,
    token_ids: list[int],
    *,
    prefix_length: int,
    context_index: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    tokens = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
    prefix = tokens[:, :prefix_length]
    bridge = tokens[:, prefix_length : prefix_length + 1]
    continuation = tokens[:, prefix_length + 1 : prefix_length + 65]
    if continuation.shape[1] != 64:
        raise ValueError("Deterministic stream did not contain 64 continuation targets")

    uninterrupted_cache = DynamicCache(config=config)
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    model(
        input_ids=prefix,
        position_ids=_position_ids(0, prefix_length, device),
        past_key_values=uninterrupted_cache,
        use_cache=True,
        logits_to_keep=1,
    )
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

    captured = capture_cache(uninterrupted_cache, config)
    captured_checksum = state_checksum(captured)
    restored_cache = restore_cache(captured, config, device=device)
    restored_checksum = state_checksum(capture_cache(restored_cache, config))
    if caches_share_storage(uninterrupted_cache, restored_cache, config):
        raise RuntimeError("Restored cache shares mutable tensor storage with uninterrupted cache")
    if captured_checksum != restored_checksum:
        raise RuntimeError("Restored cache checksum differs before continuation")

    bridge_position = _position_ids(prefix_length, 1, device)
    uninterrupted_bridge = model(
        input_ids=bridge,
        position_ids=bridge_position,
        past_key_values=uninterrupted_cache,
        use_cache=True,
        logits_to_keep=1,
    ).logits[:, -1, :]
    restored_bridge = model(
        input_ids=bridge,
        position_ids=bridge_position,
        past_key_values=restored_cache,
        use_cache=True,
        logits_to_keep=1,
    ).logits[:, -1, :]

    # Bridge logits predict continuation token 1. Feeding targets 1..63 in one
    # teacher-forced block produces predictions for targets 2..64.
    follow_inputs = continuation[:, :-1]
    follow_positions = _position_ids(prefix_length + 1, follow_inputs.shape[1], device)
    uninterrupted_follow = model(
        input_ids=follow_inputs,
        position_ids=follow_positions,
        past_key_values=uninterrupted_cache,
        use_cache=True,
        logits_to_keep=follow_inputs.shape[1],
    ).logits
    restored_follow = model(
        input_ids=follow_inputs,
        position_ids=follow_positions,
        past_key_values=restored_cache,
        use_cache=True,
        logits_to_keep=follow_inputs.shape[1],
    ).logits

    uninterrupted_logits = torch.cat([uninterrupted_bridge[:, None, :], uninterrupted_follow], dim=1)
    restored_logits = torch.cat([restored_bridge[:, None, :], restored_follow], dim=1)
    per_position = [
        compare_logits(uninterrupted_logits[0, position], restored_logits[0, position])
        for position in range(64)
    ]
    return {
        "context_index": context_index,
        "prefix_tokens": prefix_length,
        "bridge_token_id": int(bridge.item()),
        "continuation_token_ids": continuation[0].tolist(),
        "token_stream_sha256": token_stream_sha256(token_ids),
        "prefill_ms": prefill_ms,
        "captured_state_bytes": state_nbytes(captured),
        "captured_state_sha256": captured_checksum,
        "restored_state_sha256_before_bridge": restored_checksum,
        "deep_clone_no_shared_storage": True,
        "next_token": per_position[0],
        "teacher_forced_64": per_position,
        "top1_agreement_rate": sum(row["top1_agreement"] for row in per_position) / 64.0,
        "minimum_logit_cosine": min(row["cosine"] for row in per_position),
        "maximum_absolute_logit_difference": max(row["max_abs_diff"] for row in per_position),
        "minimum_top5_overlap": min(row["topk_overlap"] for row in per_position),
        "all_finite": all(row["finite"] for row in per_position),
    }


def run(model_role: str, mode: str) -> dict[str, Any]:
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["restore_contexts"])
    spec = MODEL_SPECS[model_role]
    config = load_text_config(spec)
    tokenizer = load_tokenizer()
    model = load_language_model(spec)
    if mode == "smoke":
        lengths = [256, 512]
    else:
        lengths = [value for value in prereg["phase_a"]["context_lengths"] for _ in range(4)]
    rows = []
    for context_index, prefix_length in enumerate(lengths):
        token_ids = build_restore_context(tokenizer, context_index, prefix_length + 65)
        row = validate_one(
            model,
            config,
            token_ids,
            prefix_length=prefix_length,
            context_index=context_index,
        )
        rows.append(row)
        print(
            f"{spec.role} {mode} {context_index + 1}/{len(lengths)} "
            f"L={prefix_length} cosine={row['minimum_logit_cosine']:.9f} "
            f"max_abs={row['maximum_absolute_logit_difference']:.9g}",
            flush=True,
        )

    tolerance = prereg["phase_a"]["max_abs_logit_diff_tolerance"]
    observed_max = max(row["maximum_absolute_logit_difference"] for row in rows)
    top1 = sum(
        sum(position["top1_agreement"] for position in row["teacher_forced_64"])
        for row in rows
    ) / (len(rows) * 64.0)
    min_next_cosine = min(row["next_token"]["cosine"] for row in rows)
    pass_without_tolerance = (
        top1 == prereg["phase_a"]["top1_agreement_required"]
        and min_next_cosine >= prereg["phase_a"]["min_next_logit_cosine"]
        and all(row["all_finite"] for row in rows)
    )
    passed = pass_without_tolerance and (tolerance is None or observed_max <= tolerance)
    summary = {
        "phase": "A_same_model_restore",
        "mode": mode,
        "model_role": spec.role,
        "model_repository": spec.repository,
        "model_revision": spec.revision,
        "contexts": len(rows),
        "context_lengths": lengths,
        "top1_agreement": top1,
        "minimum_next_token_logit_cosine": min_next_cosine,
        "minimum_any_position_logit_cosine": min(row["minimum_logit_cosine"] for row in rows),
        "maximum_absolute_logit_difference": observed_max,
        "frozen_max_abs_tolerance": tolerance,
        "all_finite": all(row["all_finite"] for row in rows),
        "pass": passed,
        "raw_contexts": rows,
    }
    suffix = "smoke" if mode == "smoke" else ("4b" if spec.role == "source" else "9b")
    _write_json(STATE_VALIDATION_ROOT / f"same_model_restore_{suffix}.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "raw_contexts"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("source", "target", "4b", "9b"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    result = run(args.model, args.mode)
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
