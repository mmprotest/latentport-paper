"""One training-side smoke of corrected scoring and repair through token 256."""

from __future__ import annotations

import json

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import TARGET
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import position_ids
from experiments.latentport.e001_handoff.state.cache_state import capture_cache
from experiments.latentport.e001_handoff.state.convergence import compare_convergence_states
from experiments.latentport.e002_coupler.correction.serialization import load_correction, verify_correction
from experiments.latentport.e002_coupler.factorial.composer import compose_condition
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT, load_preregistration
from experiments.latentport.e002_coupler.runtime.install import install_for_inference
from experiments.latentport.e002_coupler.runtime.scoring import capture_repair_trajectory, compare_primary, score_primary
from experiments.latentport.e002_coupler.runtime.state_io import load_state, write_json_once


def main() -> None:
    output = ATTEMPT_ROOT / "implementation" / "locked_path_smoke.json"
    if output.exists():
        raise FileExistsError("LOCKED-path smoke evidence already exists")
    seed_everything(load_preregistration()["seeds"]["global"])
    with (E002_ROOT / "data" / "validation_factorial" / "manifest.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    ids = row["token_ids"]
    prefix, bridge, continuation = ids[:4096], int(ids[4096]), [int(value) for value in ids[4097:4161]]
    repair_inputs = (continuation * 4)[:255]
    source = load_state(ATTEMPT_ROOT / "factorial" / "states" / "source" / "source_000.pt")
    translated = load_state(
        ATTEMPT_ROOT / "factorial" / "states" / "translated" / "translated_000.pt"
    )
    base_name = json.loads(
        (ATTEMPT_ROOT / "factorial" / "base_state_selection.json").read_text(encoding="utf-8")
    )["selected_state"]
    base = compose_condition(source, translated, base_name)
    metadata = verify_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
    )
    coupler = load_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
        device="cuda:0",
    )
    application = coupler.apply_to_state(base, detach_to_cpu=True)
    corrected = application.state
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    device = next(model.parameters()).device
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("target weights unexpectedly trainable in LOCKED-path smoke")
    native_cache = DynamicCache(config=config)
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([prefix], dtype=torch.long, device=device),
            position_ids=position_ids(0, 4096, device),
            past_key_values=native_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    native_prefix = capture_cache(native_cache, config, device="cpu")
    native_score = score_primary(
        model,
        native_cache,
        prefix_length=4096,
        bridge_token_id=bridge,
        continuation_token_ids=continuation,
    )
    scores = {}
    for name, state in (("BASE_STATE", base), ("JOINT_CORRECTED", corrected)):
        cache = install_for_inference(state, config)
        score = score_primary(
            model,
            cache,
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=continuation,
        )
        comparison = compare_primary(score, native_score)
        scores[name] = {
            "nll": score["nll"],
            "logits_shape": list(score["logits"].shape),
            "all_logits_finite": bool(torch.isfinite(score["logits"]).all()),
            "js_to_native": comparison["mean_js_to_native"],
        }
        del cache, score
    native_repair = capture_repair_trajectory(
        model,
        install_for_inference(native_prefix, config),
        prefix_length=4096,
        bridge_token_id=bridge,
        continuation_token_ids=repair_inputs,
    )
    repair = {}
    for name, state in (("BASE_STATE", base), ("JOINT_CORRECTED", corrected)):
        trajectory = capture_repair_trajectory(
            model,
            install_for_inference(state, config),
            prefix_length=4096,
            bridge_token_id=bridge,
            continuation_token_ids=repair_inputs,
        )
        repair[name] = {
            str(checkpoint): compare_convergence_states(native_repair[checkpoint], trajectory[checkpoint])[
                "mean_gdn_recurrent_normalized_frobenius"
            ]
            for checkpoint in (1, 4, 16, 64, 256)
        }
    record = {
        "status": "PASS",
        "training_side_document": row["document_id"],
        "canonical_locked_documents_accessed": 0,
        "base_state": base_name,
        "correction_hash": metadata["correction_hash"],
        "target_historical_prefix_tokens_in_handoff": 0,
        "conditions": scores,
        "repair_recurrent_error": repair,
        "repair_checkpoints": [1, 4, 16, 64, 256],
        "target_weights_trainable": 0,
    }
    write_json_once(output, record)
    release_model(model)
    del coupler
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
