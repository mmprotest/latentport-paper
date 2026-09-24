"""Compact E002 implementation smoke without rerunning E001 science."""

from __future__ import annotations

import json
from typing import Any

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    load_tokenizer,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import position_ids
from experiments.latentport.e001_handoff.runtime.validation_contexts import build_restore_context
from experiments.latentport.e001_handoff.state.cache_state import capture_cache, restore_cache, state_checksum
from experiments.latentport.e002_coupler.correction.losses import behavior_kl
from experiments.latentport.e002_coupler.correction.model import IdentityAnchoredCoupler
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, ensure_attempt_layout, load_preregistration
from experiments.latentport.e002_coupler.runtime.install import install_differentiable
from experiments.latentport.e002_coupler.runtime.scoring import training_logits
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


PREFIX = 256


def _prefill(model, config, prefix_ids: list[int]) -> DynamicCache:
    device = next(model.parameters()).device
    cache = DynamicCache(config=config)
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([prefix_ids], dtype=torch.long, device=device),
            position_ids=position_ids(0, len(prefix_ids), device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    return cache


def _same_model_role(spec, token_ids: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_text_config(spec)
    model = load_language_model(spec)
    cache = _prefill(model, config, token_ids[:PREFIX])
    captured = capture_cache(cache, config, device="cpu")
    native_cache = restore_cache(captured, config, device="cuda:0")
    restored_cache = restore_cache(captured, config, device="cuda:0")
    with torch.inference_mode():
        native = training_logits(
            model,
            native_cache,
            prefix_length=PREFIX,
            bridge_token_id=token_ids[PREFIX],
            continuation_token_ids=token_ids[PREFIX + 1 : PREFIX + 10],
        )
        restored = training_logits(
            model,
            restored_cache,
            prefix_length=PREFIX,
            bridge_token_id=token_ids[PREFIX],
            continuation_token_ids=token_ids[PREFIX + 1 : PREFIX + 10],
        )
    result = {
        "repository": spec.repository,
        "revision": spec.revision,
        "prefix_tokens": PREFIX,
        "logits_equal": bool(torch.equal(native, restored)),
        "maximum_absolute_logit_difference": float((native.float() - restored.float()).abs().max().item()),
        "top1_agreement": float(
            (native.argmax(dim=-1) == restored.argmax(dim=-1)).float().mean().item()
        ),
        "state_checksum": state_checksum(captured),
    }
    if not result["logits_equal"] or result["maximum_absolute_logit_difference"] != 0.0:
        raise RuntimeError(f"same-model restore failed for {spec.repository}")
    release_model(model)
    return result, captured


def run_smoke() -> dict[str, Any]:
    ensure_attempt_layout()
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["same_model_restore"])
    tokenizer = load_tokenizer()
    token_ids = build_restore_context(tokenizer, 2002, PREFIX + 10)

    source_result, source_state = _same_model_role(SOURCE, token_ids)
    target_result, target_state = _same_model_role(TARGET, token_ids)

    target_config = load_text_config(TARGET)
    target_model = load_language_model(TARGET)
    target_device = next(target_model.parameters()).device
    coupler = IdentityAnchoredCoupler(2, basis_seed=prereg["seeds"]["fixed_basis"]).to(target_device)
    if not coupler.all_trainable_parameters_are_zero():
        raise RuntimeError("correction did not initialize to zero")
    application = coupler.apply_to_state(source_state)
    for base_layer, corrected_layer in zip(source_state["layers"], application.state["layers"], strict=True):
        if base_layer["layer_type"] == "full_attention":
            if not torch.equal(base_layer["keys"].to(target_device), corrected_layer["keys"]):
                raise RuntimeError("zero KV correction changed base keys")
            if not torch.equal(base_layer["values"].to(target_device), corrected_layer["values"]):
                raise RuntimeError("zero KV correction changed base values")
        else:
            if not torch.equal(
                base_layer["recurrent_states"][0].to(target_device),
                corrected_layer["recurrent_states"][0],
            ):
                raise RuntimeError("zero recurrent correction changed the base")
            if not torch.equal(
                base_layer["conv_states"][0].to(target_device), corrected_layer["conv_states"][0]
            ):
                raise RuntimeError("zero convolution correction changed the base")

    canonical_cache = restore_cache(source_state, target_config, device=target_device)
    with torch.inference_mode():
        canonical_logits = training_logits(
            target_model,
            canonical_cache,
            prefix_length=PREFIX,
            bridge_token_id=token_ids[PREFIX],
            continuation_token_ids=token_ids[PREFIX + 1 : PREFIX + 10],
        )
        native_logits = training_logits(
            target_model,
            restore_cache(target_state, target_config, device=target_device),
            prefix_length=PREFIX,
            bridge_token_id=token_ids[PREFIX],
            continuation_token_ids=token_ids[PREFIX + 1 : PREFIX + 10],
        ).detach()
    differentiable_cache = install_differentiable(application.state, target_config)
    differentiable_logits = training_logits(
        target_model,
        differentiable_cache,
        prefix_length=PREFIX,
        bridge_token_id=token_ids[PREFIX],
        continuation_token_ids=token_ids[PREFIX + 1 : PREFIX + 10],
    )
    differentiable_equivalent = bool(torch.equal(canonical_logits, differentiable_logits.detach()))
    if not differentiable_equivalent:
        maximum = float((canonical_logits.float() - differentiable_logits.detach().float()).abs().max().item())
        raise RuntimeError(f"differentiable install is not numerically equivalent; max={maximum}")
    loss = behavior_kl(native_logits, differentiable_logits)
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.float().square().sum()
            for parameter in coupler.parameters()
            if parameter.grad is not None
        )
    )
    gradient_finite_nonzero = bool(torch.isfinite(gradient_norm) and gradient_norm > 0)
    if not gradient_finite_nonzero:
        raise RuntimeError("zero-initialized correction has no usable first-order gradient")
    result = {
        "status": "PASS",
        "source_same_model_restore": source_result,
        "target_same_model_restore": target_result,
        "zero_correction_exact": True,
        "differentiable_install_exact": differentiable_equivalent,
        "behavior_kl_finite": bool(torch.isfinite(loss)),
        "behavior_kl": float(loss.detach().item()),
        "zero_initialization_gradient_norm": float(gradient_norm.detach().item()),
        "zero_initialization_gradient_finite_nonzero": gradient_finite_nonzero,
        "target_trainable_parameters": sum(
            parameter.numel() for parameter in target_model.parameters() if parameter.requires_grad
        ),
        "correction_parameters": coupler.parameter_count,
        "target_historical_prefix_tokens_in_handoff": 0,
    }
    release_model(target_model)
    write_json_once(ATTEMPT_ROOT / "implementation" / "same_model_restore_smoke.json", result)
    return result


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

