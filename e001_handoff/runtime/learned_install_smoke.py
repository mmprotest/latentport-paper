"""Implementation-only smoke for frozen learned translation and target install."""

from __future__ import annotations

import json
import math
import time

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    load_tokenizer,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import (
    json_safe_score,
    position_ids,
    score_from_cache_segmented,
)
from experiments.latentport.e001_handoff.runtime.validation_contexts import build_restore_context
from experiments.latentport.e001_handoff.state.cache_state import (
    capture_cache,
    compose_component_state,
    install_components,
    state_checksum,
    state_nbytes,
)
from experiments.latentport.e001_handoff.translators.runtime_translation import (
    FrozenTranslators,
    translate_full_state,
)


def main() -> None:
    seed_everything(2026083106)
    prefix_length = 1024
    tokenizer = load_tokenizer()
    token_ids = build_restore_context(tokenizer, 20_000, prefix_length + 65)
    source_config = load_text_config(SOURCE)
    source_model = load_language_model(SOURCE)
    source_device = next(source_model.parameters()).device
    source_cache = DynamicCache(config=source_config)
    inputs = torch.tensor(token_ids[:prefix_length], dtype=torch.long, device=source_device).unsqueeze(0)
    with torch.inference_mode():
        source_model(
            input_ids=inputs,
            position_ids=position_ids(0, prefix_length, source_device),
            past_key_values=source_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    source_state = capture_cache(source_cache, source_config, device="cpu")
    del source_model, source_cache
    release_model(None)

    translators = FrozenTranslators()
    translated_state, translation_timing = translate_full_state(source_state, translators)
    translated_checksum = state_checksum(translated_state)
    if not all(torch.isfinite(tensor).all() for layer in translated_state["layers"] for tensor in (
        ([layer["keys"], layer["values"]] if layer["layer_type"] == "full_attention" else [layer["conv_states"][0], layer["recurrent_states"][0]])
    )):
        raise RuntimeError("Translated smoke state contains NaN/Inf")

    target_config = load_text_config(TARGET)
    target_model = load_language_model(TARGET)
    target_device = next(target_model.parameters()).device
    bridge_token = token_ids[prefix_length]
    continuation = token_ids[prefix_length + 1 : prefix_length + 65]
    conditions = {}

    native_cache = DynamicCache(config=target_config)
    native_inputs = torch.tensor(token_ids[:prefix_length], dtype=torch.long, device=target_device).unsqueeze(0)
    with torch.inference_mode():
        target_model(
            input_ids=native_inputs,
            position_ids=position_ids(0, prefix_length, target_device),
            past_key_values=native_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    native = score_from_cache_segmented(
        target_model,
        native_cache,
        prefix_length=prefix_length,
        bridge_token_id=bridge_token,
        continuation_token_ids=continuation,
    )
    native["historical_prefix_tokens_processed_in_scoring_branch"] = prefix_length
    conditions["NATIVE_9B"] = json_safe_score(native)

    state_sources = {
        "EMPTY_9B": None,
        "KV_ONLY": (translated_state, True, False, False),
        "KV_GDN_DIRECT": (
            compose_component_state(kv_state=translated_state, gdn_state=source_state),
            True,
            True,
            True,
        ),
        "FULL_TRANSLATED": (translated_state, True, True, True),
    }
    for name, specification in state_sources.items():
        install_start = time.perf_counter()
        if specification is None:
            cache = DynamicCache(config=target_config)
        else:
            state, include_kv, include_recurrent, include_conv = specification
            cache = install_components(
                state,
                target_config,
                device=target_device,
                include_kv=include_kv,
                include_gdn_recurrent=include_recurrent,
                include_gdn_convolution=include_conv,
            )
        torch.cuda.synchronize()
        install_ms = (time.perf_counter() - install_start) * 1000.0
        score = score_from_cache_segmented(
            target_model,
            cache,
            prefix_length=prefix_length,
            bridge_token_id=bridge_token,
            continuation_token_ids=continuation,
        )
        if not math.isfinite(score["nll"]):
            raise RuntimeError(f"{name} learned-install smoke produced non-finite NLL")
        score["state_install_ms"] = install_ms
        score["delta_nll_to_native"] = score["nll"] - native["nll"]
        conditions[name] = json_safe_score(score)
        del cache
        torch.cuda.empty_cache()

    result = {
        "phase": "learned_translation_install_smoke",
        "canonical_verdict_eligible": False,
        "prefix_tokens": prefix_length,
        "source_state_bytes": state_nbytes(source_state),
        "translated_state_bytes": state_nbytes(translated_state),
        "translated_state_sha256": translated_checksum,
        "translation_timing": translation_timing,
        "conditions": conditions,
        "pass": True,
    }
    output = ATTEMPT_ROOT / "implementation" / "learned_translation_install_smoke.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
