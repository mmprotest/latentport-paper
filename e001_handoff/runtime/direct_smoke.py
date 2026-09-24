"""Run preregistered zero-training state-copy smoke conditions."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache

from experiments.latentport.e001_handoff.runtime.constants import (
    ATTEMPT_ROOT,
    SOURCE,
    TARGET,
    load_preregistration,
)
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    load_tokenizer,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import (
    compare_to_reference,
    json_safe_score,
    position_ids,
    score_from_cache,
)
from experiments.latentport.e001_handoff.runtime.validation_contexts import (
    build_restore_context,
    token_stream_sha256,
)
from experiments.latentport.e001_handoff.state.cache_state import (
    capture_cache,
    install_components,
    state_checksum,
    state_nbytes,
)


OUTPUT = ATTEMPT_ROOT / "diagnostics" / "direct_zero_training_smoke.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["direct_smoke"])
    prefix_length = prereg["direct_smoke"]["prefix_tokens"]
    canonical_tokenizer = load_tokenizer()
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET.snapshot, local_files_only=True)
    contexts = []
    for index in range(prereg["direct_smoke"]["documents"]):
        text_tokens = build_restore_context(canonical_tokenizer, 10_000 + index, prefix_length + 65)
        # Decode/re-encode is only an identity audit of the second tokenizer;
        # all actual model inputs use the single canonical token sequence.
        text = canonical_tokenizer.decode(text_tokens, clean_up_tokenization_spaces=False)
        target_tokens = target_tokenizer.encode(text, add_special_tokens=False)
        if target_tokens != text_tokens:
            raise RuntimeError("INCONCLUSIVE_TOKENIZATION_MISMATCH in direct smoke")
        contexts.append(
            {
                "document_id": f"DIRECT_SMOKE_{index:03d}",
                "token_ids": text_tokens,
                "token_stream_sha256": token_stream_sha256(text_tokens),
            }
        )

    source_config = load_text_config(SOURCE)
    source_model = load_language_model(SOURCE)
    device = next(source_model.parameters()).device
    for index, row in enumerate(contexts):
        ids = torch.tensor(row["token_ids"], dtype=torch.long, device=device).unsqueeze(0)
        cache = DynamicCache(config=source_config)
        with torch.inference_mode():
            source_model(
                input_ids=ids[:, :prefix_length],
                position_ids=position_ids(0, prefix_length, device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        captured = capture_cache(cache, source_config, device="cpu")
        row["source_state"] = captured
        row["source_state_bytes"] = state_nbytes(captured)
        row["source_state_sha256"] = state_checksum(captured)
        source_score = score_from_cache(
            source_model,
            cache,
            prefix_length=prefix_length,
            bridge_token_id=row["token_ids"][prefix_length],
            continuation_token_ids=row["token_ids"][prefix_length + 1 : prefix_length + 65],
        )
        row["SOURCE_4B"] = json_safe_score(source_score)
        print(f"source capture {index + 1}/{len(contexts)}", flush=True)
    del source_model
    release_model(None)

    target_config = load_text_config(TARGET)
    target_model = load_language_model(TARGET)
    target_device = next(target_model.parameters()).device
    conditions = {
        "DIRECT_GDN": (False, True, True),
        "DIRECT_KV": (True, False, False),
        "DIRECT_FULL": (True, True, True),
    }
    for index, row in enumerate(contexts):
        ids = torch.tensor(row["token_ids"], dtype=torch.long, device=target_device).unsqueeze(0)
        native_cache = DynamicCache(config=target_config)
        with torch.inference_mode():
            target_model(
                input_ids=ids[:, :prefix_length],
                position_ids=position_ids(0, prefix_length, target_device),
                past_key_values=native_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        native = score_from_cache(
            target_model,
            native_cache,
            prefix_length=prefix_length,
            bridge_token_id=row["token_ids"][prefix_length],
            continuation_token_ids=row["token_ids"][prefix_length + 1 : prefix_length + 65],
        )
        native["historical_prefix_tokens_processed_by_target"] = prefix_length
        row["NATIVE_9B"] = json_safe_score(native)
        for condition, (include_kv, include_recurrent, include_conv) in conditions.items():
            cache = install_components(
                row["source_state"],
                target_config,
                device=target_device,
                include_kv=include_kv,
                include_gdn_recurrent=include_recurrent,
                include_gdn_convolution=include_conv,
            )
            score = score_from_cache(
                target_model,
                cache,
                prefix_length=prefix_length,
                bridge_token_id=row["token_ids"][prefix_length],
                continuation_token_ids=row["token_ids"][prefix_length + 1 : prefix_length + 65],
            )
            score.update(compare_to_reference(score, native))
            row[condition] = json_safe_score(score)
        print(f"target direct smoke {index + 1}/{len(contexts)}", flush=True)

    serializable_rows = []
    for row in contexts:
        serializable_rows.append({key: value for key, value in row.items() if key not in ("source_state", "token_ids")})
    summary = {}
    for condition in ("SOURCE_4B", "NATIVE_9B", *conditions):
        values = [row[condition]["nll"] for row in serializable_rows]
        summary[condition] = {
            "mean_nll": sum(values) / len(values),
            "document_nll": values,
        }
        if condition in conditions:
            deltas = [row[condition]["delta_nll_to_native"] for row in serializable_rows]
            summary[condition]["mean_delta_nll_to_native"] = sum(deltas) / len(deltas)
    result = {
        "phase": "B_direct_zero_training_smoke",
        "canonical_verdict_eligible": False,
        "contexts": len(serializable_rows),
        "prefix_tokens": prefix_length,
        "conditions": list(conditions),
        "summary": summary,
        "raw_documents": serializable_rows,
    }
    _write_json(OUTPUT, result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
