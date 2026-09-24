"""Verify 128-token checkpoint collection matches a one-shot 1024 prefill."""

from __future__ import annotations

import json

import torch
from transformers import DynamicCache

from experiments.latentport.e001_handoff.analysis.metrics import compare_logits
from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT, SOURCE
from experiments.latentport.e001_handoff.runtime.modeling import load_language_model, load_text_config
from experiments.latentport.e001_handoff.runtime.scoring import position_ids
from experiments.latentport.e001_handoff.state.cache_state import capture_cache, state_checksum


def main() -> None:
    row = json.loads((E001_ROOT / "data" / "fit" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ids = torch.tensor(row["token_ids"][:1025], dtype=torch.long, device="cuda:0").unsqueeze(0)
    config = load_text_config(SOURCE)
    model = load_language_model(SOURCE)
    one_shot = DynamicCache(config=config)
    chunked = DynamicCache(config=config)
    with torch.inference_mode():
        model(
            input_ids=ids[:, :1024],
            position_ids=position_ids(0, 1024, ids.device),
            past_key_values=one_shot,
            use_cache=True,
            logits_to_keep=1,
        )
        for start in range(0, 1024, 128):
            model(
                input_ids=ids[:, start : start + 128],
                position_ids=position_ids(start, 128, ids.device),
                past_key_values=chunked,
                use_cache=True,
                logits_to_keep=1,
            )
    one_hash = state_checksum(capture_cache(one_shot, config))
    chunk_hash = state_checksum(capture_cache(chunked, config))
    with torch.inference_mode():
        one_logits = model(
            input_ids=ids[:, 1024:1025],
            position_ids=position_ids(1024, 1, ids.device),
            past_key_values=one_shot,
            use_cache=True,
            logits_to_keep=1,
        ).logits[0, -1]
        chunk_logits = model(
            input_ids=ids[:, 1024:1025],
            position_ids=position_ids(1024, 1, ids.device),
            past_key_values=chunked,
            use_cache=True,
            logits_to_keep=1,
        ).logits[0, -1]
    comparison = compare_logits(one_logits, chunk_logits)
    result = {
        "document_id": row["document_id"],
        "one_shot_state_sha256": one_hash,
        "chunked_state_sha256": chunk_hash,
        "state_bit_identical": one_hash == chunk_hash,
        "bridge_logit_comparison": comparison,
        "pass": one_hash == chunk_hash and comparison["max_abs_diff"] == 0.0,
    }
    output = ATTEMPT_ROOT / "implementation" / "checkpoint_chunk_equivalence.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
