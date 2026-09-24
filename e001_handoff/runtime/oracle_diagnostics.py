"""Run exactly the two preregistered oracle diagnostics after their trigger fires."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, TARGET
from experiments.latentport.e001_handoff.runtime.modeling import load_language_model, load_text_config, release_model
from experiments.latentport.e001_handoff.runtime.scoring import (
    compare_full_distributions,
    json_safe_score,
    score_from_cache_segmented,
)
from experiments.latentport.e001_handoff.state.cache_state import compose_component_state, install_components


VERDICT = ATTEMPT_ROOT / "verdict" / "CANONICAL_4K_RESULT.json"
LOCKED_RAW = ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl"
OUTPUT = ATTEMPT_ROOT / "diagnostics" / "oracle_raw_evidence.jsonl"
MARKER = ATTEMPT_ROOT / "diagnostics" / "ORACLE_RUN_STARTED.json"


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append(value: object) -> None:
    with OUTPUT.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    result = json.loads(VERDICT.read_text(encoding="utf-8"))
    if not result["oracle_diagnostics_unlocked"]:
        raise PermissionError("Oracle diagnostics were not triggered by the frozen 4K result")
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    with MARKER.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"trigger_result_sha256": result["raw_evidence_sha256"]}) + "\n")
    raw_rows = _rows(LOCKED_RAW)
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    device = next(model.parameters()).device
    for index, row in enumerate(raw_rows):
        native_state = torch.load(
            ATTEMPT_ROOT / row["native_state_file"]["path"], map_location="cpu", weights_only=False
        )
        translated_state = torch.load(
            ATTEMPT_ROOT / "locked" / "translated_states" / f"translated_{index:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        native_cache = install_components(
            native_state, config, device=device, include_kv=True,
            include_gdn_recurrent=True, include_gdn_convolution=True,
        )
        native_score = score_from_cache_segmented(
            model, native_cache, prefix_length=4096, bridge_token_id=row["bridge_token_id"],
            continuation_token_ids=row["continuation_token_ids"], retain_logits=True,
        )
        specs = {
            "ORACLE_KV_TRANSLATED_GDN": compose_component_state(
                kv_state=native_state, gdn_state=translated_state
            ),
            "TRANSLATED_KV_ORACLE_GDN": compose_component_state(
                kv_state=translated_state, gdn_state=native_state
            ),
        }
        conditions = {}
        for name, state in specs.items():
            cache = install_components(
                state, config, device=device, include_kv=True,
                include_gdn_recurrent=True, include_gdn_convolution=True,
            )
            score = score_from_cache_segmented(
                model, cache, prefix_length=4096, bridge_token_id=row["bridge_token_id"],
                continuation_token_ids=row["continuation_token_ids"], retain_logits=True,
            )
            score.update(compare_full_distributions(score, native_score))
            score["verdict_eligible"] = False
            conditions[name] = json_safe_score(score)
            del cache, score
        _append(
            {
                "document_index": index,
                "document_id": row["document_id"],
                "bridge_token_id": row["bridge_token_id"],
                "continuation_token_ids": row["continuation_token_ids"],
                "conditions": conditions,
            }
        )
        print(f"ORACLE diagnostic {index + 1}/64", flush=True)
        del native_state, translated_state, native_cache, native_score
        torch.cuda.empty_cache()
    release_model(model)


if __name__ == "__main__":
    main()
