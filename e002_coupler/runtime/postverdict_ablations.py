"""Required component and recurrent-depth ablations after the 4K verdict freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.latentport.e001_handoff.runtime.constants import TARGET
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.runtime.scoring import json_safe_score
from experiments.latentport.e002_coupler.analysis.statistics import bootstrap_mean_ci
from experiments.latentport.e002_coupler.correction.serialization import load_correction
from experiments.latentport.e002_coupler.factorial.composer import compose_condition
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, load_preregistration
from experiments.latentport.e002_coupler.runtime.install import install_for_inference
from experiments.latentport.e002_coupler.runtime.lock_guard import load_locked_rows, verify_frozen_manifest
from experiments.latentport.e002_coupler.runtime.locked_run import (
    CORRECTION_RAW,
    SOURCE_RAW,
    TRANSLATION_RAW,
    _condition_tokens,
    _load_jsonl,
)
from experiments.latentport.e002_coupler.runtime.scoring import score_primary
from experiments.latentport.e002_coupler.runtime.state_io import load_state, write_json_once


RAW = ATTEMPT_ROOT / "diagnostics" / "postverdict_ablation_raw.jsonl"
SUMMARY = ATTEMPT_ROOT / "diagnostics" / "postverdict_ablations.json"


ABLATIONS = {
    "KV_CORRECTION_REMOVED": {"disabled_components": ("K",), "disabled_gdn_positions": ()},
    "RECURRENT_CORRECTION_REMOVED": {"disabled_components": ("R",), "disabled_gdn_positions": ()},
    "CONVOLUTION_CORRECTION_REMOVED": {"disabled_components": ("C",), "disabled_gdn_positions": ()},
    "EARLY_THIRD_RECURRENT_REMOVED": {"disabled_components": (), "disabled_gdn_positions": tuple(range(0, 8))},
    "MIDDLE_THIRD_RECURRENT_REMOVED": {"disabled_components": (), "disabled_gdn_positions": tuple(range(8, 16))},
    "LATE_THIRD_RECURRENT_REMOVED": {"disabled_components": (), "disabled_gdn_positions": tuple(range(16, 24))},
}


def _append(value: object) -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    if RAW.exists() or SUMMARY.exists():
        raise FileExistsError("post-verdict ablation evidence already exists")
    verdict = json.loads(
        (ATTEMPT_ROOT / "verdict" / "final_verdict.json").read_text(encoding="utf-8")
    )
    if verdict["freeze_state"] != "FINAL_CANONICAL_VERDICT_FROZEN_BEFORE_POSTVERDICT_ABLATIONS":
        raise PermissionError("ablations are forbidden before final canonical verdict freeze")
    frozen = verify_frozen_manifest()
    rows = load_locked_rows(authorization=True)
    source_rows = _load_jsonl(SOURCE_RAW)
    translated_rows = _load_jsonl(TRANSLATION_RAW)
    locked_raw = _load_jsonl(ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl")
    coupler = load_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
        device="cuda:0",
    )
    seed_everything(load_preregistration()["seeds"]["global"])
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    for index, row in enumerate(rows):
        _, bridge, continuation, _ = _condition_tokens(row)
        source = load_state(ATTEMPT_ROOT / source_rows[index]["state_file"])
        translated = load_state(ATTEMPT_ROOT / translated_rows[index]["state_file"])
        base = compose_condition(source, translated, frozen["base_state"])
        native_nll = locked_raw[index]["conditions"]["NATIVE_9B"]["nll"]
        corrected_nll = locked_raw[index]["conditions"]["JOINT_CORRECTED"]["nll"]
        outcomes = {}
        for name, spec in ABLATIONS.items():
            application = coupler.apply_to_state(base, **spec)
            cache = install_for_inference(application.state, config)
            score = score_primary(
                model,
                cache,
                prefix_length=4096,
                bridge_token_id=bridge,
                continuation_token_ids=continuation,
                retain_logits=False,
            )
            outcomes[name] = {
                **json_safe_score(score),
                "delta_nll_to_native": score["nll"] - native_nll,
                "impact_vs_joint_corrected": score["nll"] - corrected_nll,
            }
            del application, cache, score
        _append(
            {
                "document_index": index,
                "document_id": row["document_id"],
                "canonical_verdict_frozen": verdict["canonical_verdict"],
                "outcomes": outcomes,
            }
        )
        print(f"E002 post-verdict ablations {index + 1}/64", flush=True)
        del source, translated, base
        torch.cuda.empty_cache()
    release_model(model)
    del coupler
    evidence = _load_jsonl(RAW)
    summary: dict[str, Any] = {
        "canonical_verdict_unchanged": verdict["canonical_verdict"],
        "used_for_e002_selection": False,
        "documents": 64,
        "ablations": {},
    }
    for name in ABLATIONS:
        impacts = np.asarray(
            [record["outcomes"][name]["impact_vs_joint_corrected"] for record in evidence],
            dtype=np.float64,
        )
        deltas = np.asarray(
            [record["outcomes"][name]["delta_nll_to_native"] for record in evidence],
            dtype=np.float64,
        )
        summary["ablations"][name] = {
            "mean_delta_nll": float(np.mean(deltas)),
            "mean_impact_vs_joint_corrected": float(np.mean(impacts)),
            "median_impact_vs_joint_corrected": float(np.median(impacts)),
            "impact_bootstrap_ci": list(bootstrap_mean_ci(impacts)),
        }
    write_json_once(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
