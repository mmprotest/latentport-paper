from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.latentport.e001_handoff.analysis.integrity import (
    CANONICAL_CONDITIONS,
    translation_total_ms,
    validate_raw_evidence_record,
)
from experiments.latentport.e001_handoff.analysis.metrics import (
    delta_nll,
    paired_bootstrap_mean_ci,
    tqr,
)
from experiments.latentport.e001_handoff.analysis.verdict import NEXT_HYPOTHESES, choose_verdict
from experiments.latentport.e001_handoff.state.convergence import compare_convergence_states
from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT, SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.freeze_manifest import frozen_paths
from experiments.latentport.e001_handoff.runtime.lock_guard import (
    FROZEN_MANIFEST,
    load_locked_rows,
    sha256_file,
    verify_frozen_manifest,
)


def load_json(relative: str) -> dict:
    return json.loads((E001_ROOT / relative).read_text(encoding="utf-8"))


def base_stats() -> dict:
    return {
        "stable_finite_behavior": True,
        "full_vs_empty_bootstrap_ci": [0.1, 0.2],
        "kv_vs_empty_bootstrap_ci": [0.1, 0.2],
        "full_vs_kv_bootstrap_ci": [0.05, 0.1],
        "full_vs_kv_mean_improvement_fraction": 0.30,
        "full_vs_kv_median_improvement_fraction": 0.30,
        "condition_summary": {"FULL_TRANSLATED": {"mean_delta_nll": 0.05}},
        "aggregate_nll_tqr": 0.95,
        "full_vs_shuffled_bootstrap_ci": [0.05, 0.1],
    }


def test_source_model_identity():
    revisions = load_json("artifacts/attempt_001/implementation/model_revisions.json")
    assert revisions["source"]["repository"] == SOURCE.repository
    assert revisions["source"]["revision"] == SOURCE.revision
    assert len(revisions["source"]["weight_files"]) == 2


def test_target_model_identity():
    revisions = load_json("artifacts/attempt_001/implementation/model_revisions.json")
    assert revisions["target"]["repository"] == TARGET.repository
    assert revisions["target"]["revision"] == TARGET.revision
    assert len(revisions["target"]["weight_files"]) == 4


def test_tokenizer_identity():
    revisions = load_json("artifacts/attempt_001/implementation/model_revisions.json")
    assert revisions["source"]["tokenizer_json_sha256"] == revisions["target"]["tokenizer_json_sha256"]


def test_exact_token_equality_source_target():
    summary = load_json("data/MANIFEST_SUMMARY.json")
    assert summary["exact_token_equality_verified_for_all_selected_documents"] is True


def test_layer_type_pattern():
    expected = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
    for role in ("source", "target"):
        schema = load_json(f"artifacts/attempt_001/implementation/state_schema_{role}.json")
        assert schema["language_layers"] == 32
        assert schema["layer_type_pattern"] == expected
        assert schema["gated_deltanet_layers"] == 24
        assert schema["full_attention_layers"] == 8


def test_state_schema_completeness():
    required = {
        "field_name", "layer_index", "layer_type", "tensor_role", "shape", "dtype",
        "device", "byte_size", "semantic_interpretation", "restore_method",
    }
    for role in ("source", "target"):
        fields = load_json(f"artifacts/attempt_001/implementation/state_schema_{role}.json")["fields"]
        assert len(fields) == 217
        assert all(required <= set(field) for field in fields)


def test_same_model_4b_restore_gate():
    result = load_json("artifacts/attempt_001/state_validation/same_model_restore_4b.json")
    assert result["pass"] and result["top1_agreement"] == 1.0
    assert result["minimum_next_token_logit_cosine"] >= 0.999999
    assert result["maximum_absolute_logit_difference"] == 0.0


def test_same_model_9b_restore_gate():
    result = load_json("artifacts/attempt_001/state_validation/same_model_restore_9b.json")
    assert result["pass"] and result["top1_agreement"] == 1.0
    assert result["minimum_next_token_logit_cosine"] >= 0.999999
    assert result["maximum_absolute_logit_difference"] == 0.0


def test_kv_layer_head_correspondence():
    architecture = load_json("artifacts/attempt_001/implementation/architecture_correspondence.json")
    assert architecture["checks"]["attention_kv_geometry"] is True
    assert architecture["canonical_correspondence"]["attention_kv_shape_at_prefix_L"] == [1, 4, "L", 256]


def test_gdn_layer_head_correspondence():
    architecture = load_json("artifacts/attempt_001/implementation/architecture_correspondence.json")
    assert architecture["checks"]["gdn_recurrent_geometry"] is True
    assert architecture["canonical_correspondence"]["gdn_recurrent_shape"] == [1, 32, 128, 128]


def test_convolution_state_correspondence():
    architecture = load_json("artifacts/attempt_001/implementation/architecture_correspondence.json")
    assert architecture["checks"]["gdn_convolution_geometry"] is True
    assert architecture["canonical_correspondence"]["gdn_convolution_shape"] == [1, 8192, 4]


def test_translator_inventory_is_low_capacity_only():
    recurrent = load_json("translators/frozen/gdn_recurrent_translator.json")
    kv = load_json("translators/frozen/kv_translator.json")
    conv = load_json("translators/frozen/gdn_convolution_translator.json")
    assert recurrent["nonlinear_components"] is None
    assert recurrent["formula"] == "S9_hat = mu9 + A (S4 - mu4) B^T"
    assert "exactly three" in recurrent["fit_algorithm"]
    assert kv["component"] == "full_attention_kv"
    assert conv["qkv_mixing"] is False


def test_translator_hash_stability():
    for name in ("kv_translator", "gdn_convolution_translator", "gdn_recurrent_translator"):
        metadata = load_json(f"translators/frozen/{name}.json")
        assert sha256_file(E001_ROOT / metadata["tensor_file"]) == metadata["tensor_file_sha256"]


def test_paired_state_fit_inventory():
    for role in ("source", "target"):
        record = load_json(f"artifacts/attempt_001/fit/paired_states/{role}_collection.json")
        assert record["documents"] == 128 and record["checkpoints"] == 1024
        assert record["checkpoint_capture_mode"].startswith("independent fresh-cache")


def test_paired_state_validation_inventory():
    for role in ("source", "target"):
        record = load_json(f"artifacts/attempt_001/validation/paired_states/{role}_collection.json")
        assert record["documents"] == 32 and record["checkpoints"] == 256


def test_shuffled_state_no_fixed_point_rule():
    order = list(range(64))
    donor = {order[position]: order[(position + 1) % len(order)] for position in range(len(order))}
    assert all(recipient != source for recipient, source in donor.items())
    assert sorted(donor.values()) == order


def test_teacher_forced_continuation_identity_contract():
    source = (E001_ROOT / "runtime/locked_run.py").read_text(encoding="utf-8")
    assert "continuation = token_ids[4097:4161]" in source
    assert "continuation_token_ids=continuation" in source
    assert "bridge_token_id=bridge" in source


def test_no_target_historical_prefill_contract():
    source = (E001_ROOT / "runtime/locked_run.py").read_text(encoding="utf-8")
    handoff_block = source[source.index("condition_specs ="):source.index("if convergence is None")]
    assert "native_inputs" not in handoff_block
    assert '"no_target_historical_prefill"' in handoff_block


def test_delta_nll_calculation():
    assert delta_nll(2.5, 1.25) == 1.25


def test_tqr_calculation_and_exclusion():
    assert tqr(2.0, 1.25, 1.0) == 0.75
    assert tqr(1.0, 1.2, 1.0) is None


def test_bootstrap_reproducibility():
    values = __import__("numpy").array([0.1, 0.2, 0.3, 0.4])
    assert paired_bootstrap_mean_ci(values, resamples=1000, seed=7) == paired_bootstrap_mean_ci(
        values, resamples=1000, seed=7
    )


def test_timing_accounting():
    timing = {
        "host_device_copy_ms": 2.0,
        "kv_translation_ms": 3.0,
        "gdn_recurrent_translation_ms": 5.0,
        "gdn_convolution_translation_ms": 7.0,
        "translation_compute_ms": 15.0,
    }
    assert translation_total_ms(timing) == 17.0


def test_verdict_no_translatable_state():
    stats = base_stats(); stats["full_vs_empty_bootstrap_ci"] = [-0.1, 0.1]
    assert choose_verdict(stats) == "NO_TRANSLATABLE_STATE"


def test_verdict_kv_only_transfer():
    stats = base_stats(); stats["full_vs_kv_bootstrap_ci"] = [-0.01, 0.01]
    assert choose_verdict(stats) == "KV_ONLY_TRANSFER"


def test_verdict_recurrent_state_translatable():
    stats = base_stats(); stats["condition_summary"]["FULL_TRANSLATED"]["mean_delta_nll"] = 0.25
    assert choose_verdict(stats) == "RECURRENT_STATE_TRANSLATABLE"


def test_verdict_full_state_handoff():
    assert choose_verdict(base_stats()) == "FULL_STATE_HANDOFF"


def test_verdict_strong_requires_long_pass():
    stats = base_stats()
    assert choose_verdict(stats, long_run=True, long_pass=True) == "STRONG_FULL_STATE_HANDOFF"
    assert choose_verdict(stats, long_run=False, long_pass=False) == "FULL_STATE_HANDOFF"


def test_exactly_one_next_hypothesis_per_verdict():
    assert set(NEXT_HYPOTHESES) == {
        "NO_TRANSLATABLE_STATE", "KV_ONLY_TRANSFER", "RECURRENT_STATE_TRANSLATABLE",
        "FULL_STATE_HANDOFF", "STRONG_FULL_STATE_HANDOFF",
    }
    assert all(text.startswith("H2: ") for text in NEXT_HYPOTHESES.values())


def test_locked_data_access_prevention():
    with pytest.raises(PermissionError):
        load_locked_rows()


def test_freeze_inventory_has_required_inputs():
    relative = {path.relative_to(E001_ROOT).as_posix() for path in frozen_paths(require_all=False)}
    assert "PREREGISTRATION.json" in relative
    assert "data/locked/manifest.jsonl" in relative
    assert "runtime/locked_run.py" in relative
    assert "translators/frozen/gdn_recurrent_translator.safetensors" in relative


def test_manifest_verification_contract():
    if FROZEN_MANIFEST.exists():
        assert verify_frozen_manifest()["freeze_state"] == "FROZEN_BEFORE_LOCKED"
    else:
        source = (E001_ROOT / "runtime/lock_guard.py").read_text(encoding="utf-8")
        assert "sha256_file(path) != record" in source


def test_state_convergence_metrics():
    import torch

    native = {
        "new_tokens": 1,
        "layers": [
            {
                "layer_index": 0,
                "layer_type": "linear_attention",
                "recurrent": torch.ones(1, 2, 2, 2),
                "convolution": torch.ones(1, 4, 2),
            },
            {
                "layer_index": 1,
                "layer_type": "full_attention",
                "new_keys": torch.ones(1, 1, 1, 2),
                "new_values": torch.ones(1, 1, 1, 2),
            },
        ],
    }
    candidate = {
        "new_tokens": 1,
        "layers": [
            {
                "layer_index": 0,
                "layer_type": "linear_attention",
                "recurrent": torch.ones(1, 2, 2, 2),
                "convolution": torch.ones(1, 4, 2),
            },
            {
                "layer_index": 1,
                "layer_type": "full_attention",
                "new_keys": torch.ones(1, 1, 1, 2),
                "new_values": torch.ones(1, 1, 1, 2),
            },
        ],
    }
    result = compare_convergence_states(native, candidate)
    assert result["mean_gdn_recurrent_normalized_frobenius"] == 0.0
    assert result["mean_new_key_normalized_l2"] == 0.0


def test_raw_evidence_completeness_contract():
    condition = {
        "nll": 1.0,
        "per_token_nll": [1.0] * 64,
        "historical_prefix_tokens_processed_in_scoring_branch": 0,
        "continuation_schedule": [1, 4, 16, 64],
    }
    record = {
        "document_id": "d", "corpus": "c", "split": "LOCKED", "prefix_token_count": 4096,
        "bridge_token_id": 1, "continuation_token_ids": list(range(64)),
        "conditions": {name: dict(condition) for name in CANONICAL_CONDITIONS},
        "state_checksums": {}, "state_bytes": {},
        "translator_hash": "0" * 64,
        "shuffle": {"fixed_point": False},
        "timing": {
            "host_device_copy_ms": 1.0, "kv_translation_ms": 1.0,
            "gdn_recurrent_translation_ms": 1.0, "gdn_convolution_translation_ms": 1.0,
            "translation_compute_ms": 3.0,
        },
        "state_convergence": {str(key): {} for key in (1, 4, 16, 64)},
        "token_accounting": {},
    }
    assert validate_raw_evidence_record(record) == []
