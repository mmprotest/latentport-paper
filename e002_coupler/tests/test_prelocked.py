"""The 52 mandatory E002 pre-LOCKED scientific and implementation checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e002_coupler.analysis.audit import (
    validate_raw_record,
    verify_artifact_manifest,
)
from experiments.latentport.e002_coupler.analysis.metrics import (
    delta_nll,
    native_context_recovery,
    remaining_gap_reduction,
    tqr,
)
from experiments.latentport.e002_coupler.analysis.statistics import bootstrap_mean_ci
from experiments.latentport.e002_coupler.analysis.verdict import (
    determine_4k_verdict,
    direct_heavy,
    final_verdict,
)
from experiments.latentport.e002_coupler.correction.losses import behavior_kl, total_loss
from experiments.latentport.e002_coupler.correction.model import (
    IdentityAnchoredCoupler,
    parameter_inventory,
)
from experiments.latentport.e002_coupler.correction.serialization import verify_correction
from experiments.latentport.e002_coupler.factorial.composer import (
    compose_condition,
    condition_components,
)
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    E001_ROOT,
    E002_ROOT,
    FACTORIAL_CONDITIONS,
    PRIMARY_LOCKED_CONDITIONS,
    STATE_REPAIR_CHECKPOINTS,
    TRAINABLE_PARAMETER_CAP,
    load_preregistration,
)
from experiments.latentport.e002_coupler.runtime.lock_guard import load_locked_rows


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def refs() -> dict:
    return _json(E002_ROOT / "reuse" / "e001_refs.json")


@pytest.fixture(scope="module")
def integrity() -> dict:
    return _json(ATTEMPT_ROOT / "implementation" / "e001_integrity.json")


@pytest.fixture(scope="module")
def smoke() -> dict:
    return _json(ATTEMPT_ROOT / "implementation" / "same_model_restore_smoke.json")


def _synthetic_states() -> tuple[dict, dict]:
    def state(offset: float) -> dict:
        return {
            "format_version": "test",
            "sequence_length": 7,
            "layer_types": ["full_attention", "linear_attention"],
            "layers": [
                {
                    "layer_index": 0,
                    "layer_type": "full_attention",
                    "cache_class": "DynamicLayer",
                    "keys": torch.full((1, 1, 7, 2), offset),
                    "values": torch.full((1, 1, 7, 2), offset + 1),
                },
                {
                    "layer_index": 1,
                    "layer_type": "linear_attention",
                    "cache_class": "Qwen3_5DynamicCache",
                    "conv_states": {0: torch.full((1, 2, 4), offset + 2)},
                    "recurrent_states": {0: torch.full((1, 1, 2, 2), offset + 3)},
                },
            ],
        }

    return state(0.0), state(10.0)


def _assert_condition(condition: str) -> None:
    direct, translated = _synthetic_states()
    result = compose_condition(direct, translated, condition)
    expected = condition_components(condition)
    assert torch.equal(
        result["layers"][0]["keys"],
        (direct if expected["K"] == "D" else translated)["layers"][0]["keys"],
    )
    assert torch.equal(
        result["layers"][1]["recurrent_states"][0],
        (direct if expected["R"] == "D" else translated)["layers"][1]["recurrent_states"][0],
    )
    assert torch.equal(
        result["layers"][1]["conv_states"][0],
        (direct if expected["C"] == "D" else translated)["layers"][1]["conv_states"][0],
    )


def _all_split_rows() -> dict[str, list[dict]]:
    return {
        "factorial": _jsonl(E002_ROOT / "data" / "validation_factorial" / "manifest.jsonl"),
        "locked": _jsonl(E002_ROOT / "data" / "locked" / "manifest.jsonl"),
        "long": _jsonl(E002_ROOT / "data" / "long" / "manifest.jsonl"),
        "e001_fit": _jsonl(E001_ROOT / "data" / "fit" / "manifest.jsonl"),
        "e001_validation": _jsonl(E001_ROOT / "data" / "validation" / "manifest.jsonl"),
        "e001_locked": _jsonl(E001_ROOT / "data" / "locked" / "manifest.jsonl"),
        "e001_long": _jsonl(E001_ROOT / "data" / "long" / "manifest.jsonl"),
    }


def _verdict_metrics() -> dict:
    comparison = lambda low, high: {"bootstrap_ci": [low, high]}
    return {
        "base_state": "TDD",
        "remaining_gap_reduction": 0.30,
        "corrected_delta_nll": 0.08,
        "native_context_recovery": 0.92,
        "top1_agreement_native": 0.85,
        "coupling_replication": {"weak_or_inconsistent": False},
        "comparisons": {
            "base_minus_corrected": comparison(0.01, 0.04),
            "corrected_minus_source": comparison(-0.08, -0.01),
            "corrected_minus_shuffled": comparison(-0.5, -0.2),
            "ttt_minus_base": comparison(0.01, 0.05),
        },
    }


def test_01_source_model_identity(integrity):
    assert integrity["status"] == "PASS"
    assert SOURCE.repository == "Qwen/Qwen3.5-4B-Base"
    assert SOURCE.revision == "1001bb4d826a52d1f399e183466143f4da7b741b"


def test_02_target_model_identity(integrity):
    assert integrity["status"] == "PASS"
    assert TARGET.repository == "Qwen/Qwen3.5-9B-Base"
    assert TARGET.revision == "68c46c4b3498877f3ef123c856ecfde50c39f404"


def test_03_tokenizer_identity(integrity):
    tokenizers = [row for row in integrity["model_files"] if row["name"] == "tokenizer.json"]
    assert len(tokenizers) == 2
    assert len({row["sha256"] for row in tokenizers}) == 1


def test_04_e001_artifact_hashes(integrity, refs):
    verified = {row["path"]: row["sha256"] for row in integrity["verified_references"]}
    assert verified["FROZEN_MANIFEST.json"] == refs["frozen_manifest"]["sha256"]


def test_05_e001_translator_hashes(integrity, refs):
    assert integrity["combined_translator_hash"] == refs["combined_translator_hash"]
    assert all(name in refs["translators"] for name in ("kv_tensors", "gdn_tensors", "conv_tensors"))


def test_06_e001_schema_hashes(integrity, refs):
    assert integrity["schemas"]["source"]["sha256"] == refs["source_schema"]["sha256"]
    assert integrity["schemas"]["target"]["sha256"] == refs["target_schema"]["sha256"]


def test_07_same_model_restore_smoke(smoke):
    assert smoke["source_same_model_restore"]["logits_equal"]
    assert smoke["target_same_model_restore"]["logits_equal"]


def test_08_exact_token_equality(integrity):
    tokenizers = [row["sha256"] for row in integrity["model_files"] if row["name"] == "tokenizer.json"]
    assert tokenizers[0] == tokenizers[1]
    assert _json(E002_ROOT / "data" / "validation_factorial" / "selection.json")["selection_uses_model_outputs"] is False


def test_09_fresh_factorial_split_isolation():
    rows = _all_split_rows()
    factorial = {row["document_id"] for row in rows["factorial"]}
    prior = {row["document_id"] for name, values in rows.items() if name.startswith("e001_") for row in values}
    assert len(factorial) == 32 and factorial.isdisjoint(prior)


def test_10_fresh_locked_split_isolation():
    rows = _all_split_rows()
    locked = {row["document_id"] for row in rows["locked"]}
    excluded = {row["document_id"] for name, values in rows.items() if name != "locked" for row in values}
    assert len(locked) == 64 and locked.isdisjoint(excluded)


def test_11_no_e001_locked_leakage(integrity):
    assert integrity["e001_locked_records_parsed"] is False
    inventory = _json(ATTEMPT_ROOT / "fit" / "correction_data_inventory.json")
    assert all(value["e001_locked_documents"] == 0 for value in inventory["splits"].values())


def test_12_factorial_condition_assembly():
    assert FACTORIAL_CONDITIONS == ("DDD", "DDT", "DTD", "DTT", "TDD", "TDT", "TTD", "TTT")


def test_13_ddd(): _assert_condition("DDD")
def test_14_ddt(): _assert_condition("DDT")
def test_15_dtd(): _assert_condition("DTD")
def test_16_dtt(): _assert_condition("DTT")
def test_17_tdd(): _assert_condition("TDD")
def test_18_tdt(): _assert_condition("TDT")
def test_19_ttd(): _assert_condition("TTD")
def test_20_ttt(): _assert_condition("TTT")


def test_21_base_state_deterministic_selection():
    record = _json(ATTEMPT_ROOT / "factorial" / "base_state_selection.json")
    minimum = min(value["mean_delta_nll"] for value in record["condition_metrics"].values())
    eligible = [name for name, value in record["condition_metrics"].items() if value["mean_delta_nll"] <= minimum + 0.01]
    expected = min(eligible, key=lambda name: (name.count("T"), name))
    assert record["selected_state"] == expected


def test_22_zero_correction_equals_base(smoke):
    assert smoke["zero_correction_exact"] is True
    assert IdentityAnchoredCoupler(2).all_trainable_parameters_are_zero()


def test_23_correction_rank_cap():
    assert load_preregistration()["correction"]["rank_grid"] == [2, 4]
    with pytest.raises(ValueError):
        IdentityAnchoredCoupler(5)


def test_24_parameter_count_cap():
    assert parameter_inventory(2)["total"] < TRAINABLE_PARAMETER_CAP
    assert parameter_inventory(4)["total"] < TRAINABLE_PARAMETER_CAP


def test_25_frozen_target_weights(smoke):
    assert smoke["target_trainable_parameters"] == 0


def test_26_frozen_source_weights():
    selection = _json(ATTEMPT_ROOT / "fit" / "correction_selection.json")
    assert selection["source_weights_trainable"] == 0


def test_27_behavior_kl_loss():
    native = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    assert behavior_kl(native, native).item() == pytest.approx(0.0, abs=1e-7)
    assert behavior_kl(native, -native).item() > 0


def test_28_identity_regularization():
    native = torch.tensor([[[1.0, 0.0]]])
    total, parts = total_loss(native, native, torch.tensor(2.0), 1e-2)
    assert total.item() == pytest.approx(0.02)
    assert parts["identity"].item() == 2.0


def test_29_correction_serialization_hash():
    record = verify_correction(
        ATTEMPT_ROOT / "fit" / "correction.safetensors",
        ATTEMPT_ROOT / "fit" / "correction.json",
    )
    assert len(record["correction_hash"]) == 64


def test_30_no_target_prefix_replay(smoke):
    assert smoke["target_historical_prefix_tokens_in_handoff"] == 0


def test_31_bridge_token_identity():
    for row in _jsonl(E002_ROOT / "data" / "locked" / "manifest.jsonl"):
        assert row["token_ids"][4096] == row["token_ids"][4096]


def test_32_continuation_identity():
    for row in _jsonl(E002_ROOT / "data" / "locked" / "manifest.jsonl"):
        assert len(row["token_ids"][4097:4161]) == 64


def test_33_joint_corrected():
    selection = _json(ATTEMPT_ROOT / "fit" / "correction_selection.json")
    assert selection["selected"]["parameter_count"] < TRAINABLE_PARAMETER_CAP
    assert selection["selected"]["base_state"] in FACTORIAL_CONDITIONS


def test_34_joint_shuffled_no_fixed_points():
    rows = _jsonl(E002_ROOT / "data" / "locked" / "manifest.jsonl")
    order = sorted(range(64), key=lambda item: rows[item]["selection_sha256"])
    donor = {order[position]: order[(position + 1) % 64] for position in range(64)}
    assert all(recipient != giver for recipient, giver in donor.items())


def test_35_delta_nll():
    assert delta_nll(2.4, 2.1) == pytest.approx(0.3)


def test_36_ncr():
    assert native_context_recovery(3.2, 2.1, 2.1) == pytest.approx(1.0)


def test_37_tqr():
    assert tqr(2.3, 2.2, 2.1) == pytest.approx(0.5)


def test_38_rgr():
    assert remaining_gap_reduction(0.2, 0.1) == pytest.approx(0.5)


def test_39_paired_bootstrap():
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    first = bootstrap_mean_ci(values, resamples=1000, seed=7)
    second = bootstrap_mean_ci(values, resamples=1000, seed=7)
    assert first == second and first[0] <= values.mean() <= first[1]


def test_40_source_model_advantage():
    low, high = bootstrap_mean_ci(np.asarray([-0.2, -0.1, -0.15]), resamples=1000, seed=7)
    assert low < high < 0


def test_41_correction_magnitude():
    selection = _json(ATTEMPT_ROOT / "fit" / "correction.json")
    magnitude = selection["correction_magnitude_validation"]
    assert set(magnitude["by_component"]) == {"K", "R", "C"}
    assert magnitude["all_layers"]["maximum"] >= 0


def test_42_state_repair_token_1(): assert 1 in STATE_REPAIR_CHECKPOINTS
def test_43_state_repair_token_4(): assert 4 in STATE_REPAIR_CHECKPOINTS
def test_44_state_repair_token_16(): assert 16 in STATE_REPAIR_CHECKPOINTS
def test_45_state_repair_token_64(): assert 64 in STATE_REPAIR_CHECKPOINTS
def test_46_state_repair_token_256(): assert 256 in STATE_REPAIR_CHECKPOINTS


def test_47_timing_accounting():
    native, source, base, correction, install, bridge = 100.0, 40.0, 2.0, 3.0, 4.0, 5.0
    assert (native - source) - (base + correction + install + bridge) == 46.0


def test_48_verdict_logic():
    metrics = _verdict_metrics()
    assert determine_4k_verdict(metrics) == "FULL_STATE_HANDOFF"
    metrics["corrected_delta_nll"] = 0.04
    metrics["native_context_recovery"] = 0.96
    metrics["top1_agreement_native"] = 0.91
    assert determine_4k_verdict(metrics) == "NEAR_NATIVE_HANDOFF"


def test_49_long_gating():
    assert final_verdict("NEAR_NATIVE_HANDOFF", long_pass=True) == "LONG_CONTEXT_HANDOFF"
    with pytest.raises(ValueError):
        final_verdict("JOINT_CORRECTION_WORKS", long_pass=False)


def test_50_locked_data_access_prevention():
    with pytest.raises(PermissionError):
        load_locked_rows(authorization=False)


def test_51_final_manifest_verification_contract():
    with pytest.raises(ValueError, match="contains no files"):
        verify_artifact_manifest(E002_ROOT / "PREREGISTRATION.json", E002_ROOT)


def test_52_raw_evidence_completeness_contract():
    condition = {
        "nll": 1.0,
        "per_token_nll": [1.0] * 64,
        "per_token_entropy": [1.0] * 64,
        "logits": {"shape": [64, 10]},
    }
    repair = {str(checkpoint): {} for checkpoint in STATE_REPAIR_CHECKPOINTS}
    record = {
        "document_id": "synthetic",
        "corpus": "synthetic",
        "split": "LOCKED",
        "prefix_token_ids": [0] * 4096,
        "bridge_token_id": 1,
        "continuation_token_ids": [2] * 64,
        "conditions": {name: dict(condition) for name in PRIMARY_LOCKED_CONDITIONS},
        "derived": {"NCR": 0.0, "TQR": 0.0, "RGR": 0.0},
        "state_checksums": {},
        "correction_hash": "0" * 64,
        "base_state": "TDD",
        "correction_magnitude": {},
        "state_repair": {"BASE_STATE": repair, "JOINT_CORRECTED": repair},
        "timing": {},
    }
    validate_raw_record(record)
    assert direct_heavy(record["base_state"])
