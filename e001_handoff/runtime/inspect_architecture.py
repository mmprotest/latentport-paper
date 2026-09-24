"""Inspect both exact Qwen3.5 runtimes and write state schemas/correspondence."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5

from experiments.latentport.e001_handoff.runtime.constants import (
    IMPLEMENTATION_ROOT,
    SOURCE,
    TARGET,
)
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    release_model,
    seed_everything,
)
from experiments.latentport.e001_handoff.state.schema import schema_from_cache


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_digest(obj) -> dict[str, str]:
    path = Path(inspect.getsourcefile(obj)).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def inspect_one(spec) -> tuple[dict, dict]:
    config = load_text_config(spec)
    model = load_language_model(spec)
    cache = DynamicCache(config=config)
    probe_ids = torch.tensor([[760, 4434, 1528, 1902, 381, 16508, 6681, 13]], device="cuda:0")
    with torch.inference_mode():
        model(input_ids=probe_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    schema = schema_from_cache(
        cache,
        config,
        model_repository=spec.repository,
        model_revision=spec.revision,
    )
    architecture = {
        "role": spec.role,
        "repository": spec.repository,
        "revision": spec.revision,
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hidden_size": config.hidden_size,
        "language_layers": config.num_hidden_layers,
        "layer_types": list(config.layer_types),
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_value_head_dim": config.linear_value_head_dim,
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
        "attention_heads": config.num_attention_heads,
        "attention_kv_heads": config.num_key_value_heads,
        "attention_head_dim": config.head_dim,
        "rope_parameters": config.rope_parameters,
        "mamba_ssm_dtype": config.mamba_ssm_dtype,
        "cache_layer_classes": [type(layer).__name__ for layer in cache.layers],
    }
    release_model(model)
    return schema, architecture


def main() -> None:
    seed_everything(2026083101)
    schemas = {}
    architectures = {}
    for spec in (SOURCE, TARGET):
        schema, architecture = inspect_one(spec)
        schemas[spec.role] = schema
        architectures[spec.role] = architecture
        _write_json(IMPLEMENTATION_ROOT / f"state_schema_{spec.role}.json", schema)

    source = architectures["source"]
    target = architectures["target"]
    expected_pattern = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
    checks = {
        "language_layer_count": source["language_layers"] == target["language_layers"] == 32,
        "source_layer_pattern": source["layer_types"] == expected_pattern,
        "target_layer_pattern": target["layer_types"] == expected_pattern,
        "gdn_recurrent_geometry": all(
            key in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
            and source[key] == target[key]
            for key in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
        ),
        "gdn_convolution_geometry": (
            source["linear_num_key_heads"] == target["linear_num_key_heads"]
            and source["linear_num_value_heads"] == target["linear_num_value_heads"]
            and source["linear_key_head_dim"] == target["linear_key_head_dim"]
            and source["linear_value_head_dim"] == target["linear_value_head_dim"]
            and source["linear_conv_kernel_dim"] == target["linear_conv_kernel_dim"]
        ),
        "attention_kv_geometry": (
            source["attention_kv_heads"] == target["attention_kv_heads"]
            and source["attention_head_dim"] == target["attention_head_dim"]
        ),
        "rope_geometry": source["rope_parameters"] == target["rope_parameters"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"INCONCLUSIVE_ARCHITECTURE_MISMATCH: {checks}")

    correspondence = {
        "status": "PASS",
        "checks": checks,
        "source": source,
        "target": target,
        "canonical_correspondence": {
            "layer_mapping": "identity by zero-based language-layer index",
            "gdn_head_mapping": "identity by value-head index; repeated Q/K head semantics preserved",
            "attention_kv_head_mapping": "identity by KV-head index",
            "gdn_recurrent_shape": [1, 32, 128, 128],
            "gdn_convolution_shape": [1, 8192, 4],
            "attention_kv_shape_at_prefix_L": [1, 4, "L", 256],
            "ad_hoc_alignment": None,
        },
        "runtime_sources": {
            "qwen3_5_modeling": _source_digest(modeling_qwen3_5.Qwen3_5TextModel),
            "dynamic_cache": _source_digest(DynamicCache),
        },
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    _write_json(IMPLEMENTATION_ROOT / "architecture_correspondence.json", correspondence)
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
