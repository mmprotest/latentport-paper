"""Apply the three frozen translator families to a captured source cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.latentport.e001_handoff.runtime.constants import E001_ROOT
from experiments.latentport.e001_handoff.state.cache_state import STATE_FORMAT_VERSION
from experiments.latentport.e001_handoff.translators.rope import derotate_cached_key, rerotate_key


TRANSLATOR_ROOT = E001_ROOT / "translators" / "frozen"


class FrozenTranslators:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.kv = load_file(TRANSLATOR_ROOT / "kv_translator.safetensors", device=str(self.device))
        self.conv = load_file(
            TRANSLATOR_ROOT / "gdn_convolution_translator.safetensors", device=str(self.device)
        )
        self.recurrent = load_file(
            TRANSLATOR_ROOT / "gdn_recurrent_translator.safetensors", device=str(self.device)
        )

    def inventory(self) -> dict[str, Any]:
        result = {}
        for name in ("kv_translator", "gdn_convolution_translator", "gdn_recurrent_translator"):
            result[name] = json.loads((TRANSLATOR_ROOT / f"{name}.json").read_text(encoding="utf-8"))
        return result


def _apply_linear(
    source: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    prefix: str,
    index: int,
) -> torch.Tensor:
    source_mean = tensors[f"{prefix}.source_mean"][index]
    source_scale = tensors[f"{prefix}.source_scale"][index]
    target_mean = tensors[f"{prefix}.target_mean"][index]
    target_scale = tensors[f"{prefix}.target_scale"][index]
    weight = tensors[f"{prefix}.weight"][index]
    normalized = (source - source_mean[:, None, :]) / source_scale[:, None, :]
    return torch.bmm(normalized, weight) * target_scale[:, None, :] + target_mean[:, None, :]


def _conv_groups(source: torch.Tensor) -> torch.Tensor:
    q, k, v = source.split((2048, 2048, 4096), dim=0)
    groups = [q.reshape(16, 128, 4), k.reshape(16, 128, 4), v.reshape(32, 128, 4)]
    return torch.cat(groups, dim=0).permute(0, 2, 1).contiguous()


def _pack_conv_groups(groups: torch.Tensor) -> torch.Tensor:
    q, k, v = groups.split((16, 16, 32), dim=0)
    packed = [
        q.permute(0, 2, 1).reshape(2048, 4),
        k.permute(0, 2, 1).reshape(2048, 4),
        v.permute(0, 2, 1).reshape(4096, 4),
    ]
    return torch.cat(packed, dim=0)


@torch.inference_mode()
def translate_full_state(
    source_state: dict[str, Any],
    translators: FrozenTranslators,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    device = translators.device
    translated_layers = []
    attention_position = 0
    gdn_position = 0
    host_device_ms = 0.0
    kv_ms = 0.0
    recurrent_ms = 0.0
    convolution_ms = 0.0
    bytes_host_to_device = 0
    bytes_device_to_host = 0
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()
    for record in source_state["layers"]:
        if record["layer_type"] == "full_attention":
            copy_start = time.perf_counter()
            source_keys = record["keys"].to(device=device, dtype=torch.float32)
            source_values = record["values"].to(device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            host_device_ms += (time.perf_counter() - copy_start) * 1000.0
            bytes_host_to_device += record["keys"].numel() * record["keys"].element_size()
            bytes_host_to_device += record["values"].numel() * record["values"].element_size()
            positions = torch.arange(source_keys.shape[-2], device=device, dtype=torch.long)
            transform_start = time.perf_counter()
            key_heads = torch.stack(
                [derotate_cached_key(source_keys[0, head], positions) for head in range(4)], dim=0
            )
            key_heads = _apply_linear(key_heads, translators.kv, "k", attention_position)
            translated_keys = torch.stack(
                [rerotate_key(key_heads[head], positions) for head in range(4)], dim=0
            ).to(torch.bfloat16)
            translated_values = _apply_linear(
                source_values[0], translators.kv, "v", attention_position
            ).to(torch.bfloat16)
            torch.cuda.synchronize()
            kv_ms += (time.perf_counter() - transform_start) * 1000.0
            copy_start = time.perf_counter()
            translated_keys_cpu = translated_keys.unsqueeze(0).cpu()
            translated_values_cpu = translated_values.unsqueeze(0).cpu()
            torch.cuda.synchronize()
            host_device_ms += (time.perf_counter() - copy_start) * 1000.0
            bytes_device_to_host += translated_keys_cpu.numel() * translated_keys_cpu.element_size()
            bytes_device_to_host += translated_values_cpu.numel() * translated_values_cpu.element_size()
            translated_layers.append(
                {
                    "layer_index": record["layer_index"],
                    "layer_type": record["layer_type"],
                    "cache_class": record["cache_class"],
                    "keys": translated_keys_cpu,
                    "values": translated_values_cpu,
                    "is_initialized": True,
                }
            )
            attention_position += 1
        else:
            copy_start = time.perf_counter()
            source_recurrent = record["recurrent_states"][0][0].to(device=device, dtype=torch.float32)
            source_conv = record["conv_states"][0][0].to(device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            host_device_ms += (time.perf_counter() - copy_start) * 1000.0
            bytes_host_to_device += record["recurrent_states"][0].numel() * record[
                "recurrent_states"
            ][0].element_size()
            bytes_host_to_device += record["conv_states"][0].numel() * record["conv_states"][
                0
            ].element_size()

            transform_start = time.perf_counter()
            centered = source_recurrent - translators.recurrent["source_mean"][gdn_position]
            translated_recurrent = translators.recurrent["target_mean"][gdn_position] + torch.matmul(
                torch.matmul(translators.recurrent["key_map"][gdn_position], centered),
                translators.recurrent["value_map"][gdn_position].transpose(-1, -2),
            )
            torch.cuda.synchronize()
            recurrent_ms += (time.perf_counter() - transform_start) * 1000.0

            transform_start = time.perf_counter()
            conv_groups = _conv_groups(source_conv)
            translated_groups = _apply_linear(conv_groups, translators.conv, "conv", gdn_position)
            translated_conv = _pack_conv_groups(translated_groups).to(torch.bfloat16)
            torch.cuda.synchronize()
            convolution_ms += (time.perf_counter() - transform_start) * 1000.0

            copy_start = time.perf_counter()
            recurrent_cpu = translated_recurrent.unsqueeze(0).cpu()
            conv_cpu = translated_conv.unsqueeze(0).cpu()
            torch.cuda.synchronize()
            host_device_ms += (time.perf_counter() - copy_start) * 1000.0
            bytes_device_to_host += recurrent_cpu.numel() * recurrent_cpu.element_size()
            bytes_device_to_host += conv_cpu.numel() * conv_cpu.element_size()
            translated_layers.append(
                {
                    "layer_index": record["layer_index"],
                    "layer_type": record["layer_type"],
                    "cache_class": record["cache_class"],
                    "number_of_states": 1,
                    "conv_states": {0: conv_cpu},
                    "recurrent_states": {0: recurrent_cpu},
                    "is_conv_states_initialized": {0: True},
                    "is_recurrent_states_initialized": {0: True},
                    "has_previous_state": {0: True},
                    "conv_kernel_size": {0: 4},
                    "record_past": False,
                }
            )
            gdn_position += 1
    if attention_position != 8 or gdn_position != 24:
        raise RuntimeError("Translated state did not cover all corresponding layers")
    translated = {
        "format_version": STATE_FORMAT_VERSION,
        "num_hidden_layers": 32,
        "layer_types": list(source_state["layer_types"]),
        "sequence_length": int(source_state["sequence_length"]),
        "layers": translated_layers,
    }
    timing = {
        "host_device_copy_ms": host_device_ms,
        "kv_translation_ms": kv_ms,
        "gdn_recurrent_translation_ms": recurrent_ms,
        "gdn_convolution_translation_ms": convolution_ms,
        "translation_compute_ms": kv_ms + recurrent_ms + convolution_ms,
        "bytes_host_to_device": bytes_host_to_device,
        "bytes_device_to_host": bytes_device_to_host,
        "translation_total_wall_ms": (time.perf_counter() - total_wall_start) * 1000.0,
        "translation_cpu_ms": (time.process_time() - total_cpu_start) * 1000.0,
    }
    return translated, timing
