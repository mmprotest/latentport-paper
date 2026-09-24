"""Trainable-zero, fixed-basis identity-anchored E002 correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection

import torch
from torch import nn

from experiments.latentport.e001_handoff.state.cache_state import STATE_FORMAT_VERSION
from experiments.latentport.e002_coupler.runtime.constants import TRAINABLE_PARAMETER_CAP


ATTENTION_LAYERS = tuple(range(3, 32, 4))
GDN_LAYERS = tuple(index for index in range(32) if index not in ATTENTION_LAYERS)


def _orthonormal_bases(count: int, dimension: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    bases = []
    for _ in range(count):
        raw = torch.randn(dimension, rank, generator=generator, dtype=torch.float32)
        q, r = torch.linalg.qr(raw, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1
        bases.append(q * signs.unsqueeze(0))
    return torch.stack(bases)


@dataclass
class CorrectionApplication:
    state: dict[str, Any]
    identity_loss: torch.Tensor
    component_relative_squared: dict[str, torch.Tensor]
    layer_relative_magnitudes: list[dict[str, Any]]


class IdentityAnchoredCoupler(nn.Module):
    """Rank-constrained residual with an exact identity initialization.

    Each low-rank product has one frozen orthonormal factor and one trainable
    factor initialized to exactly zero. The trainable parameter set therefore
    starts at zero without the zero-gradient pathology of two zero factors.
    """

    def __init__(self, rank: int, *, basis_seed: int = 2026090102):
        super().__init__()
        if rank not in (2, 4):
            raise ValueError("E002 rank must be 2 or 4")
        self.rank = int(rank)
        self.basis_seed = int(basis_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.basis_seed)

        self.register_buffer(
            "kv_key_basis", _orthonormal_bases(len(ATTENTION_LAYERS), 256, rank, generator)
        )
        self.register_buffer(
            "kv_value_basis", _orthonormal_bases(len(ATTENTION_LAYERS), 256, rank, generator)
        )
        self.register_buffer(
            "recurrent_left_basis", _orthonormal_bases(len(GDN_LAYERS), 128, rank, generator)
        )
        self.register_buffer(
            "recurrent_right_basis", _orthonormal_bases(len(GDN_LAYERS), 128, rank, generator)
        )

        self.kv_key_output = nn.Parameter(torch.zeros(len(ATTENTION_LAYERS), 256, rank))
        self.kv_value_output = nn.Parameter(torch.zeros(len(ATTENTION_LAYERS), 256, rank))
        self.recurrent_left_output = nn.Parameter(torch.zeros(len(GDN_LAYERS), 128, rank))
        self.recurrent_right_output = nn.Parameter(torch.zeros(len(GDN_LAYERS), 128, rank))
        self.convolution_scale_residual = nn.Parameter(torch.zeros(len(GDN_LAYERS), 8192))
        self.convolution_bias = nn.Parameter(torch.zeros(len(GDN_LAYERS), 8192))

        if self.parameter_count >= TRAINABLE_PARAMETER_CAP:
            raise ValueError("E002 correction violates the trainable parameter cap")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def all_trainable_parameters_are_zero(self) -> bool:
        return all(bool(torch.count_nonzero(parameter.detach()) == 0) for parameter in self.parameters())

    def _kv_delta(self, x: torch.Tensor, basis: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(x.float(), basis)
        return torch.matmul(projected, output.transpose(0, 1))

    def _recurrent_delta(
        self,
        state: torch.Tensor,
        left_basis: torch.Tensor,
        right_basis: torch.Tensor,
        left_output: torch.Tensor,
        right_output: torch.Tensor,
    ) -> torch.Tensor:
        base = state.float()
        left = torch.einsum("ir,hrj->hij", left_output, torch.einsum("ir,hij->hrj", left_basis, base))
        right = torch.einsum("hir,jr->hij", torch.einsum("hij,jr->hir", base, right_basis), right_output)
        return (left + right).unsqueeze(0)

    def apply_to_state(
        self,
        base_state: dict[str, Any],
        *,
        detach_to_cpu: bool = False,
        disabled_components: Collection[str] = (),
        disabled_gdn_positions: Collection[int] = (),
    ) -> CorrectionApplication:
        device = next(self.parameters()).device
        disabled = set(disabled_components)
        disabled_depth = set(int(value) for value in disabled_gdn_positions)
        if not disabled <= {"K", "R", "C"}:
            raise ValueError("disabled component must be K, R, or C")

        layers = []
        component_numerator = {name: torch.zeros((), device=device) for name in ("K", "R", "C")}
        component_denominator = {name: torch.zeros((), device=device) for name in ("K", "R", "C")}
        magnitudes: list[dict[str, Any]] = []
        attention_position = 0
        gdn_position = 0
        for record in base_state["layers"]:
            if record["layer_type"] == "full_attention":
                keys = record["keys"].to(device)
                values = record["values"].to(device)
                if "K" in disabled:
                    key_delta = torch.zeros_like(keys, dtype=torch.float32)
                    value_delta = torch.zeros_like(values, dtype=torch.float32)
                else:
                    key_delta = self._kv_delta(
                        keys,
                        self.kv_key_basis[attention_position],
                        self.kv_key_output[attention_position],
                    )
                    value_delta = self._kv_delta(
                        values,
                        self.kv_value_basis[attention_position],
                        self.kv_value_output[attention_position],
                    )
                corrected_keys = keys + key_delta.to(keys.dtype)
                corrected_values = values + value_delta.to(values.dtype)
                component_numerator["K"] = component_numerator["K"] + key_delta.square().sum() + value_delta.square().sum()
                component_denominator["K"] = component_denominator["K"] + keys.float().square().sum() + values.float().square().sum()
                denominator = (keys.float().square().sum() + values.float().square().sum()).sqrt().clamp_min(1e-12)
                relative = (key_delta.square().sum() + value_delta.square().sum()).sqrt() / denominator
                magnitudes.append(
                    {
                        "layer_index": int(record["layer_index"]),
                        "component": "K",
                        "relative_norm": relative,
                    }
                )
                layers.append(
                    {
                        "layer_index": record["layer_index"],
                        "layer_type": record["layer_type"],
                        "cache_class": record["cache_class"],
                        "keys": corrected_keys,
                        "values": corrected_values,
                        "is_initialized": True,
                    }
                )
                attention_position += 1
                continue

            recurrent = record["recurrent_states"][0].to(device)
            convolution = record["conv_states"][0].to(device)
            if "R" in disabled or gdn_position in disabled_depth:
                recurrent_delta = torch.zeros_like(recurrent, dtype=torch.float32)
            else:
                recurrent_delta = self._recurrent_delta(
                    recurrent[0],
                    self.recurrent_left_basis[gdn_position],
                    self.recurrent_right_basis[gdn_position],
                    self.recurrent_left_output[gdn_position],
                    self.recurrent_right_output[gdn_position],
                )
            if "C" in disabled:
                convolution_delta = torch.zeros_like(convolution, dtype=torch.float32)
            else:
                convolution_delta = (
                    convolution.float() * self.convolution_scale_residual[gdn_position][None, :, None]
                    + self.convolution_bias[gdn_position][None, :, None]
                )
            corrected_recurrent = recurrent + recurrent_delta.to(recurrent.dtype)
            corrected_convolution = convolution + convolution_delta.to(convolution.dtype)
            component_numerator["R"] = component_numerator["R"] + recurrent_delta.square().sum()
            component_denominator["R"] = component_denominator["R"] + recurrent.float().square().sum()
            component_numerator["C"] = component_numerator["C"] + convolution_delta.square().sum()
            component_denominator["C"] = component_denominator["C"] + convolution.float().square().sum()
            for name, delta, base in (
                ("R", recurrent_delta, recurrent),
                ("C", convolution_delta, convolution),
            ):
                magnitudes.append(
                    {
                        "layer_index": int(record["layer_index"]),
                        "gdn_position": gdn_position,
                        "component": name,
                        "relative_norm": delta.square().sum().sqrt()
                        / base.float().square().sum().sqrt().clamp_min(1e-12),
                    }
                )
            layers.append(
                {
                    "layer_index": record["layer_index"],
                    "layer_type": record["layer_type"],
                    "cache_class": record["cache_class"],
                    "number_of_states": 1,
                    "conv_states": {0: corrected_convolution},
                    "recurrent_states": {0: corrected_recurrent},
                    "is_conv_states_initialized": {0: True},
                    "is_recurrent_states_initialized": {0: True},
                    "has_previous_state": {0: True},
                    "conv_kernel_size": {0: 4},
                    "record_past": False,
                }
            )
            gdn_position += 1

        relative_squared = {
            name: component_numerator[name] / component_denominator[name].clamp_min(1e-12)
            for name in ("K", "R", "C")
        }
        identity_loss = torch.stack(tuple(relative_squared.values())).mean()
        state = {
            "format_version": STATE_FORMAT_VERSION,
            "num_hidden_layers": 32,
            "layer_types": list(base_state["layer_types"]),
            "sequence_length": int(base_state["sequence_length"]),
            "layers": layers,
            "e002_base_condition": base_state.get("e002_condition"),
            "e002_corrected": True,
        }
        if detach_to_cpu:
            for layer in state["layers"]:
                if layer["layer_type"] == "full_attention":
                    layer["keys"] = layer["keys"].detach().cpu()
                    layer["values"] = layer["values"].detach().cpu()
                else:
                    layer["conv_states"][0] = layer["conv_states"][0].detach().cpu()
                    layer["recurrent_states"][0] = layer["recurrent_states"][0].detach().cpu()
        return CorrectionApplication(state, identity_loss, relative_squared, magnitudes)


def parameter_inventory(rank: int) -> dict[str, int]:
    model = IdentityAnchoredCoupler(rank)
    return {
        "kv": model.kv_key_output.numel() + model.kv_value_output.numel(),
        "recurrent": model.recurrent_left_output.numel() + model.recurrent_right_output.numel(),
        "convolution": model.convolution_scale_residual.numel() + model.convolution_bias.numel(),
        "total": model.parameter_count,
    }

