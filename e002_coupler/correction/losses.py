"""Behavioral and identity objectives for the E002 coupler."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def behavior_kl(native_logits: torch.Tensor, handoff_logits: torch.Tensor) -> torch.Tensor:
    if native_logits.shape != handoff_logits.shape or native_logits.ndim != 3:
        raise ValueError("native and handoff logits must have identical [batch, positions, vocab] shape")
    native_logp = F.log_softmax(native_logits.float(), dim=-1)
    handoff_logp = F.log_softmax(handoff_logits.float(), dim=-1)
    native_probability = native_logp.exp()
    return (native_probability * (native_logp - handoff_logp)).sum(dim=-1).mean()


def total_loss(
    native_logits: torch.Tensor,
    handoff_logits: torch.Tensor,
    identity_loss: torch.Tensor,
    identity_lambda: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    behavioral = behavior_kl(native_logits, handoff_logits)
    total = behavioral + float(identity_lambda) * identity_loss
    return total, {"behavior_kl": behavioral, "identity": identity_loss, "total": total}

