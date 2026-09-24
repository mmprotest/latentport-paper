"""Bridge-token and teacher-forced continuation scoring for E001."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import torch
import torch.nn.functional as F


def position_ids(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)


@torch.inference_mode()
def score_from_cache(
    model,
    cache,
    *,
    prefix_length: int,
    bridge_token_id: int,
    continuation_token_ids: list[int],
    retain_logits: bool = False,
) -> dict[str, Any]:
    """Process one bridge token and score exactly 64 held-out tokens."""

    if len(continuation_token_ids) != 64:
        raise ValueError("E001 scoring requires exactly 64 continuation targets")
    device = next(model.parameters()).device
    bridge = torch.tensor([[bridge_token_id]], dtype=torch.long, device=device)
    targets = torch.tensor(continuation_token_ids, dtype=torch.long, device=device)
    bridge_logits = model(
        input_ids=bridge,
        position_ids=position_ids(prefix_length, 1, device),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    ).logits[:, -1, :]
    teacher_inputs = targets[:-1].unsqueeze(0)
    teacher_logits = model(
        input_ids=teacher_inputs,
        position_ids=position_ids(prefix_length + 1, teacher_inputs.shape[1], device),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=teacher_inputs.shape[1],
    ).logits[0]
    logits = torch.cat([bridge_logits, teacher_logits], dim=0).float()
    per_token_nll = -F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    top5 = torch.topk(logits, k=5, dim=-1).indices
    probabilities = F.softmax(logits, dim=-1)
    entropy = -(probabilities * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    byte_view = logits.detach().to(torch.bfloat16).contiguous().cpu().view(torch.uint8).numpy().tobytes()
    result: dict[str, Any] = {
        "nll": float(per_token_nll.mean().item()),
        "per_token_nll": per_token_nll.cpu().tolist(),
        "top1_token_ids": top5[:, 0].cpu().tolist(),
        "top5_token_ids": top5.cpu().tolist(),
        "per_token_entropy": entropy.cpu().tolist(),
        "logits_bfloat16_sha256": hashlib.sha256(byte_view).hexdigest(),
        "tokens_processed_by_target": 64,
        "historical_prefix_tokens_processed_by_target": 0,
    }
    if retain_logits:
        result["logits"] = logits.cpu()
    return result


def compare_to_reference(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_top1 = candidate["top1_token_ids"]
    reference_top1 = reference["top1_token_ids"]
    top1_agreement = sum(a == b for a, b in zip(candidate_top1, reference_top1, strict=True)) / 64.0
    top5_overlap = []
    for candidate_set, reference_set in zip(
        candidate["top5_token_ids"], reference["top5_token_ids"], strict=True
    ):
        top5_overlap.append(len(set(candidate_set) & set(reference_set)) / 5.0)
    return {
        "delta_nll_to_native": candidate["nll"] - reference["nll"],
        "top1_agreement_to_native": top1_agreement,
        "mean_top5_overlap_to_native": sum(top5_overlap) / len(top5_overlap),
        "per_token_top5_overlap_to_native": top5_overlap,
    }


def json_safe_score(score: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if key not in ("logits", "convergence_states")}


@torch.inference_mode()
def score_from_cache_segmented(
    model,
    cache,
    *,
    prefix_length: int,
    bridge_token_id: int,
    continuation_token_ids: list[int],
    retain_logits: bool = False,
    capture_convergence: bool = False,
) -> dict[str, Any]:
    """Canonical score using the identical 1/4/16/64 state-capture schedule."""

    from experiments.latentport.e001_handoff.state.convergence import capture_convergence_state

    if len(continuation_token_ids) != 64:
        raise ValueError("E001 scoring requires exactly 64 continuation targets")
    device = next(model.parameters()).device
    targets = torch.tensor(continuation_token_ids, dtype=torch.long, device=device)
    logits_parts = []
    convergence_states = {}
    torch.cuda.synchronize()
    bridge_start = time.perf_counter()
    bridge_cpu_start = time.process_time()
    bridge_logits = model(
        input_ids=torch.tensor([[bridge_token_id]], dtype=torch.long, device=device),
        position_ids=position_ids(prefix_length, 1, device),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    ).logits[0]
    torch.cuda.synchronize()
    bridge_ms = (time.perf_counter() - bridge_start) * 1000.0
    bridge_cpu_ms = (time.process_time() - bridge_cpu_start) * 1000.0
    logits_parts.append(bridge_logits)
    if capture_convergence:
        convergence_states[1] = capture_convergence_state(
            cache, model.config, historical_prefix_length=prefix_length
        )
    processed = 1
    continuation_ms = 0.0
    continuation_cpu_ms = 0.0
    for checkpoint in (4, 16, 64):
        token_count = checkpoint - processed
        start_index = processed - 1
        inputs = targets[start_index : start_index + token_count].unsqueeze(0)
        torch.cuda.synchronize()
        segment_start = time.perf_counter()
        segment_cpu_start = time.process_time()
        segment_logits = model(
            input_ids=inputs,
            position_ids=position_ids(prefix_length + processed, token_count, device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=token_count,
        ).logits[0]
        torch.cuda.synchronize()
        continuation_ms += (time.perf_counter() - segment_start) * 1000.0
        continuation_cpu_ms += (time.process_time() - segment_cpu_start) * 1000.0
        logits_parts.append(segment_logits)
        processed = checkpoint
        if capture_convergence:
            convergence_states[checkpoint] = capture_convergence_state(
                cache, model.config, historical_prefix_length=prefix_length
            )
    logits = torch.cat(logits_parts, dim=0).float()
    if logits.shape[0] != 64 or cache.get_seq_length() not in (64, prefix_length + 64):
        raise RuntimeError("Canonical segmented scoring accounting failed")
    log_probabilities = F.log_softmax(logits, dim=-1)
    per_token_nll = -log_probabilities.gather(1, targets[:, None]).squeeze(1)
    top5 = torch.topk(logits, k=5, dim=-1).indices
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    logits_bf16_cpu = logits.detach().to(torch.bfloat16).contiguous().cpu()
    per_token_hashes = [
        hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest()
        for row in logits_bf16_cpu
    ]
    result: dict[str, Any] = {
        "nll": float(per_token_nll.mean().item()),
        "per_token_nll": per_token_nll.cpu().tolist(),
        "top1_token_ids": top5[:, 0].cpu().tolist(),
        "top5_token_ids": top5.cpu().tolist(),
        "per_token_entropy": entropy.cpu().tolist(),
        "per_token_logits_bfloat16_sha256": per_token_hashes,
        "logits_bfloat16_sha256": hashlib.sha256(
            logits_bf16_cpu.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "tokens_processed_after_handoff": 64,
        "historical_prefix_tokens_processed_in_scoring_branch": 0,
        "logical_position_start": prefix_length,
        "continuation_schedule": [1, 4, 16, 64],
        "bridge_ms": bridge_ms,
        "bridge_cpu_ms": bridge_cpu_ms,
        "continuation_ms": continuation_ms,
        "continuation_cpu_ms": continuation_cpu_ms,
    }
    if retain_logits:
        result["logits"] = logits.cpu()
    if capture_convergence:
        result["convergence_states"] = convergence_states
    return result


def compare_full_distributions(candidate: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """Compute full-vocabulary fidelity metrics, then permit logits to be discarded."""

    if "logits" not in candidate or "logits" not in native:
        raise ValueError("Full distribution comparison requires retained logits")
    comparison_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate_logits = candidate["logits"].to(comparison_device).float()
    native_logits = native["logits"].to(comparison_device).float()
    if candidate_logits.shape != native_logits.shape:
        raise ValueError("Candidate/native logit shapes differ")
    native_logp = F.log_softmax(native_logits, dim=-1)
    candidate_logp = F.log_softmax(candidate_logits, dim=-1)
    native_p = native_logp.exp()
    candidate_p = candidate_logp.exp()
    mixture_logp = torch.logaddexp(native_logp, candidate_logp) - math.log(2.0)
    js = 0.5 * (
        (native_p * (native_logp - mixture_logp)).sum(dim=-1)
        + (candidate_p * (candidate_logp - mixture_logp)).sum(dim=-1)
    )
    kl_native_to_candidate = (native_p * (native_logp - candidate_logp)).sum(dim=-1)
    cosine = F.cosine_similarity(native_logits, candidate_logits, dim=-1)
    top1 = [a == b for a, b in zip(candidate["top1_token_ids"], native["top1_token_ids"], strict=True)]
    top5_overlap = [
        len(set(a) & set(b)) / 5.0
        for a, b in zip(candidate["top5_token_ids"], native["top5_token_ids"], strict=True)
    ]
    return {
        "top1_agreement_to_native": sum(top1) / 64.0,
        "per_token_top1_agreement_to_native": top1,
        "mean_top5_overlap_to_native": sum(top5_overlap) / 64.0,
        "per_token_top5_overlap_to_native": top5_overlap,
        "mean_js_to_native": float(js.mean().item()),
        "per_token_js_to_native": js.cpu().tolist(),
        "mean_kl_native_to_condition": float(kl_native_to_candidate.mean().item()),
        "per_token_kl_native_to_condition": kl_native_to_candidate.cpu().tolist(),
        "mean_logit_cosine_to_native": float(cosine.mean().item()),
        "per_token_logit_cosine_to_native": cosine.cpu().tolist(),
    }
