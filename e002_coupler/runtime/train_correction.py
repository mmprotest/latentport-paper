"""Fit, validate, select, and freeze the E002 correction candidates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from experiments.latentport.e001_handoff.runtime.constants import TARGET
from experiments.latentport.e001_handoff.runtime.modeling import (
    load_language_model,
    load_text_config,
    release_model,
    seed_everything,
)
from experiments.latentport.e002_coupler.correction.losses import behavior_kl, total_loss
from experiments.latentport.e002_coupler.correction.model import IdentityAnchoredCoupler
from experiments.latentport.e002_coupler.correction.serialization import (
    load_correction,
    save_correction,
)
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    TRAINABLE_PARAMETER_CAP,
    ensure_attempt_layout,
    load_preregistration,
)
from experiments.latentport.e002_coupler.runtime.data_manifests import read_training_rows
from experiments.latentport.e002_coupler.runtime.install import install_differentiable
from experiments.latentport.e002_coupler.runtime.prepare_correction import _paths
from experiments.latentport.e002_coupler.runtime.scoring import training_logits
from experiments.latentport.e002_coupler.runtime.state_io import load_logits, load_state, write_json_once


def _candidate_name(rank: int, identity_lambda: float) -> str:
    return f"rank_{rank}_lambda_{identity_lambda:.0e}".replace("+", "")


def _document_inputs(row: dict[str, Any]) -> tuple[int, list[int], torch.Tensor]:
    ids = row["token_ids"]
    continuation = [int(value) for value in ids[1025:1034]]
    targets = torch.tensor([continuation], dtype=torch.long, device="cuda:0")
    return int(ids[1024]), continuation, targets


def _nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()


def _run_validation(
    model,
    config,
    coupler: IdentityAnchoredCoupler,
) -> dict[str, Any]:
    rows = read_training_rows("validation_correction")
    behavioral_values = []
    identity_values = []
    nll_values = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            base = load_state(_paths("validation_correction", index)["base_state"])
            application = coupler.apply_to_state(base)
            cache = install_differentiable(application.state, config)
            bridge, continuation, targets = _document_inputs(row)
            logits = training_logits(
                model,
                cache,
                prefix_length=1024,
                bridge_token_id=bridge,
                continuation_token_ids=continuation,
            )
            native = load_logits(
                _paths("validation_correction", index)["native_logits"], device="cuda:0"
            )
            behavioral_values.append(float(behavior_kl(native, logits).item()))
            identity_values.append(float(application.identity_loss.item()))
            nll_values.append(float(_nll(logits, targets).item()))
            del base, application, cache, logits, native, targets
    return {
        "documents": len(rows),
        "mean_behavior_kl": float(np.mean(behavioral_values)),
        "median_behavior_kl": float(np.median(behavioral_values)),
        "mean_identity_loss": float(np.mean(identity_values)),
        "mean_9_position_nll": float(np.mean(nll_values)),
        "document_behavior_kl": behavioral_values,
        "document_identity_loss": identity_values,
        "document_9_position_nll": nll_values,
    }


def train_candidate(
    model,
    config,
    *,
    rank: int,
    identity_lambda: float,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    name = _candidate_name(rank, identity_lambda)
    root = ATTEMPT_ROOT / "fit" / "candidates" / name
    tensor_path = root / "correction.safetensors"
    metadata_path = root / "correction.json"
    if tensor_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    coupler = IdentityAnchoredCoupler(rank, basis_seed=prereg["seeds"]["fixed_basis"]).to("cuda:0")
    if coupler.parameter_count >= TRAINABLE_PARAMETER_CAP or not coupler.all_trainable_parameters_are_zero():
        raise RuntimeError("candidate architecture gate failed")
    training = prereg["training"]
    optimizer = torch.optim.AdamW(
        coupler.parameters(),
        lr=training["learning_rate_grid"][0],
        betas=tuple(training["betas"]),
        eps=training["epsilon"],
        weight_decay=training["weight_decay"],
    )
    rows = read_training_rows("fit_correction")
    best_validation = math.inf
    best_epoch = None
    best_state = None
    patience = 0
    epoch_records = []
    started = time.perf_counter()
    for epoch in range(1, training["max_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        document_records = []
        for index, row in enumerate(rows):
            base = load_state(_paths("fit_correction", index)["base_state"])
            application = coupler.apply_to_state(base)
            cache = install_differentiable(application.state, config)
            bridge, continuation, targets = _document_inputs(row)
            logits = training_logits(
                model,
                cache,
                prefix_length=1024,
                bridge_token_id=bridge,
                continuation_token_ids=continuation,
            )
            native = load_logits(_paths("fit_correction", index)["native_logits"], device="cuda:0")
            loss, parts = total_loss(
                native, logits, application.identity_loss, identity_lambda
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite correction loss in {name}")
            (loss / training["gradient_accumulation"]).backward()
            if (index + 1) % training["gradient_accumulation"] == 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    coupler.parameters(), training["gradient_clip_norm"]
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                gradient_norm = torch.tensor(float("nan"))
            document_records.append(
                {
                    "document_id": row["document_id"],
                    "behavior_kl": float(parts["behavior_kl"].detach().item()),
                    "identity_loss": float(parts["identity"].detach().item()),
                    "total_loss": float(parts["total"].detach().item()),
                    "handoff_9_position_nll": float(_nll(logits.detach(), targets).item()),
                    "optimizer_step": (index + 1) % training["gradient_accumulation"] == 0,
                    "gradient_norm_if_step": None
                    if not torch.isfinite(gradient_norm)
                    else float(gradient_norm.detach().item()),
                }
            )
            del base, application, cache, logits, native, loss, parts, targets
            if (index + 1) % 16 == 0:
                print(f"{name} epoch {epoch} train {index + 1}/{len(rows)}", flush=True)
        validation = _run_validation(model, config, coupler)
        epoch_record = {
            "epoch": epoch,
            "train_mean_behavior_kl": float(
                np.mean([record["behavior_kl"] for record in document_records])
            ),
            "train_mean_identity_loss": float(
                np.mean([record["identity_loss"] for record in document_records])
            ),
            "train_documents": document_records,
            "validation": validation,
        }
        epoch_records.append(epoch_record)
        metric = validation["mean_behavior_kl"]
        if metric < best_validation - training["early_stopping_min_delta"]:
            best_validation = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().contiguous().cpu().clone()
                for key, value in coupler.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        print(f"{name} epoch {epoch} validation_KL={metric:.8f}", flush=True)
        if patience >= training["early_stopping_patience"]:
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("candidate training produced no selectable checkpoint")
    coupler.load_state_dict(best_state, strict=True)
    metadata = save_correction(
        coupler,
        tensor_path,
        metadata_path,
        {
            "candidate": name,
            "identity_lambda": identity_lambda,
            "learning_rate": training["learning_rate_grid"][0],
            "best_epoch": best_epoch,
            "best_validation_behavior_kl": best_validation,
            "epochs": epoch_records,
            "wall_seconds": time.perf_counter() - started,
            "base_state": json.loads(
                (ATTEMPT_ROOT / "factorial" / "base_state_selection.json").read_text(encoding="utf-8")
            )["selected_state"],
            "fit_documents": 128,
            "validation_documents": 32,
            "target_historical_prefix_replay": 0,
        },
    )
    del coupler, optimizer
    torch.cuda.empty_cache()
    return metadata


def _magnitude_summary(model: IdentityAnchoredCoupler) -> dict[str, Any]:
    records = []
    component_squared = {name: [] for name in ("K", "R", "C")}
    rows = read_training_rows("validation_correction")
    with torch.no_grad():
        for index, _ in enumerate(rows):
            base = load_state(_paths("validation_correction", index)["base_state"])
            application = model.apply_to_state(base)
            for name, value in application.component_relative_squared.items():
                component_squared[name].append(float(value.item()))
            for row in application.layer_relative_magnitudes:
                records.append(
                    {
                        **{key: value for key, value in row.items() if key != "relative_norm"},
                        "relative_norm": float(row["relative_norm"].item()),
                    }
                )
    summaries = {}
    for name in ("K", "R", "C"):
        values = np.asarray(
            [row["relative_norm"] for row in records if row["component"] == name], dtype=np.float64
        )
        summaries[name] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(np.max(values)),
            "global_relative_norm_mean": float(
                np.mean(np.sqrt(np.asarray(component_squared[name], dtype=np.float64)))
            ),
        }
    all_values = np.asarray([row["relative_norm"] for row in records], dtype=np.float64)
    return {
        "by_component": summaries,
        "all_layers": {
            "median": float(np.median(all_values)),
            "mean": float(np.mean(all_values)),
            "p90": float(np.quantile(all_values, 0.9)),
            "maximum": float(np.max(all_values)),
        },
        "layer_records": records,
    }


def select_and_freeze(candidates: list[dict[str, Any]], prereg: dict[str, Any]) -> dict[str, Any]:
    minimum = min(record["best_validation_behavior_kl"] for record in candidates)
    eligible = [
        record
        for record in candidates
        if record["best_validation_behavior_kl"] <= minimum + 1e-6
    ]
    selected = min(
        eligible,
        key=lambda record: (
            record["rank"],
            -float(record["identity_lambda"]),
            record["best_epoch"],
            record["candidate"],
        ),
    )
    candidate_root = ATTEMPT_ROOT / "fit" / "candidates" / selected["candidate"]
    selected_model = load_correction(
        candidate_root / "correction.safetensors",
        candidate_root / "correction.json",
        device="cuda:0",
    )
    final_tensor = ATTEMPT_ROOT / "fit" / "correction.safetensors"
    final_metadata = ATTEMPT_ROOT / "fit" / "correction.json"
    magnitude = _magnitude_summary(selected_model)
    final = save_correction(
        selected_model,
        final_tensor,
        final_metadata,
        {
            "selected_candidate": selected["candidate"],
            "identity_lambda": selected["identity_lambda"],
            "best_epoch": selected["best_epoch"],
            "best_validation_behavior_kl": selected["best_validation_behavior_kl"],
            "base_state": selected["base_state"],
            "selection_rule": "minimum validation behavioral KL; 1e-6 ties lower rank, larger lambda, earlier epoch",
            "correction_magnitude_validation": magnitude,
            "target_parameters": TARGET.expected_parameter_count,
            "target_parameter_ratio": selected_model.parameter_count / TARGET.expected_parameter_count,
            "target_bytes_bfloat16": TARGET.expected_parameter_count * 2,
        },
    )
    final["parameter_bytes_ratio_to_target_bfloat16"] = final["tensor_bytes"] / (
        TARGET.expected_parameter_count * 2
    )
    selection = {
        "candidate_grid": [
            {
                "candidate": record["candidate"],
                "rank": record["rank"],
                "identity_lambda": record["identity_lambda"],
                "best_epoch": record["best_epoch"],
                "best_validation_behavior_kl": record["best_validation_behavior_kl"],
                "parameter_count": record["parameter_count"],
                "correction_hash": record["correction_hash"],
            }
            for record in candidates
        ],
        "selected": final,
        "eligible_within_tolerance": [record["candidate"] for record in eligible],
        "parameter_cap": TRAINABLE_PARAMETER_CAP,
        "parameter_cap_pass": final["parameter_count"] < TRAINABLE_PARAMETER_CAP,
        "target_weights_trainable": 0,
        "source_weights_trainable": 0,
        "locked_documents_accessed": 0,
    }
    write_json_once(ATTEMPT_ROOT / "fit" / "correction_selection.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_attempt_layout()
    prereg = load_preregistration()
    seed_everything(prereg["seeds"]["training"])
    config = load_text_config(TARGET)
    model = load_language_model(TARGET)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("target weights are not frozen")
    candidates = []
    for rank in prereg["correction"]["rank_grid"]:
        for identity_lambda in prereg["training"]["identity_lambda_grid"]:
            candidates.append(
                train_candidate(
                    model,
                    config,
                    rank=rank,
                    identity_lambda=float(identity_lambda),
                    prereg=prereg,
                )
            )
    selection = select_and_freeze(candidates, prereg)
    release_model(model)
    print(json.dumps(selection["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

