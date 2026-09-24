"""Pinned, language-only Qwen3.5 model loading and identity checks."""

from __future__ import annotations

import gc
import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from .constants import ModelSpec, SOURCE


LANGUAGE_PREFIX_MAPPING = {r"^model\.language_model\.": "model."}


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_text_config(spec: ModelSpec):
    if not spec.snapshot.is_dir():
        raise FileNotFoundError(f"Pinned snapshot is absent: {spec.snapshot}")
    outer = AutoConfig.from_pretrained(spec.snapshot, local_files_only=True)
    config = outer.text_config
    # E001 uses inspectable native PyTorch paths. Optional external kernels and
    # attention fusions could alter numerical behavior and are not canonical.
    config._attn_implementation = "eager"
    config.use_kernels = False
    config.use_cache = True
    return config


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(SOURCE.snapshot, local_files_only=True)
    # The model matrix has 248,320 rows; the canonical tokenizer defines
    # 248,077 IDs (maximum ID 248,076), leaving reserved embedding rows.
    if len(tokenizer) != 248_077:
        raise RuntimeError(f"Unexpected tokenizer vocabulary size: {len(tokenizer)}")
    return tokenizer


def load_language_model(
    spec: ModelSpec,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3_5ForCausalLM:
    """Load the text tower only, proving every expected tensor was populated.

    Official Qwen3.5 Base checkpoints are packaged under the multimodal
    ``model.language_model`` namespace. The explicit regex mapping loads that
    tower into ``Qwen3_5ForCausalLM`` and excludes visual/MTP modules without
    changing language weights.
    """

    config = load_text_config(spec)
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        spec.snapshot,
        config=config,
        key_mapping=LANGUAGE_PREFIX_MAPPING,
        dtype=dtype,
        device_map={"": device},
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    problems = {
        "missing_keys": sorted(loading_info["missing_keys"]),
        "unexpected_keys": sorted(loading_info["unexpected_keys"]),
        "mismatched_keys": sorted(loading_info["mismatched_keys"]),
        "error_messages": list(loading_info.get("error_msgs", [])),
    }
    if any(problems.values()):
        raise RuntimeError(f"Language-only checkpoint load was not exact: {problems}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec.expected_parameter_count:
        raise RuntimeError(
            f"{spec.repository} parameter count {parameter_count:,} != "
            f"frozen {spec.expected_parameter_count:,}"
        )
    model.requires_grad_(False)
    model.eval()
    return model


def release_model(model: Any | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def tokenizer_identity() -> dict[str, Any]:
    tokenizer = load_tokenizer()
    tokenizer_file = SOURCE.snapshot / "tokenizer.json"
    return {
        "repository": SOURCE.repository,
        "revision": SOURCE.revision,
        "class": type(tokenizer).__name__,
        "vocabulary_size": len(tokenizer),
        "tokenizer_json_sha256": sha256_file(tokenizer_file),
    }
