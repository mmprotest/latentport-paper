"""Deterministic, leakage-excluded E002 document manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from experiments.latentport.e001_handoff.runtime.constants import SOURCE, TARGET
from experiments.latentport.e001_handoff.runtime.modeling import load_tokenizer
from experiments.latentport.e002_coupler.runtime.constants import E001_ROOT, E002_ROOT, load_preregistration


FINEWEB_PATH = Path(
    r"C:\Users\Simon\.cache\huggingface\hub\datasets--HuggingFaceFW--fineweb-edu"
    r"\snapshots\87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    r"\sample\100BT\000_00000.parquet"
)


def sha256_file(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_hash(prereg: dict[str, Any], row_index: int, document_id: str) -> str:
    corpus = prereg["corpus"]
    payload = "\0".join(
        (
            prereg["seeds"]["split_hash_salt"],
            corpus["revision"],
            corpus["immutable_file"],
            str(row_index),
            document_id,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _e001_exclusions() -> tuple[set[str], set[str], dict[str, int]]:
    ids: set[str] = set()
    texts: set[str] = set()
    counts = {}
    for split in ("fit", "validation", "locked", "long"):
        path = E001_ROOT / "data" / split / "manifest.jsonl"
        split_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                ids.add(str(row["document_id"]))
                texts.add(str(row["text_sha256"]))
                split_count += 1
        counts[split] = split_count
    return ids, texts, counts


def _existing_e002_exclusions() -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    texts: set[str] = set()
    for split in ("validation_factorial", "locked", "long"):
        path = E002_ROOT / "data" / split / "manifest.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    ids.add(str(row["document_id"]))
                    texts.add(str(row["text_sha256"]))
    return ids, texts


def _iter_rows(limit: int = 100_000) -> Iterable[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(FINEWEB_PATH)
    seen = 0
    columns = ["text", "id", "dump", "url", "language", "token_count", "score", "int_score"]
    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            if seen >= limit:
                return
            yield seen, row
            seen += 1


def _candidates(required_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prereg = load_preregistration()
    expected_corpus_hash = prereg["corpus"]["file_sha256"]
    actual_corpus_hash = sha256_file(FINEWEB_PATH)
    if actual_corpus_hash != expected_corpus_hash:
        raise RuntimeError("E002 corpus hash mismatch")
    source_tokenizer = load_tokenizer()
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET.snapshot, local_files_only=True)
    e001_ids, e001_texts, e001_counts = _e001_exclusions()
    e002_ids, e002_texts = _existing_e002_exclusions()
    coarse_minimum = max(1, required_tokens - 512)
    candidates = []
    coarse_rows = 0
    for row_index, row in _iter_rows():
        if int(row["token_count"]) < coarse_minimum:
            continue
        coarse_rows += 1
        document_id = str(row["id"])
        text_sha = _text_hash(row["text"])
        if document_id in e001_ids or text_sha in e001_texts or document_id in e002_ids or text_sha in e002_texts:
            continue
        token_ids = source_tokenizer.encode(row["text"], add_special_tokens=False)
        if len(token_ids) < required_tokens:
            continue
        candidates.append(
            {
                "corpus": prereg["corpus"]["repository"],
                "corpus_revision": prereg["corpus"]["revision"],
                "source_file": prereg["corpus"]["immutable_file"],
                "physical_row_index": row_index,
                "document_id": document_id,
                "url": row["url"],
                "dump": row["dump"],
                "language": row["language"],
                "source_reported_token_count": int(row["token_count"]),
                "fineweb_edu_score": row["score"],
                "fineweb_edu_int_score": row["int_score"],
                "text_sha256": text_sha,
                "selection_sha256": _selection_hash(prereg, row_index, document_id),
                "full_token_count": len(token_ids),
                "_token_ids": token_ids,
                "_text": row["text"],
            }
        )
    candidates.sort(key=lambda value: value["selection_sha256"])
    unique = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for row in candidates:
        if row["document_id"] in seen_ids or row["text_sha256"] in seen_texts:
            continue
        seen_ids.add(row["document_id"])
        seen_texts.add(row["text_sha256"])
        unique.append(row)
    audit = {
        "candidate_rows": 100_000,
        "coarse_minimum": coarse_minimum,
        "coarse_rows": coarse_rows,
        "exact_eligible_after_exclusion": len(candidates),
        "exact_unique_after_exclusion": len(unique),
        "e001_exclusion_counts": e001_counts,
        "corpus_sha256": actual_corpus_hash,
    }
    return unique, audit


def _write_manifest(split: str, rows: list[dict[str, Any]], stored_tokens: int) -> dict[str, Any]:
    output_dir = E002_ROOT / "data" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.jsonl"
    digest = hashlib.sha256()
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET.snapshot, local_files_only=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for split_index, source in enumerate(rows):
            row = {key: value for key, value in source.items() if key not in ("_token_ids", "_text")}
            ids = source["_token_ids"][:stored_tokens]
            target_ids = target_tokenizer.encode(source["_text"], add_special_tokens=False)[:stored_tokens]
            if ids != target_ids:
                raise RuntimeError("source/target token mismatch in selected E002 document")
            row.update(
                {
                    "split": split.upper(),
                    "split_index": split_index,
                    "stored_token_count": stored_tokens,
                    "token_ids": ids,
                }
            )
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return {
        "split": split.upper(),
        "documents": len(rows),
        "stored_tokens": stored_tokens,
        "manifest": str(path.relative_to(E002_ROOT)).replace("\\", "/"),
        "manifest_sha256": digest.hexdigest(),
        "document_ids_sha256": hashlib.sha256(
            "\n".join(row["document_id"] for row in rows).encode("utf-8")
        ).hexdigest(),
    }


def _write_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_correction_references() -> None:
    refs = {
        "fit_correction": ("fit", 128),
        "validation_correction": ("validation", 32),
    }
    e001_manifest = json.loads((E001_ROOT / "FROZEN_MANIFEST.json").read_text(encoding="utf-8"))
    frozen = {record["path"]: record for record in e001_manifest["files"]}
    for e002_split, (e001_split, documents) in refs.items():
        relative = f"data/{e001_split}/manifest.jsonl"
        record = {
            "source_experiment": "LATENTPORT_E001_HANDOFF",
            "source_split": e001_split.upper(),
            "documents": documents,
            "prefix_tokens": 1024,
            "path": f"experiments/latentport/e001_handoff/{relative}",
            "sha256": frozen[relative]["sha256"],
            "e001_locked": False,
        }
        path = E002_ROOT / "data" / e002_split / "reference.json"
        if not path.exists():
            _write_json_once(path, record)


def build_factorial() -> dict[str, Any]:
    prereg = load_preregistration()
    split = prereg["splits"]["factorial_validation"]
    candidates, audit = _candidates(split["stored_tokens"])
    if len(candidates) < split["documents"]:
        raise RuntimeError("insufficient frozen factorial candidates")
    selected = candidates[: split["documents"]]
    manifest = _write_manifest("validation_factorial", selected, split["stored_tokens"])
    result = {"selection_uses_model_outputs": False, "manifest": manifest, "audit": audit}
    _write_json_once(E002_ROOT / "data" / "validation_factorial" / "selection.json", result)
    _write_correction_references()
    return result


def build_locked_and_long() -> dict[str, Any]:
    prereg = load_preregistration()
    long_spec = prereg["splits"]["long"]
    locked_spec = prereg["splits"]["locked"]
    long_candidates, long_audit = _candidates(long_spec["stored_tokens"])
    if len(long_candidates) < long_spec["documents"]:
        raise RuntimeError("insufficient frozen LONG candidates")
    long_rows = long_candidates[: long_spec["documents"]]
    # _candidates observes the just-written factorial manifest but LONG is not yet
    # written, so explicitly reserve its identities before choosing LOCKED.
    reserved_ids = {row["document_id"] for row in long_rows}
    reserved_texts = {row["text_sha256"] for row in long_rows}
    locked_candidates, locked_audit = _candidates(locked_spec["stored_tokens"])
    locked_candidates = [
        row
        for row in locked_candidates
        if row["document_id"] not in reserved_ids and row["text_sha256"] not in reserved_texts
    ]
    if len(locked_candidates) < locked_spec["documents"]:
        raise RuntimeError("insufficient frozen LOCKED candidates")
    locked_rows = locked_candidates[: locked_spec["documents"]]
    long_manifest = _write_manifest("long", long_rows, long_spec["stored_tokens"])
    locked_manifest = _write_manifest("locked", locked_rows, locked_spec["stored_tokens"])
    result = {
        "selection_uses_model_outputs": False,
        "locked": locked_manifest,
        "long": long_manifest,
        "locked_audit": locked_audit,
        "long_audit": long_audit,
        "overlap_count": len(
            {row["document_id"] for row in long_rows}
            & {row["document_id"] for row in locked_rows}
        ),
    }
    _write_json_once(E002_ROOT / "data" / "locked" / "selection.json", result)
    return result


def read_training_rows(split: str) -> list[dict[str, Any]]:
    mapping = {"fit_correction": "fit", "validation_correction": "validation"}
    if split not in mapping:
        raise ValueError("training rows are restricted to E001 FIT/VALIDATION")
    path = E001_ROOT / "data" / mapping[split] / "manifest.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("factorial", "locked"))
    args = parser.parse_args()
    result = build_factorial() if args.phase == "factorial" else build_locked_and_long()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
