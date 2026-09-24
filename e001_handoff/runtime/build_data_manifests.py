"""Build deterministic, corpus-isolated E001 token manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from experiments.latentport.e001_handoff.runtime.constants import E001_ROOT, SOURCE, TARGET, load_preregistration
from experiments.latentport.e001_handoff.runtime.modeling import load_tokenizer


FINEWEB_PATH = Path(
    r"C:\Users\Simon\.cache\huggingface\hub\datasets--HuggingFaceFW--fineweb-edu"
    r"\snapshots\87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    r"\sample\100BT\000_00000.parquet"
)
PG19_PATH = Path(
    r"C:\Users\Simon\.cache\huggingface\hub\datasets--emozilla--pg19"
    r"\snapshots\c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"
    r"\data\test-00000-of-00001-29a571947c0b5ccc.parquet"
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def selection_hash(
    salt: str,
    revision: str,
    immutable_file: str,
    row_index: int,
    document_id: str,
) -> str:
    payload = "\0".join((salt, revision, immutable_file, str(row_index), document_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_parquet_rows(path: Path, columns: list[str], limit: int | None = None) -> Iterable[tuple[int, dict]]:
    parquet = pq.ParquetFile(path)
    seen = 0
    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        values = batch.to_pylist()
        for row in values:
            if limit is not None and seen >= limit:
                return
            yield seen, row
            seen += 1


def _tokenize_both(source_tokenizer, target_tokenizer, text: str) -> list[int]:
    source_ids = source_tokenizer.encode(text, add_special_tokens=False)
    target_ids = target_tokenizer.encode(text, add_special_tokens=False)
    if source_ids != target_ids:
        raise RuntimeError("E001_STATUS = INCONCLUSIVE_TOKENIZATION_MISMATCH")
    return source_ids


def _deduplicate_hash_sorted(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    candidates.sort(key=lambda row: row["selection_sha256"])
    result = []
    document_ids: set[str] = set()
    text_hashes: set[str] = set()
    duplicates = 0
    for row in candidates:
        if row["document_id"] in document_ids or row["text_sha256"] in text_hashes:
            duplicates += 1
            continue
        document_ids.add(row["document_id"])
        text_hashes.add(row["text_sha256"])
        result.append(row)
    return result, duplicates


def build_fineweb(prereg: dict, source_tokenizer, target_tokenizer) -> tuple[list[dict], list[dict], dict]:
    corpus = prereg["corpora"]["fit_validation"]
    minimum = prereg["corpora"]["minimum_tokens"]["fit"]
    candidates: list[dict[str, Any]] = []
    for row_index, row in iter_parquet_rows(
        FINEWEB_PATH,
        ["text", "id", "dump", "url", "language", "token_count", "score", "int_score"],
        limit=20_000,
    ):
        ids = _tokenize_both(source_tokenizer, target_tokenizer, row["text"])
        if len(ids) >= minimum:
            doc_id = str(row["id"])
            candidates.append(
                {
                    "corpus": corpus["repository"],
                    "corpus_revision": corpus["revision"],
                    "source_file": corpus["immutable_file"],
                    "physical_row_index": row_index,
                    "document_id": doc_id,
                    "url": row["url"],
                    "dump": row["dump"],
                    "language": row["language"],
                    "source_reported_token_count": row["token_count"],
                    "fineweb_edu_score": row["score"],
                    "fineweb_edu_int_score": row["int_score"],
                    "text_sha256": text_hash(row["text"]),
                    "selection_sha256": selection_hash(
                        prereg["seeds"]["split_hash_salt"],
                        corpus["revision"],
                        corpus["immutable_file"],
                        row_index,
                        doc_id,
                    ),
                    "full_token_count": len(ids),
                    "token_ids": ids[:minimum],
                }
            )
        if (row_index + 1) % 1_000 == 0:
            print(f"FineWeb rows {row_index + 1}/20000; eligible={len(candidates)}", flush=True)
    unique, duplicates = _deduplicate_hash_sorted(candidates)
    if len(unique) < 160:
        raise RuntimeError(f"Frozen FineWeb universe has only {len(unique)} eligible unique documents")
    fit = unique[:128]
    validation = unique[128:160]
    return fit, validation, {
        "candidate_rows": 20_000,
        "eligible_rows": len(candidates),
        "eligible_unique_documents": len(unique),
        "duplicates_removed": duplicates,
    }


def build_pg19(prereg: dict, source_tokenizer, target_tokenizer) -> tuple[list[dict], list[dict], dict]:
    corpus = prereg["corpora"]["locked_long"]
    locked_minimum = prereg["corpora"]["minimum_tokens"]["locked"]
    long_minimum = prereg["corpora"]["minimum_tokens"]["long"]
    candidates: list[dict[str, Any]] = []
    for row_index, row in iter_parquet_rows(
        PG19_PATH,
        ["short_book_title", "publication_date", "url", "text"],
    ):
        ids = _tokenize_both(source_tokenizer, target_tokenizer, row["text"])
        doc_id = str(row["url"])
        candidates.append(
            {
                "corpus": corpus["repository"],
                "corpus_revision": corpus["revision"],
                "source_file": corpus["immutable_file"],
                "physical_row_index": row_index,
                "document_id": doc_id,
                "url": row["url"],
                "short_book_title": row["short_book_title"],
                "publication_date": row["publication_date"],
                "text_sha256": text_hash(row["text"]),
                "selection_sha256": selection_hash(
                    prereg["seeds"]["split_hash_salt"],
                    corpus["revision"],
                    corpus["immutable_file"],
                    row_index,
                    doc_id,
                ),
                "full_token_count": len(ids),
                "_all_token_ids": ids,
            }
        )
        print(f"PG19 rows {row_index + 1}/100", flush=True)
    unique, duplicates = _deduplicate_hash_sorted(candidates)
    long_rows = [row for row in unique if row["full_token_count"] >= long_minimum][:16]
    long_ids = {row["document_id"] for row in long_rows}
    locked_rows = [
        row
        for row in unique
        if row["document_id"] not in long_ids and row["full_token_count"] >= locked_minimum
    ][:64]
    if len(long_rows) != 16 or len(locked_rows) != 64:
        raise RuntimeError(f"Frozen PG19 universe yielded LONG={len(long_rows)}, LOCKED={len(locked_rows)}")
    for row in long_rows:
        row["token_ids"] = row.pop("_all_token_ids")[:long_minimum]
    for row in locked_rows:
        row["token_ids"] = row.pop("_all_token_ids")[:locked_minimum]
    return locked_rows, long_rows, {
        "candidate_rows": len(candidates),
        "eligible_long": sum(row["full_token_count"] >= long_minimum for row in unique),
        "eligible_locked": sum(row["full_token_count"] >= locked_minimum for row in unique),
        "eligible_unique_documents": len(unique),
        "duplicates_removed": duplicates,
    }


def _write_split(name: str, rows: list[dict[str, Any]], required_tokens: int) -> dict[str, Any]:
    output_dir = E001_ROOT / "data" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    digest = hashlib.sha256()
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for split_index, source_row in enumerate(rows):
            row = dict(source_row)
            row["split"] = name.upper()
            row["split_index"] = split_index
            row["stored_token_count"] = len(row["token_ids"])
            if row["stored_token_count"] != required_tokens:
                raise RuntimeError(f"{name} row {split_index} has wrong stored length")
            line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return {
        "split": name.upper(),
        "documents": len(rows),
        "required_tokens_per_document": required_tokens,
        "manifest": str(manifest.relative_to(E001_ROOT)).replace("\\", "/"),
        "manifest_sha256": digest.hexdigest(),
        "document_ids_sha256": hashlib.sha256(
            "\n".join(row["document_id"] for row in rows).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    prereg = load_preregistration()
    if not prereg["corpora"]["identities_frozen"]:
        raise RuntimeError("Corpus identities must be frozen before manifest generation")
    expected_hashes = {
        FINEWEB_PATH: prereg["corpora"]["fit_validation"]["file_sha256"],
        PG19_PATH: prereg["corpora"]["locked_long"]["file_sha256"],
    }
    verified_files = {}
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Corpus file hash mismatch for {path}: {actual} != {expected}")
        verified_files[str(path)] = actual

    source_tokenizer = load_tokenizer()
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET.snapshot, local_files_only=True)
    fit, validation, fineweb_audit = build_fineweb(prereg, source_tokenizer, target_tokenizer)
    locked, long_rows, pg19_audit = build_pg19(prereg, source_tokenizer, target_tokenizer)
    splits = {
        "fit": _write_split("fit", fit, prereg["corpora"]["minimum_tokens"]["fit"]),
        "validation": _write_split(
            "validation", validation, prereg["corpora"]["minimum_tokens"]["validation"]
        ),
        "locked": _write_split("locked", locked, prereg["corpora"]["minimum_tokens"]["locked"]),
        "long": _write_split("long", long_rows, prereg["corpora"]["minimum_tokens"]["long"]),
    }
    all_ids = [row["document_id"] for rows in (fit, validation, locked, long_rows) for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Split isolation failure: duplicate document identity")
    corpus_by_split = {
        "fit": {row["corpus"] for row in fit},
        "validation": {row["corpus"] for row in validation},
        "locked": {row["corpus"] for row in locked},
        "long": {row["corpus"] for row in long_rows},
    }
    if corpus_by_split["fit"] & corpus_by_split["locked"]:
        raise RuntimeError("Corpus isolation failure")
    summary = {
        "manifest_version": "latentport-e001-data-v1",
        "selection_uses_model_outputs": False,
        "tokenizer_source_repository": SOURCE.repository,
        "tokenizer_source_revision": SOURCE.revision,
        "tokenizer_target_repository": TARGET.repository,
        "tokenizer_target_revision": TARGET.revision,
        "exact_token_equality_verified_for_all_selected_documents": True,
        "verified_corpus_files": verified_files,
        "fineweb_candidate_audit": fineweb_audit,
        "pg19_candidate_audit": pg19_audit,
        "splits": splits,
        "document_overlap_count": len(all_ids) - len(set(all_ids)),
        "corpus_by_split": {key: sorted(value) for key, value in corpus_by_split.items()},
    }
    output = E001_ROOT / "data" / "MANIFEST_SUMMARY.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
