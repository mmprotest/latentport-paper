"""Read-only scientific inputs and deterministic public-output helpers."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1 = "e001_handoff"
E2 = "e002_coupler"
A1 = f"{E1}/artifacts/attempt_001"
A2 = f"{E2}/artifacts/attempt_001"
R1 = f"{A1}/verdict/RESULT.json"
R2 = f"{A2}/verdict/RESULT.json"
M1 = f"{E1}/data/locked/manifest.jsonl"
M2 = f"{E2}/data/locked/manifest.jsonl"
AB = f"{A2}/diagnostics/postverdict_ablation_raw.jsonl"
FA = f"{A2}/factorial/raw_evidence.jsonl"
CELLS = ("DDD", "DDT", "DTD", "DTT", "TDD", "TDT", "TTD", "TTT")
C1 = {
    "NATIVE_9B": "native_9b_nll", "SOURCE_4B": "source_4b_nll",
    "EMPTY_9B": "empty_9b_nll", "KV_ONLY": "kv_only_nll",
    "KV_GDN_DIRECT": "kv_gdn_direct_nll", "FULL_TRANSLATED": "full_translated_nll",
    "FULL_SHUFFLED": "full_shuffled_nll",
}
C2 = {
    "NATIVE_9B": "native_9b_nll", "SOURCE_4B": "source_4b_nll",
    "EMPTY_9B": "empty_9b_nll", "BASE_STATE": "base_state_nll",
    "JOINT_CORRECTED": "joint_corrected_nll", "JOINT_SHUFFLED": "joint_shuffled_nll",
}
SOURCES: dict[str, str] = {}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read(relative):
    path = ROOT / relative
    require(path.is_file(), f"Required artifact missing: {relative}")
    SOURCES[relative] = sha256(path)
    return parse(path.read_text(encoding="utf-8"))


def read_lines(relative):
    path = ROOT / relative
    require(path.is_file(), f"Required artifact missing: {relative}")
    SOURCES[relative] = sha256(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines and all(line.strip() for line in lines), f"Empty JSONL record: {relative}")
    return [parse(line) for line in lines]


def array(values):
    result = np.asarray(values, dtype=np.float64)
    require(result.ndim == 1 and result.size > 0 and np.isfinite(result).all(),
            "Expected a nonempty finite numeric vector")
    return result


def bootstrap(values, seed, resamples=10000, level=0.95):
    """Independent NumPy implementation of the sealed paired document bootstrap."""
    values = array(values)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return np.quantile(means, [alpha, 1.0 - alpha]).tolist()


def ratio_bootstrap(rows, seed):
    """E002 remaining-gap ratio: resample whole paired rows, then ratio of means."""
    rows = np.asarray(rows, dtype=np.float64)
    require(rows.ndim == 2 and rows.shape[1] == 3 and np.isfinite(rows).all(),
            "RGR requires finite base/corrected/native rows")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(10000):
        sample = rows[rng.integers(0, len(rows), size=len(rows))]
        base = np.mean(sample[:, 0] - sample[:, 2])
        corrected = np.mean(sample[:, 1] - sample[:, 2])
        require(abs(base) >= 1e-12, "Numerically zero RGR bootstrap denominator")
        estimates.append((base - corrected) / base)
    alpha = (1.0 - 0.95) / 2.0
    return np.quantile(estimates, [alpha, 1.0 - alpha]).tolist()


def contrasts(matrix):
    """Rows are documents; columns follow CELLS. D=-1, T=+1."""
    matrix = np.asarray(matrix, dtype=np.float64)
    require(matrix.ndim == 2 and matrix.shape[1] == 8, "Expected eight factorial cells")
    signs = np.array([[1 if c == "T" else -1 for c in cell] for cell in CELLS])
    axes = {
        "KV": (0,), "recurrent": (1,), "convolution": (2,),
        "KV_x_recurrent": (0, 1), "KV_x_convolution": (0, 2),
        "recurrent_x_convolution": (1, 2), "KV_x_recurrent_x_convolution": (0, 1, 2),
    }
    return {name: np.sum(matrix * signs[:, axis].prod(axis=1), axis=1) /
            {1: 4, 2: 2, 3: 1}[len(axis)] for name, axis in axes.items()}


def validate_roster(rows, count, split):
    require(len(rows) == count, f"{split}: expected {count} documents, got {len(rows)}")
    require(len({r["document_id"] for r in rows}) == count, f"{split}: duplicate documents")
    require([r["split_index"] for r in rows] == list(range(count)), f"{split}: unexpected order")
    require(all(r["split"] == split for r in rows), f"{split}: wrong split")
    return rows


def canonical():
    pointer = read(f"{E1}/artifacts/CURRENT_ATTEMPT.json")
    require(pointer["attempt"] == "attempt_001", "E001 canonical attempt changed")
    f1, f2 = read(f"{E1}/FROZEN_MANIFEST.json"), read(f"{E2}/FROZEN_MANIFEST.json")
    for root, frozen in ((E1, f1), (E2, f2)):
        attempts = sorted(p.name for p in (ROOT / root / "artifacts").glob("attempt_*") if p.is_dir())
        require(attempts == ["attempt_001"] and frozen["attempt"] == "attempt_001",
                f"{root}: ambiguous canonical attempt; review required")
    r1, r2 = read(R1), read(R2)
    require(r1["experiment_id"] == "LATENTPORT_E001_HANDOFF", "Unexpected E001 schema")
    require(r2["experiment_id"] == "LATENTPORT_E002_COUPLER", "Unexpected E002 schema")
    require(r1["locked_documents"] == r2["locked_documents"] == 64, "Locked cohort changed")
    require(r2["base_state"] == f2["base_state"] == "TDD", "E002 base selection changed")
    for root, result in ((E1, r1), (E2, r2)):
        require(result["frozen_manifest_hash"] == sha256(ROOT / root / "FROZEN_MANIFEST.json"),
                f"{root}: frozen manifest hash mismatch")
    return r1, r2


def output_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "Output escapes repository")
    require(path.parts[len(ROOT.parts)] in ("derived", "figures", "tables"),
            "Public scripts may write only derived/, figures/, or tables/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(relative, value):
    output_path(relative).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")


def write_csv(relative, fields, rows):
    with output_path(relative).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_sources():
    for path, digest in sorted(SOURCES.items()):
        print(f"SOURCE {path}  sha256={digest}")
