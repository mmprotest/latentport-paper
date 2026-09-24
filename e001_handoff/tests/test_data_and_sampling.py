from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.latentport.e001_handoff.runtime.constants import E001_ROOT, load_preregistration
from experiments.latentport.e001_handoff.runtime.paired_state_collection import kv_positions


def _rows(split: str):
    path = E001_ROOT / "data" / split / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_split_and_corpus_isolation():
    splits = {name: _rows(name) for name in ("fit", "validation", "locked", "long")}
    ids = {name: {row["document_id"] for row in rows} for name, rows in splits.items()}
    for left_index, left in enumerate(ids):
        for right in list(ids)[left_index + 1 :]:
            assert ids[left].isdisjoint(ids[right])
    assert {row["corpus"] for row in splits["fit"]} == {"HuggingFaceFW/fineweb-edu"}
    assert {row["corpus"] for row in splits["locked"]} == {"emozilla/pg19"}


def test_manifest_counts_and_lengths():
    prereg = load_preregistration()
    for split in ("fit", "validation", "locked", "long"):
        rows = _rows(split)
        assert len(rows) == prereg["splits"][split]["documents"]
        expected = prereg["corpora"]["minimum_tokens"][split]
        assert all(len(row["token_ids"]) == expected for row in rows)


def test_kv_position_sampling_is_reproducible_and_includes_boundaries():
    row = _rows("fit")[0]
    prereg = load_preregistration()
    first = kv_positions(row, count=64, length=1024, seed=prereg["seeds"]["kv_position_sampling"])
    second = kv_positions(row, count=64, length=1024, seed=prereg["seeds"]["kv_position_sampling"])
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 64
    assert {0, 127, 255, 383, 511, 639, 767, 895, 1023}.issubset(set(first.tolist()))
