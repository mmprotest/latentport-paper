"""Deterministic implementation-only token streams for the Phase A gate."""

from __future__ import annotations

import hashlib


def build_restore_context(tokenizer, context_index: int, required_tokens: int) -> list[int]:
    """Produce model-agnostic prose without drawing from FIT/LOCKED corpora."""

    vocabulary = (
        "amber bridge cedar delta ember field granite harbor iris juniper kernel lantern "
        "meadow nexus orbit prism quartz river signal timber umber vector willow xenon yarrow zephyr"
    ).split()
    lines: list[str] = []
    segment = 0
    while True:
        rotated = vocabulary[segment % len(vocabulary) :] + vocabulary[: segment % len(vocabulary)]
        lines.append(
            f"Implementation validation stream {context_index}, segment {segment}. "
            f"The recorder observes {' '.join(rotated)}. "
            "Every numbered observation is deterministic and exists only to exercise cache restoration.\n"
        )
        if segment % 64 == 63:
            token_ids = tokenizer.encode("".join(lines), add_special_tokens=False)
            if len(token_ids) >= required_tokens:
                return token_ids[:required_tokens]
        segment += 1


def token_stream_sha256(token_ids: list[int]) -> str:
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

