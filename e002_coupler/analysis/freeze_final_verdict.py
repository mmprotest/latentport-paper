"""Freeze the final canonical verdict after conditional LONG and before ablations."""

from __future__ import annotations

import json

from experiments.latentport.e002_coupler.analysis.verdict import final_verdict, next_hypothesis
from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT
from experiments.latentport.e002_coupler.runtime.lock_guard import sha256_file
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


def main() -> None:
    verdict_4k_path = ATTEMPT_ROOT / "verdict" / "4k_verdict.json"
    record_4k = json.loads(verdict_4k_path.read_text(encoding="utf-8"))
    verdict_4k = record_4k["canonical_4k_verdict"]
    long_path = ATTEMPT_ROOT / "long_context" / "long_result.json"
    if verdict_4k == "NEAR_NATIVE_HANDOFF" and not long_path.is_file():
        raise RuntimeError("conditional LONG must complete before final verdict freeze")
    if verdict_4k != "NEAR_NATIVE_HANDOFF" and long_path.exists():
        raise RuntimeError("ineligible LONG evidence exists before final verdict freeze")
    long = json.loads(long_path.read_text(encoding="utf-8")) if long_path.exists() else None
    verdict = final_verdict(verdict_4k, long_pass=long["long_pass"] if long else None)
    output = {
        "freeze_state": "FINAL_CANONICAL_VERDICT_FROZEN_BEFORE_POSTVERDICT_ABLATIONS",
        "canonical_4k_verdict": verdict_4k,
        "canonical_verdict": verdict,
        "long_test_run": long is not None,
        "long_pass": long["long_pass"] if long else None,
        "four_k_verdict_sha256": sha256_file(verdict_4k_path),
        "long_result_sha256": sha256_file(long_path) if long else None,
        "post_verdict_ablations_started": False,
        "next_hypothesis": next_hypothesis(verdict),
    }
    write_json_once(ATTEMPT_ROOT / "verdict" / "final_verdict.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
