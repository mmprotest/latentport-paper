"""Run and record all 52 mandatory checks before FROZEN_MANIFEST creation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from experiments.latentport.e002_coupler.runtime.constants import ATTEMPT_ROOT, E002_ROOT
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


class _Collector:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.skipped: list[str] = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.when != "call":
            return
        if report.passed:
            self.passed.append(report.nodeid)
        elif report.failed:
            self.failed.append(report.nodeid)
        elif report.skipped:
            self.skipped.append(report.nodeid)


def main() -> None:
    if (ATTEMPT_ROOT / "locked" / "LOCKED_RUN_STARTED.json").exists():
        raise PermissionError("pre-LOCKED tests cannot be recorded after LOCKED begins")
    output = ATTEMPT_ROOT / "implementation" / "prelocked_test_results.json"
    if output.exists():
        raise FileExistsError("pre-LOCKED test evidence already exists")
    collector = _Collector()
    code = pytest.main(
        [str(E002_ROOT / "tests" / "test_prelocked.py"), "-q", "--disable-warnings"],
        plugins=[collector],
    )
    if code != pytest.ExitCode.OK or collector.failed or collector.skipped or len(collector.passed) != 52:
        raise RuntimeError(
            f"pre-LOCKED test gate failed: code={int(code)}, passed={len(collector.passed)}, "
            f"failed={collector.failed}, skipped={collector.skipped}"
        )
    record = {
        "experiment_id": "LATENTPORT_E002_COUPLER",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "tests_required": 52,
        "tests_passed": len(collector.passed),
        "failed": collector.failed,
        "skipped": collector.skipped,
        "node_ids": collector.passed,
        "locked_run_started": False,
    }
    write_json_once(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
