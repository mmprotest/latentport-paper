"""Canonical E002 fidelity metrics."""

from __future__ import annotations


def delta_nll(condition_nll: float, native_nll: float) -> float:
    return float(condition_nll - native_nll)


def native_context_recovery(empty_nll: float, condition_nll: float, native_nll: float) -> float:
    denominator = empty_nll - native_nll
    if denominator == 0:
        raise ZeroDivisionError("NCR denominator is zero")
    return float((empty_nll - condition_nll) / denominator)


def tqr(source_nll: float, condition_nll: float, native_nll: float) -> float | None:
    denominator = source_nll - native_nll
    if denominator <= 0:
        return None
    return float((source_nll - condition_nll) / denominator)


def remaining_gap_reduction(base_delta_nll: float, corrected_delta_nll: float) -> float | None:
    if base_delta_nll <= 0:
        return None
    return float((base_delta_nll - corrected_delta_nll) / base_delta_nll)

