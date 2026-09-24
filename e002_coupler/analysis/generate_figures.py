"""Generate the ten frozen-verdict E002 scientific figures (and conditional headline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.latentport.e002_coupler.analysis.metrics import native_context_recovery
from experiments.latentport.e002_coupler.analysis.statistics import bootstrap_mean_ci
from experiments.latentport.e002_coupler.analysis.verdict import final_verdict
from experiments.latentport.e002_coupler.runtime.constants import (
    ATTEMPT_ROOT,
    FACTORIAL_CONDITIONS,
)
from experiments.latentport.e002_coupler.runtime.state_io import write_json_once


FIGURES = ATTEMPT_ROOT / "figures"
COLORS = {
    "native": "#2D6A4F",
    "source": "#6C757D",
    "base": "#457B9D",
    "corrected": "#E76F51",
    "shuffled": "#8E44AD",
    "validation": "#457B9D",
    "locked": "#E76F51",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw() -> list[dict[str, Any]]:
    with (ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl").open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(ax, *, ylabel: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def _errorbar_mean(ax, x: float, values: np.ndarray, color: str, label: str) -> None:
    low, high = bootstrap_mean_ci(values)
    mean = float(values.mean())
    ax.errorbar(
        [x],
        [mean],
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        markersize=8,
        capsize=5,
        color=color,
        label=label,
    )


def _final_verdict() -> str:
    return _json(ATTEMPT_ROOT / "verdict" / "final_verdict.json")["canonical_verdict"]


def generate() -> list[str]:
    metrics = _json(ATTEMPT_ROOT / "locked" / "locked_metrics.json")
    factorial = _json(ATTEMPT_ROOT / "factorial" / "base_state_selection.json")
    rows = _raw()
    condition = metrics["conditions"]
    created = []

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    values = [factorial["condition_metrics"][name]["mean_nll"] for name in FACTORIAL_CONDITIONS]
    colors = [COLORS["base"] if name == factorial["selected_state"] else "#A8B8C8" for name in FACTORIAL_CONDITIONS]
    ax.bar(FACTORIAL_CONDITIONS, values, color=colors)
    native_validation = np.mean(
        [
            factorial["condition_metrics"][name]["mean_nll"]
            - factorial["condition_metrics"][name]["mean_delta_nll"]
            for name in FACTORIAL_CONDITIONS
        ]
    )
    ax.axhline(native_validation, color=COLORS["native"], linestyle="--", label="Native 9B")
    ax.set_title("Validation factorial: direct versus translated components")
    _style(ax, ylabel="NLL (nats/token; lower is better)")
    ax.legend(frameon=False)
    _save(fig, "factorial_condition_nll.png")
    created.append("factorial_condition_nll.png")

    effects = ("KV", "recurrent", "convolution", "KV_x_recurrent", "KV_x_convolution", "recurrent_x_convolution")
    labels = ("KV", "Recurrent", "Conv", "KV×R", "KV×C", "R×C")
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(effects))
    for offset, source, color, label in (
        (-0.12, factorial["interaction_metrics"]["effects"], COLORS["validation"], "Validation"),
        (0.12, metrics["factorial_interactions"]["effects"], COLORS["locked"], "LOCKED"),
    ):
        estimates = np.asarray([source[name]["estimate"] for name in effects])
        cis = np.asarray([source[name]["bootstrap_ci"] for name in effects])
        ax.errorbar(x + offset, estimates, yerr=np.vstack((estimates - cis[:, 0], cis[:, 1] - estimates)), fmt="o", capsize=4, color=color, label=label)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_title("Factorial main effects and interactions")
    _style(ax, ylabel="Effect on DeltaNLL (T minus D)")
    ax.legend(frameon=False)
    _save(fig, "factorial_interactions.png")
    created.append("factorial_interactions.png")

    native = np.asarray([row["conditions"]["NATIVE_9B"]["nll"] for row in rows])
    base_delta = np.asarray([row["conditions"]["BASE_STATE"]["nll"] for row in rows]) - native
    corrected_delta = np.asarray([row["conditions"]["JOINT_CORRECTED"]["nll"] for row in rows]) - native
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _errorbar_mean(ax, 0, base_delta, COLORS["base"], "Base")
    _errorbar_mean(ax, 1, corrected_delta, COLORS["corrected"], "Joint corrected")
    ax.set_xticks([0, 1], [metrics["base_state"], "Joint corrected"])
    ax.axhline(0, color=COLORS["native"], linestyle="--", label="Native 9B")
    ax.set_title("Remaining target-fidelity gap")
    _style(ax, ylabel="DeltaNLL to native 9B")
    _save(fig, "base_vs_corrected_delta_nll.png")
    created.append("base_vs_corrected_delta_nll.png")

    empty_nll = condition["EMPTY_9B"]["nll"]
    native_nll = condition["NATIVE_9B"]["nll"]
    ncr_values = [
        native_context_recovery(empty_nll, condition[name]["nll"], native_nll)
        for name in ("BASE_STATE", "JOINT_CORRECTED", "JOINT_SHUFFLED")
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(["Base", "Joint corrected", "Shuffled"], ncr_values, color=[COLORS["base"], COLORS["corrected"], COLORS["shuffled"]])
    ax.axhline(1, color=COLORS["native"], linestyle="--", label="All native context benefit")
    ax.set_title("Native Context Recovery (raw, unclipped)")
    _style(ax, ylabel="NCR")
    ax.legend(frameon=False)
    _save(fig, "native_context_recovery.png")
    created.append("native_context_recovery.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    source_values = np.asarray([row["conditions"]["SOURCE_4B"]["nll"] for row in rows])
    corrected_values = np.asarray([row["conditions"]["JOINT_CORRECTED"]["nll"] for row in rows])
    _errorbar_mean(ax, 0, source_values, COLORS["source"], "Source 4B")
    _errorbar_mean(ax, 1, corrected_values, COLORS["corrected"], "Joint corrected")
    ax.axhline(condition["NATIVE_9B"]["nll"], color=COLORS["native"], linestyle="--", label="Native 9B")
    ax.set_xticks([0, 1], ["Source 4B", "Joint corrected"])
    ax.set_title("Practical source-switch comparison")
    _style(ax, ylabel="NLL (nats/token)")
    ax.legend(frameon=False)
    _save(fig, "source_vs_corrected.png")
    created.append("source_vs_corrected.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    shuffled_values = np.asarray([row["conditions"]["JOINT_SHUFFLED"]["nll"] for row in rows])
    _errorbar_mean(ax, 0, corrected_values, COLORS["corrected"], "Corrected")
    _errorbar_mean(ax, 1, shuffled_values, COLORS["shuffled"], "Shuffled")
    ax.set_xticks([0, 1], ["Corrected", "Donor-shuffled"])
    ax.set_title("Content specificity of corrected state")
    _style(ax, ylabel="NLL (nats/token)")
    _save(fig, "corrected_vs_shuffled.png")
    created.append("corrected_vs_shuffled.png")

    layer_records = metrics["correction_magnitude"]["layer_records"]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for component, color in (("K", COLORS["base"]), ("R", COLORS["corrected"]), ("C", "#2A9D8F")):
        layer_ids = sorted({row["layer_index"] for row in layer_records if row["component"] == component})
        values = [np.mean([row["relative_norm"] for row in layer_records if row["component"] == component and row["layer_index"] == layer]) for layer in layer_ids]
        ax.plot(layer_ids, values, marker="o", markersize=3.5, label=component, color=color)
    ax.set_title("Relative correction magnitude by layer")
    ax.set_xlabel("Target layer")
    _style(ax, ylabel="||Delta|| / ||BASE||")
    ax.legend(frameon=False, ncol=3)
    _save(fig, "correction_magnitude_by_layer.png")
    created.append("correction_magnitude_by_layer.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    checkpoints = [1, 4, 16, 64, 256]
    for name, label, color in (("BASE_STATE", "Base", COLORS["base"]), ("JOINT_CORRECTED", "Joint corrected", COLORS["corrected"])):
        values = [metrics["state_repair"][str(token)][name]["recurrent_normalized_frobenius"] for token in checkpoints]
        ax.plot(checkpoints, values, marker="o", label=label, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(checkpoints, [str(value) for value in checkpoints])
    ax.set_xlabel("Tokens processed by target after handoff")
    ax.set_title("Target-side recurrent-state self-repair")
    _style(ax, ylabel="Normalized Frobenius error to native")
    ax.legend(frameon=False)
    _save(fig, "state_repair_to_256_tokens.png")
    created.append("state_repair_to_256_tokens.png")

    timing = metrics["timing"]
    stages = ["4B prefill", "Base build", "Correction", "Install", "Bridge", "9B prefill"]
    values = [timing["source_4b_prefill_ms"], timing["base_construction_ms"], timing["joint_correction_ms"], timing["state_install_ms"], timing["bridge_ms"], timing["native_9b_prefill_ms"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(stages, values, color=[COLORS["source"], COLORS["base"], COLORS["corrected"], "#F4A261", "#E9C46A", COLORS["native"]])
    ax.set_title("Prototype latency breakdown (document medians)")
    ax.tick_params(axis="x", rotation=20)
    _style(ax, ylabel="Wall-clock latency (ms)")
    _save(fig, "timing_breakdown.png")
    created.append("timing_breakdown.png")

    verdict = _final_verdict()
    comparison = metrics["comparisons"]["base_minus_corrected"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axis("off")
    text = (
        f"Canonical verdict\n{verdict}\n\n"
        f"Base state: {metrics['base_state']}\n"
        f"Corrected DeltaNLL: {metrics['corrected_delta_nll']:.4f}\n"
        f"Remaining-gap reduction: {metrics['remaining_gap_reduction']:.1%}\n"
        f"NCR: {metrics['native_context_recovery']:.3f}\n"
        f"Base − corrected NLL: {comparison['mean_difference']:.4f} "
        f"(95% CI {comparison['bootstrap_ci'][0]:.4f}, {comparison['bootstrap_ci'][1]:.4f})"
    )
    ax.text(0.04, 0.92, text, va="top", ha="left", fontsize=15, linespacing=1.45, color="#222222")
    ax.text(0.04, 0.05, "Fresh 4K LOCKED corpus · 64 documents · document-level bootstrap", fontsize=10, color="#666666")
    _save(fig, "verdict_summary.png")
    created.append("verdict_summary.png")

    if verdict in ("NEAR_NATIVE_HANDOFF", "LONG_CONTEXT_HANDOFF"):
        headline_names = ("NATIVE_9B", "SOURCE_4B", "BASE_STATE", "JOINT_CORRECTED", "JOINT_SHUFFLED")
        headline_labels = ("Native 9B", "Source 4B", f"Base ({metrics['base_state']})", "Joint corrected", "Shuffled")
        headline_values = [condition[name]["nll"] - native_nll for name in headline_names]
        fig, ax = plt.subplots(figsize=(9.4, 5.0))
        ax.bar(headline_labels, headline_values, color=[COLORS["native"], COLORS["source"], COLORS["base"], COLORS["corrected"], COLORS["shuffled"]])
        ax.set_title("Qwen3.5-4B → 9B recurrent-state handoff")
        ax.tick_params(axis="x", rotation=16)
        _style(ax, ylabel="DeltaNLL to native 9B")
        _save(fig, "headline_result.png")
        created.append("headline_result.png")

    chart_map = {
        "generated_after_4k_verdict_freeze": True,
        "canonical_verdict": verdict,
        "charts": {
            "factorial_condition_nll.png": "Ranks all eight validation factorial cells and marks native 9B.",
            "factorial_interactions.png": "Compares validation and LOCKED within-document factorial contrasts.",
            "base_vs_corrected_delta_nll.png": "Shows primary paired target-gap comparison with document bootstrap intervals.",
            "native_context_recovery.png": "Shows raw unclipped NCR for base, corrected, and shuffled state.",
            "source_vs_corrected.png": "Shows whether switching beats continued source-model inference.",
            "corrected_vs_shuffled.png": "Tests content specificity against no-fixed-point donor rotation.",
            "correction_magnitude_by_layer.png": "Shows relative residual magnitude by component and depth.",
            "state_repair_to_256_tokens.png": "Shows recurrent error trajectories through 256 target tokens.",
            "timing_breakdown.png": "Shows measured prototype latency stages without a production-speedup claim.",
            "verdict_summary.png": "States the frozen verdict and its central quantitative evidence.",
        },
    }
    if "headline_result.png" in created:
        chart_map["charts"]["headline_result.png"] = "Near-native headline comparison required by the protocol."
    write_json_once(FIGURES / "chart_map.json", chart_map)
    return created


def main() -> None:
    print(json.dumps({"created": generate()}, indent=2))


if __name__ == "__main__":
    main()
