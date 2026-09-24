"""Post-verdict inspection, final machine result, tables, and required figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.latentport.e001_handoff.analysis.integrity import validate_raw_evidence_file
from experiments.latentport.e001_handoff.analysis.metrics import paired_bootstrap_mean_ci
from experiments.latentport.e001_handoff.runtime.constants import ATTEMPT_ROOT, E001_ROOT
from experiments.latentport.e001_handoff.runtime.lock_guard import sha256_file


RAW = ATTEMPT_ROOT / "locked" / "raw_evidence.jsonl"
CANONICAL = ATTEMPT_ROOT / "verdict" / "CANONICAL_4K_RESULT.json"
DERIVED = ATTEMPT_ROOT / "locked" / "derived_statistics.json"
DIAGNOSTICS = ATTEMPT_ROOT / "diagnostics" / "postverdict_analysis.json"
TABLES = ATTEMPT_ROOT / "diagnostics" / "required_tables.json"
RESULT = ATTEMPT_ROOT / "verdict" / "RESULT.json"
CSV = ATTEMPT_ROOT / "locked" / "per_document_metrics.csv"
FIGURES = ATTEMPT_ROOT / "figures"
SEED = 2026083103
CONDITIONS = (
    "NATIVE_9B", "SOURCE_4B", "EMPTY_9B", "KV_ONLY", "KV_GDN_DIRECT",
    "FULL_TRANSLATED", "FULL_SHUFFLED",
)
COLORS = {
    "NATIVE_9B": "#222222", "SOURCE_4B": "#777777", "EMPTY_9B": "#d8a03d",
    "KV_ONLY": "#4c78a8", "KV_GDN_DIRECT": "#59a14f",
    "FULL_TRANSLATED": "#e45756", "FULL_SHUFFLED": "#b279a2",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def rows() -> list[dict[str, Any]]:
    with RAW.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        condition: np.asarray([row["conditions"][condition]["nll"] for row in records])
        for condition in CONDITIONS
    }


def mean_ci(values: np.ndarray) -> list[float]:
    return list(paired_bootstrap_mean_ci(values, resamples=10_000, seed=SEED))


def grouped_quartiles(values: np.ndarray, outcomes: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, 4)
    return [
        {
            "quartile": index + 1,
            "documents": int(len(group)),
            "mean_driver": float(values[group].mean()),
            "mean_full_vs_kv_nll_improvement": float(outcomes[group].mean()),
            "bootstrap_ci": mean_ci(outcomes[group]),
        }
        for index, group in enumerate(groups)
    ]


def translator_layer_errors(name: str, field: str, layer_key: str = "language_layer") -> dict[str, float]:
    metadata = read_json(E001_ROOT / "translators" / "frozen" / f"{name}.json")
    source = metadata.get("maps", metadata.get("layer_head_role_maps"))
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in source:
        grouped[int(record[layer_key])].append(float(record[field]))
    return {str(layer): float(np.mean(values)) for layer, values in sorted(grouped.items())}


def build_diagnostics(records: list[dict[str, Any]], nll: dict[str, np.ndarray]) -> dict[str, Any]:
    native = nll["NATIVE_9B"]
    full_vs_kv = nll["KV_ONLY"] - nll["FULL_TRANSLATED"]
    source_target_gap = nll["SOURCE_4B"] - native
    native_entropy = np.asarray(
        [np.mean(row["conditions"]["NATIVE_9B"]["per_token_entropy"]) for row in records]
    )
    bins = {"1": (0, 1), "2-4": (1, 4), "5-16": (4, 16), "17-64": (16, 64)}
    position_bins = {}
    for label, (start, end) in bins.items():
        position_bins[label] = {}
        native_tokens = np.asarray(
            [row["conditions"]["NATIVE_9B"]["per_token_nll"][start:end] for row in records]
        )
        for condition in CONDITIONS:
            tokens = np.asarray(
                [row["conditions"][condition]["per_token_nll"][start:end] for row in records]
            )
            position_bins[label][condition] = {
                "mean_nll": float(tokens.mean()),
                "mean_delta_nll": float((tokens - native_tokens).mean()),
            }
    convergence = {}
    for checkpoint in (1, 4, 16, 64):
        values = [row["state_convergence"][str(checkpoint)] for row in records]
        convergence[str(checkpoint)] = {
            key: float(np.mean([value[key] for value in values]))
            for key in (
                "mean_gdn_recurrent_normalized_frobenius",
                "mean_gdn_recurrent_cosine",
                "mean_convolution_normalized_l2",
                "mean_new_key_normalized_l2",
                "mean_new_value_normalized_l2",
            )
        }
    recurrent = read_json(E001_ROOT / "translators" / "frozen" / "gdn_recurrent_translator.json")
    recurrent_heads = sorted(
        recurrent["maps"], key=lambda row: row["validation_normalized_frobenius"], reverse=True
    )
    timing_keys = (
        "native_9b_prefill_wall_ms", "source_4b_prefill_wall_ms", "source_state_extraction_ms",
        "host_device_copy_ms", "kv_translation_ms", "gdn_recurrent_translation_ms",
        "gdn_convolution_translation_ms",
    )
    timing = {
        key: {
            "mean_ms": float(np.mean([row["timing"][key] for row in records])),
            "median_ms": float(np.median([row["timing"][key] for row in records])),
            "p95_ms": float(np.quantile([row["timing"][key] for row in records], 0.95)),
        }
        for key in timing_keys
    }
    for key in ("state_install_ms", "bridge_ms"):
        vals = [row["timing"][key]["FULL_TRANSLATED"] for row in records]
        timing[key] = {
            "mean_ms": float(np.mean(vals)), "median_ms": float(np.median(vals)),
            "p95_ms": float(np.quantile(vals, 0.95)),
        }
    timing["translation_budget_for_win_ms"] = {
        "mean_ms": timing["native_9b_prefill_wall_ms"]["mean_ms"]
        - timing["source_4b_prefill_wall_ms"]["mean_ms"]
    }
    timing["prototype_source_plus_translation_install_bridge_ms"] = {
        "mean_ms": sum(
            timing[key]["mean_ms"]
            for key in (
                "source_4b_prefill_wall_ms", "source_state_extraction_ms", "host_device_copy_ms",
                "kv_translation_ms", "gdn_recurrent_translation_ms",
                "gdn_convolution_translation_ms", "state_install_ms", "bridge_ms",
            )
        )
    }
    return {
        "analysis_is_post_verdict": True,
        "raw_evidence_sha256": sha256_file(RAW),
        "source_target_gap_quartiles": grouped_quartiles(source_target_gap, full_vs_kv),
        "native_entropy_quartiles": grouped_quartiles(native_entropy, full_vs_kv),
        "continuation_position_bins": position_bins,
        "state_convergence": convergence,
        "gdn_recurrent_validation_error_by_layer": translator_layer_errors(
            "gdn_recurrent_translator", "validation_normalized_frobenius"
        ),
        "gdn_convolution_validation_error_by_layer": translator_layer_errors(
            "gdn_convolution_translator", "validation_normalized_l2"
        ),
        "kv_key_validation_error_by_layer": {
            str(layer): float(np.mean([row["validation_normalized_l2"] for row in read_json(
                E001_ROOT / "translators" / "frozen" / "kv_translator.json"
            )["layer_head_role_maps"] if row["language_layer"] == layer and row["role"] == "K"]))
            for layer in (3, 7, 11, 15, 19, 23, 27, 31)
        },
        "kv_value_validation_error_by_layer": {
            str(layer): float(np.mean([row["validation_normalized_l2"] for row in read_json(
                E001_ROOT / "translators" / "frozen" / "kv_translator.json"
            )["layer_head_role_maps"] if row["language_layer"] == layer and row["role"] == "V"]))
            for layer in (3, 7, 11, 15, 19, 23, 27, 31)
        },
        "worst_20_gdn_heads": recurrent_heads[:20],
        "translator_component_mean_validation_error": {
            "gdn_recurrent": float(recurrent["mean_validation_normalized_frobenius"]),
            "gdn_recurrent_direct_copy": float(
                recurrent["mean_direct_copy_validation_normalized_frobenius"]
            ),
            "kv_key": float(np.mean([value for value in translator_layer_errors(
                "kv_translator", "validation_normalized_l2"
            ).values()])),
            "gdn_convolution": float(np.mean([value for value in translator_layer_errors(
                "gdn_convolution_translator", "validation_normalized_l2"
            ).values()])),
        },
        "timing": timing,
        "peak_vram_bytes": {
            stage: int(max(row["peak_vram_bytes"][stage] for row in records))
            for stage in ("source_pass", "translation", "target_pass")
        },
        "bytes_moved_mean": {
            "translation_host_to_device": float(np.mean([
                row["bytes_moved"]["translation_host_to_device"] for row in records
            ])),
            "translation_device_to_host": float(np.mean([
                row["bytes_moved"]["translation_device_to_host"] for row in records
            ])),
        },
    }


def build_tables(canonical: dict, stats: dict, diagnostics: dict) -> dict[str, Any]:
    architecture = read_json(ATTEMPT_ROOT / "implementation" / "architecture_correspondence.json")
    restore4 = read_json(ATTEMPT_ROOT / "state_validation" / "same_model_restore_4b.json")
    restore9 = read_json(ATTEMPT_ROOT / "state_validation" / "same_model_restore_9b.json")
    translator_rows = []
    for name, granularity, samples in (
        ("kv_translator", "layer / KV head / K or V", 8192),
        ("gdn_recurrent_translator", "GDN layer / value head", 1024),
        ("gdn_convolution_translator", "GDN layer / packed component / head", 4096),
    ):
        item = read_json(E001_ROOT / "translators" / "frozen" / f"{name}.json")
        translator_rows.append(
            {
                "component": name,
                "granularity": granularity,
                "parameters": item["parameter_count_weights"],
                "fit_samples": samples,
                "validation_rule": "minimum frozen validation normalized error over registered ridge/normalization grid",
            }
        )
    return {
        "architecture_correspondence": [
            {"component": "language layers", "4B": 32, "9B": 32, "correspondence": "identity", "canonical_action": "map same index"},
            {"component": "Gated DeltaNet", "4B": "24; [32,128,128]", "9B": "24; [32,128,128]", "correspondence": "exact geometry", "canonical_action": "bilinear per layer/head"},
            {"component": "GDN convolution", "4B": "[8192,4]", "9B": "[8192,4]", "correspondence": "exact packed geometry", "canonical_action": "ridge per packed component/head"},
            {"component": "full-attention KV", "4B": "8; 4x256", "9B": "8; 4x256", "correspondence": "exact geometry", "canonical_action": "de-rotated ridge per layer/head/role"},
        ],
        "same_model_restore": [
            {"model": "4B", "contexts": 16, "top1_agreement": restore4["top1_agreement"], "min_logit_cosine": restore4["minimum_next_token_logit_cosine"], "max_abs_diff": restore4["maximum_absolute_logit_difference"], "pass": restore4["pass"]},
            {"model": "9B", "contexts": 16, "top1_agreement": restore9["top1_agreement"], "min_logit_cosine": restore9["minimum_next_token_logit_cosine"], "max_abs_diff": restore9["maximum_absolute_logit_difference"], "pass": restore9["pass"]},
        ],
        "translator_inventory": translator_rows,
        "locked_fidelity": [
            {"condition": condition, **stats["condition_summary"][condition]}
            for condition in CONDITIONS
        ],
        "h1_comparison": {
            "KV_ONLY_delta_nll": canonical["kv_only_delta_nll"],
            "FULL_TRANSLATED_delta_nll": canonical["full_translated_delta_nll"],
            "delta": canonical["full_vs_kv_delta"],
            "bootstrap_ci": canonical["full_vs_kv_bootstrap_ci"],
        },
        "target_quality_retention": {
            "split": "LOCKED_4K", "tqr": canonical["tqr"],
            "delta_nll": canonical["full_translated_delta_nll"],
            "pass_threshold": "TQR >= 0.75 and DeltaNLL <= 0.20", "pass": False,
        },
        "timing": diagnostics["timing"],
    }


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def make_figures(records: list[dict[str, Any]], nll: dict[str, np.ndarray], canonical: dict, diagnostics: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    native = nll["NATIVE_9B"]
    display = ["EMPTY_9B", "KV_ONLY", "KV_GDN_DIRECT", "FULL_TRANSLATED", "FULL_SHUFFLED"]
    means = [(nll[name] - native).mean() for name in display]
    cis = [mean_ci(nll[name] - native) for name in display]
    yerr = [[mean - ci[0] for mean, ci in zip(means, cis)], [ci[1] - mean for mean, ci in zip(means, cis)]]
    plt.figure(figsize=(9, 5)); plt.bar(display, means, color=[COLORS[name] for name in display], yerr=yerr, capsize=4)
    plt.axhline(0, color="black", linewidth=1); plt.ylabel("Excess NLL vs native 9B (nats/token)"); plt.title("LOCKED 4K handoff fidelity (95% paired bootstrap CI)"); plt.xticks(rotation=20)
    savefig("delta_nll_conditions.png")

    x = nll["KV_ONLY"] - native; y = nll["FULL_TRANSLATED"] - native
    lim = (min(x.min(), y.min()) - .05, max(x.max(), y.max()) + .05)
    plt.figure(figsize=(6, 6)); plt.scatter(x, y, alpha=.75, color=COLORS["FULL_TRANSLATED"]); plt.plot(lim, lim, "--", color="gray")
    plt.xlim(lim); plt.ylim(lim); plt.xlabel("KV-only DeltaNLL"); plt.ylabel("Full-translated DeltaNLL"); plt.title("Every point below the diagonal favors recurrent-state translation")
    savefig("kv_vs_full_translation.png")

    plt.figure(figsize=(6, 5)); plt.bar(["FULL_TRANSLATED"], [canonical["tqr"]], color=COLORS["FULL_TRANSLATED"]); plt.axhline(.75, color="#59a14f", linestyle="--", label="Full-state threshold 0.75"); plt.axhline(0, color="black", linewidth=1); plt.ylim(min(-.6, canonical["tqr"]-.1), 1.05); plt.ylabel("Aggregate target-quality retention"); plt.title("Target-quality retention"); plt.legend()
    savefig("target_quality_retention.png")

    positions = np.arange(1, 65)
    plt.figure(figsize=(9, 5))
    for name in ("NATIVE_9B", "KV_ONLY", "FULL_TRANSLATED", "FULL_SHUFFLED"):
        curve = np.asarray([row["conditions"][name]["per_token_nll"] for row in records]).mean(axis=0)
        plt.plot(positions, curve, label=name, color=COLORS[name])
    plt.xlabel("Continuation target position"); plt.ylabel("Mean NLL"); plt.title("Continuation fidelity after handoff"); plt.legend()
    savefig("native_vs_handoff_position_curve.png")

    plt.figure(figsize=(6, 6)); plt.scatter(nll["FULL_TRANSLATED"], nll["FULL_SHUFFLED"], alpha=.75, color=COLORS["FULL_SHUFFLED"]); lim=(min(nll["FULL_TRANSLATED"].min(), nll["FULL_SHUFFLED"].min())-.05,max(nll["FULL_TRANSLATED"].max(), nll["FULL_SHUFFLED"].max())+.05); plt.plot(lim,lim,"--",color="gray"); plt.xlim(lim); plt.ylim(lim); plt.xlabel("Correct translated state NLL"); plt.ylabel("Shuffled translated state NLL"); plt.title("Content specificity of transferred state")
    savefig("shuffled_state_control.png")

    gdn = diagnostics["gdn_recurrent_validation_error_by_layer"]
    plt.figure(figsize=(9, 5)); plt.bar([int(k) for k in gdn], list(gdn.values()), color="#e45756"); plt.xlabel("GDN language-layer index"); plt.ylabel("Validation normalized Frobenius error"); plt.title("Bilinear recurrent-state translation error by layer")
    savefig("gdn_translation_error_by_layer.png")

    conv = diagnostics["state_convergence"]; points=[1,4,16,64]
    plt.figure(figsize=(8, 5)); plt.plot(points,[conv[str(p)]["mean_gdn_recurrent_normalized_frobenius"] for p in points],"o-",label="GDN recurrent"); plt.plot(points,[conv[str(p)]["mean_convolution_normalized_l2"] for p in points],"o-",label="GDN convolution"); plt.plot(points,[conv[str(p)]["mean_new_value_normalized_l2"] for p in points],"o-",label="New target V entries"); plt.xscale("log",base=2); plt.xticks(points,points); plt.xlabel("New tokens after bridge"); plt.ylabel("Normalized error vs native state"); plt.title("Does target inference repair translated-state error?"); plt.legend()
    savefig("state_convergence_over_tokens.png")

    keys=diagnostics["kv_key_validation_error_by_layer"]; vals=diagnostics["kv_value_validation_error_by_layer"]; layers=[int(k) for k in keys]; width=.35
    plt.figure(figsize=(9, 5)); plt.bar(np.asarray(layers)-width/2,list(keys.values()),width,label="K"); plt.bar(np.asarray(layers)+width/2,list(vals.values()),width,label="V"); plt.xlabel("Full-attention language-layer index"); plt.ylabel("Validation normalized L2 error"); plt.title("KV translation error by layer"); plt.legend()
    savefig("kv_translation_error_by_layer.png")

    timing=diagnostics["timing"]; labels=["4B prefill","Extract","Host/device","KV map","GDN recurrent","GDN conv","Install","Bridge"]; keys=["source_4b_prefill_wall_ms","source_state_extraction_ms","host_device_copy_ms","kv_translation_ms","gdn_recurrent_translation_ms","gdn_convolution_translation_ms","state_install_ms","bridge_ms"]; values=[timing[key]["mean_ms"] for key in keys]
    plt.figure(figsize=(10, 5)); plt.bar(labels,values,color="#4c78a8"); plt.axhline(timing["native_9b_prefill_wall_ms"]["mean_ms"],color="#e45756",linestyle="--",label="Native 9B prefill"); plt.ylabel("Mean prototype wall time (ms)"); plt.title("Prototype timing breakdown; not a production speed claim"); plt.xticks(rotation=25); plt.legend()
    savefig("timing_breakdown.png")

    metrics=[("Full vs KV CI lower",canonical["full_vs_kv_bootstrap_ci"][0],0,"PASS"),("Mean improvement fraction",canonical["full_vs_kv_improvement_fraction"],.25,"PASS"),("Full DeltaNLL",canonical["full_translated_delta_nll"],.20,"FAIL"),("TQR",canonical["tqr"],.75,"FAIL"),("Full vs shuffled CI lower",canonical["full_vs_shuffled_bootstrap_ci"][0],0,"PASS")]
    plt.figure(figsize=(10, 4.8)); plt.axis("off"); plt.title("Canonical verdict: RECURRENT_STATE_TRANSLATABLE",fontsize=16,weight="bold",pad=16)
    table=plt.table(cellText=[[name,f"{value:.3f}",f"{threshold:.2f}",status] for name,value,threshold,status in metrics],colLabels=["Gate","Observed","Threshold","Result"],loc="center",cellLoc="center",colWidths=[.45,.17,.17,.16]); table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1,1.7)
    for row, (_,_,_,status) in enumerate(metrics, start=1): table[(row,3)].set_facecolor("#b7e1a1" if status=="PASS" else "#f4b6b2")
    savefig("verdict_summary.png")


def main() -> None:
    integrity = validate_raw_evidence_file(RAW)
    if not integrity["pass"]:
        raise RuntimeError(integrity["errors"][:10])
    records = rows(); nll = arrays(records); canonical = read_json(CANONICAL); stats = read_json(DERIVED)
    diagnostics = build_diagnostics(records, nll)
    tables = build_tables(canonical, stats, diagnostics)
    result = dict(canonical)
    result.update(
        {
            "result_stage": "FINAL_VERDICT_FROZEN_BEFORE_NARRATIVE",
            "e001_status": "COMPLETE",
            "canonical_verdict": canonical["canonical_4k_verdict"],
            "long_test_run": False,
            "long_pass": False,
            "long_not_run_reason": "4K verdict did not reach FULL_STATE_HANDOFF",
            "oracle_diagnostics_run": False,
            "oracle_not_run_reason": "FULL_TRANSLATED passed the frozen primary FULL-vs-KV gate",
            "postverdict_diagnostics_sha256": None,
        }
    )
    if not DIAGNOSTICS.exists():
        write_once(DIAGNOSTICS, diagnostics)
    result["postverdict_diagnostics_sha256"] = sha256_file(DIAGNOSTICS)
    if not TABLES.exists():
        write_once(TABLES, tables)
    if not RESULT.exists():
        write_once(RESULT, result)
    if not CSV.exists():
        with CSV.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["document_id", *[f"{name}_nll" for name in CONDITIONS], "full_vs_kv_improvement"])
            for index, row in enumerate(records):
                writer.writerow([row["document_id"], *[nll[name][index] for name in CONDITIONS], nll["KV_ONLY"][index]-nll["FULL_TRANSLATED"][index]])
    make_figures(records, nll, canonical, diagnostics)
    print(json.dumps({"result": str(RESULT), "figures": 10, "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()
