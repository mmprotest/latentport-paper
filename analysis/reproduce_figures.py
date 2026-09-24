"""Rebuild public quantitative figures exclusively from derived/."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, bootstrap, csv_rows, output_path, parse, require

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.hashsalt": "latentport-paper",
})


def data(name):
    return parse((ROOT / "derived" / name).read_text(encoding="utf-8"))


def save(fig, name):
    # Suppress wall-clock metadata so identical inputs produce identical bytes.
    fig.savefig(output_path(f"figures/{name}.pdf"),
                metadata={"Creator": "latentport-paper", "CreationDate": None, "ModDate": None})
    fig.savefig(output_path(f"figures/{name}.png"), dpi=300,
                metadata={"Software": "latentport-paper"})
    plt.close(fig)
    print(f"WROTE figures/{name}.pdf and .png")


def excess(summary, conditions):
    native = summary["conditions"]["NATIVE_9B"]["estimate"]
    return [summary["conditions"][c]["estimate"] - native for c in conditions]


def bars(ax, labels, values):
    ax.barh(range(len(values)), values, height=0.56)
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    upper = max(values) * 1.28
    ax.set_xlim(0, upper if upper > 0 else 1)
    for i, value in enumerate(values):
        ax.text(value + upper * 0.02, i, f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Excess NLL over native 9B (nats/token)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)


def hero(h):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.37, top=0.78, wspace=0.75)
    fig.suptitle("Persistent state adds continuation benefit beyond KV-only transfer", fontsize=15, y=0.97)
    fig.text(0.5, 0.90, f'{h["route"]}  |  4,096-token prefix  |  64 teacher-forced targets per document',
             ha="center", fontsize=11)
    bars(axes[0], ["Empty 9B", "KV-only", "KV + recurrent /\nconvolution state", "Native 9B"],
         excess(h["E001"], ["EMPTY_9B", "KV_ONLY", "FULL_TRANSLATED", "NATIVE_9B"]))
    axes[0].set_title("E001 · 64 PG19 documents", fontsize=12, pad=15)
    bars(axes[1], ["Continued 4B", "Uncorrected\nTDD base", "Corrected 9B", "Native 9B"],
         excess(h["E002"], ["SOURCE_4B", "BASE_STATE", "JOINT_CORRECTED", "NATIVE_9B"]))
    axes[1].set_title("E002 · 64 FineWeb-Edu documents", fontsize=12, pad=15)
    a = h["E001"]["comparisons"]["full_vs_kv"]
    b = h["E002"]["comparisons"]["base_minus_corrected"]
    fig.text(0.08, 0.23,
             f'Full state lowers NLL by {a["estimate"]:.4f} vs KV-only.\n'
             f'Reproduced paired 95% CI [{a["ci95"][0]:.4f}, {a["ci95"][1]:.4f}].',
             fontsize=10, linespacing=1.6)
    fig.text(0.56, 0.23,
             f'Correction lowers NLL by {b["estimate"]:.4f} vs the base.\n'
             f'Reproduced paired 95% CI [{b["ci95"][0]:.4f}, {b["ci95"][1]:.4f}].',
             fontsize=10, linespacing=1.6)
    fig.text(0.08, 0.10, "Panel scales and corpora differ. Bars are split means; intervals concern paired differences.",
             fontsize=9)
    fig.text(0.08, 0.05, "Means and paired intervals are reproduced from the included frozen observations.",
             fontsize=9)
    save(fig, "main_handoff_comparison")


def factorial(h):
    cells = ("DDD", "DDT", "DTD", "DTT", "TDD", "TDT", "TTD", "TTT")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.21, top=0.80, wspace=0.3)
    max_value = 0
    for ax, filename, title in zip(axes,
            ("e002_factorial_validation.csv", "e002_per_document.csv"),
            ("Validation · 32 documents", "LOCKED · 64 documents")):
        rows = csv_rows(f"derived/{filename}")
        native = np.array([float(r["native_9b_nll"]) for r in rows])
        for index, cell in enumerate(cells):
            values = np.array([float(r[f"{cell.lower()}_nll"]) for r in rows]) - native
            ci = bootstrap(values, h["E002"]["bootstrap_seed"])
            mean = values.mean()
            ax.errorbar(mean, index, xerr=[[mean-ci[0]], [ci[1]-mean]], fmt="o", capsize=3)
            max_value = max(max_value, ci[1])
        ax.set_yticks(range(8), cells)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Excess NLL (nats/token)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlim(0, max_value * 1.1)
    fig.suptitle("E002 · translated KV and direct persistent state form the selected base", y=0.96, fontsize=13)
    fig.text(0.5, 0.89, f'{h["route"]} · teacher-forced continuation', ha="center")
    fig.text(0.08, 0.10, "D = direct copy; T = learned translation. Letter order: KV / recurrent / convolution.", fontsize=9)
    fig.text(0.08, 0.05, "Means and 95% document bootstrap intervals recomputed from included observations. TDD was selected before LOCKED.",
             fontsize=9)
    save(fig, "e002_factorial_comparison")


def complete_e002(h):
    conditions = ["EMPTY_9B", "SOURCE_4B", "BASE_STATE", "JOINT_CORRECTED", "JOINT_SHUFFLED", "NATIVE_9B"]
    labels = ["Empty 9B", "Continued 4B", "Uncorrected TDD", "Corrected 9B", "Wrong donor", "Native 9B"]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    fig.subplots_adjust(left=0.23, right=0.95, bottom=0.22, top=0.77)
    bars(ax, labels, excess(h["E002"], conditions))
    fig.suptitle("E002 · complete handoff and control comparison", fontsize=14, y=0.96)
    fig.text(0.5, 0.88, f'{h["route"]} · 64 documents · teacher-forced', ha="center")
    fig.text(0.08, 0.10, "All six condition means are reproduced from included LOCKED observations.", fontsize=9)
    fig.text(0.08, 0.05, "Reported paired control intervals are in tables/e002_summary.csv; they are not intervals for these means.", fontsize=9)
    save(fig, "e002_complete_comparison")


def convergence(h, secondary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.23, top=0.78, wspace=0.30)
    e1 = secondary["e001_diagnostics"]["data"]["state_convergence"]
    x1 = sorted(map(int, e1))
    axes[0].plot(x1, [e1[str(k)]["mean_gdn_recurrent_normalized_frobenius"] for k in x1],
                 marker="o", label="Full translated state")
    e2 = secondary["e002_repair"]["data"]
    x2 = sorted(map(int, e2))
    for condition, label, marker in (("BASE_STATE", "Uncorrected TDD", "o"), ("JOINT_CORRECTED", "Corrected", "s")):
        axes[1].plot(x2, [e2[str(k)][condition]["recurrent_normalized_frobenius"] for k in x2],
                     marker=marker, label=label)
    for ax, title, ticks in zip(axes, ("E001 · PG19", "E002 · FineWeb-Edu"), (x1, x2)):
        ax.set_ylim(0, 0.7)
        ax.set_xscale("log", base=4)
        ax.set_title(title)
        ax.set_xticks(ticks, [str(value) for value in ticks])
        ax.set_xlabel("Tokens processed after handoff")
        ax.set_ylabel("Recurrent normalized Frobenius error")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("Secondary state tracking · error decreases but persists", fontsize=14, y=0.96)
    fig.text(0.5, 0.88, f'{h["route"]} · shared teacher-forced inputs', ha="center")
    fig.text(0.08, 0.10, "Means over recorded documents and state components. No checkpoint CIs are available.", fontsize=9)
    fig.text(0.08, 0.05, "E002 tracks state through 256 tokens; its primary continuation NLL still scores 64 targets.", fontsize=9)
    save(fig, "state_convergence")


def main():
    h = data("headline_results.json")
    require(h["schema_version"] == 1, "Unexpected derived schema")
    hero(h)
    factorial(h)
    complete_e002(h)
    convergence(h, data("secondary_results.json"))
    rows = csv_rows("derived/e001_per_document.csv")
    if "kv_only_nll" in rows[0]:
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.array([float(r["kv_only_nll"]) for r in rows])
        y = np.array([float(r["full_translated_nll"]) for r in rows])
        ax.scatter(x, y)
        bound = max(max(x), max(y)) * 1.05
        ax.plot([0, bound], [0, bound], linestyle="--")
        ax.set(xlim=(0, bound), ylim=(0, bound), xlabel="KV-only NLL", ylabel="Full translated NLL",
               title="E001 · paired document NLL · teacher-forced")
        fig.tight_layout()
        save(fig, "e001_paired_documents")
    else:
        print("UNAVAILABLE: paper Figure 3 requires E001 per-document NLLs; no scatter is fabricated.")


if __name__ == "__main__":
    main()
