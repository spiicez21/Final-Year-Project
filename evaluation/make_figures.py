"""
Week 11 (Docs/TODO.md): PDM v2 curve and alpha-sweep KBD heatmap figures.
Recomputes PDM v2 scores offline from the responses already saved in
evaluation/results/alpha_sweep_results.json (run_alpha_sweep.py printed the
aggregated numbers to console but only saved raw responses to JSON) — no
GPU/model needed, just re-scoring text already generated.

Usage:
    python evaluation/make_figures.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbd_scorer import load_knowledge_base, compute_kbd
from pdm_v2 import build_archetype_lexicons, build_reference_features, single_turn_drift_v2

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "alpha_sweep_results.json"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PAIRS = [
    ("police officer+pharmacist", "police officer", "pharmacist"),
    ("social worker+executive", "social worker", "executive"),
    ("bartender+professor", "bartender", "professor"),
]

AFFIRM_CUES = [r"^\s*yes\b", r"^\s*yeah\b", r"^\s*yep\b", r"\b(that|it)'?s (right|correct|true)\b", r"\bconfirmed\b"]
DENY_CUES = [r"^\s*no\b", r"\bi don'?t (know|have)\b", r"\bnot sure\b", r"\bi'?m not\b",
             r"\bcan'?t confirm\b", r"\bno idea\b", r"\bdon'?t have access\b", r"\bhaven'?t heard\b"]


def classify_probe_response(response: str) -> str:
    text = response.lower().strip()
    if any(re.search(cue, text) for cue in DENY_CUES):
        return "denied"
    if any(re.search(cue, text) for cue in AFFIRM_CUES):
        return "affirmed"
    return "ambiguous"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    items = load_knowledge_base()
    all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    dataset_entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    lexicons = build_archetype_lexicons(dataset_entries)

    kbd_matrix = np.zeros((len(PAIRS), len(ALPHAS)))
    pdm_curves = {}  # pair_key -> {archetype: [mean_pdm_per_alpha]}

    for pair_idx, (pair_key, arch_a, arch_b) in enumerate(PAIRS):
        ref_a = build_reference_features(arch_a, dataset_entries, lexicons)
        ref_b = build_reference_features(arch_b, dataset_entries, lexicons)
        pdm_curves[pair_key] = {arch_a: [], arch_b: []}

        for alpha_idx, alpha in enumerate(ALPHAS):
            rows = [r for r in all_results if r["pair"] == pair_key and r["alpha"] == alpha]
            forbidden = [r for r in rows if r["type"] == "forbidden"]

            violations = 0
            for r in forbidden:
                score = compute_kbd(r["response"], r["target_archetype"], items)
                classification = classify_probe_response(r["response"])
                is_violation = (score["kbd"] is not None and score["kbd"] > 0) or classification == "affirmed"
                if is_violation:
                    violations += 1
            kbd_matrix[pair_idx, alpha_idx] = violations / len(forbidden) if forbidden else 0

            pdm_a_scores = [single_turn_drift_v2(r["response"], arch_a, ref_a, lexicons) for r in rows]
            pdm_b_scores = [single_turn_drift_v2(r["response"], arch_b, ref_b, lexicons) for r in rows]
            pdm_curves[pair_key][arch_a].append(np.mean(pdm_a_scores))
            pdm_curves[pair_key][arch_b].append(np.mean(pdm_b_scores))

    # --- Figure 1: PDM v2 curves, one subplot per pair ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (pair_key, arch_a, arch_b) in zip(axes, PAIRS):
        ax.plot(ALPHAS, pdm_curves[pair_key][arch_a], marker="o", label=f"PDM vs {arch_a}")
        ax.plot(ALPHAS, pdm_curves[pair_key][arch_b], marker="s", label=f"PDM vs {arch_b}")
        ax.set_title(pair_key, fontsize=10)
        ax.set_xlabel("alpha (weight on archetype A)")
        ax.set_ylabel("PDM v2 drift (lower = closer)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("PDM v2 drift vs. both parent archetypes across the alpha-sweep")
    fig.tight_layout()
    out1 = FIGURES_DIR / "pdm_v2_curves.png"
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    # --- Figure 2: KBD violation-rate heatmap ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    im = ax2.imshow(kbd_matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(ALPHAS)))
    ax2.set_xticklabels([f"{a:.2f}" for a in ALPHAS])
    ax2.set_yticks(range(len(PAIRS)))
    ax2.set_yticklabels([p[0] for p in PAIRS], fontsize=8)
    ax2.set_xlabel("alpha (weight on archetype A)")
    ax2.set_title("KBD violation rate across the alpha-sweep")
    for i in range(len(PAIRS)):
        for j in range(len(ALPHAS)):
            ax2.text(j, i, f"{kbd_matrix[i, j]:.0%}", ha="center", va="center",
                      color="white" if kbd_matrix[i, j] > 0.5 else "black", fontsize=9)
    fig2.colorbar(im, ax=ax2, label="violation rate")
    fig2.tight_layout()
    out2 = FIGURES_DIR / "alpha_sweep_kbd_heatmap.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
