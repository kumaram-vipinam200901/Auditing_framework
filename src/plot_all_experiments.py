"""
Generate all conference-quality figures from the comprehensive experiments.

Reads runs/full_experiments/all_experiments.json and per-run JSONL files.
Produces 9 publication-ready figures (PNG + PDF).

Usage:
    python -m src.plot_all_experiments --input runs/full_experiments
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paper-quality settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "evo": "#e6550d",
    "rand": "#6baed6",
    "gpt2": "#3182bd",
    "pythia": "#31a354",
    "opt": "#756bb1",
    "fill": "#fd8d3c",
    "violation": "#de2d26",
    "safe": "#31a354",
}


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_master(exp_dir: Path) -> List[Dict]:
    with open(exp_dir / "all_experiments.json") as f:
        return json.load(f)


def _save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    logger.info("Saved %s", name)


# ------------------------------------------------------------------
# Fig 1: Model pair comparison across families  (grouped bar)
# ------------------------------------------------------------------

def fig_model_pairs(summaries: List[Dict], out_dir: Path):
    """Grouped bar: evolutionary vs random for every model pair, coloured by family."""
    pairs = [s for s in summaries if s.get("experiment") == "model_pair"]
    if not pairs:
        return

    # group by tag → {method: summary}
    groups = {}
    order = []
    for s in pairs:
        label = s["tag"].replace("mp_evo_", "").replace("mp_rand_", "")
        if label not in groups:
            groups[label] = {}
            order.append(label)
        groups[label][s["method"]] = s

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.4), 5))
    x = np.arange(len(order))
    w = 0.35

    evo_vals, rand_vals, families = [], [], []
    for label in order:
        g = groups[label]
        evo_vals.append(g.get("evolutionary", {}).get("copyright_leakage_score", 0))
        rand_vals.append(g.get("random", {}).get("copyright_leakage_score", 0))
        fam = g.get("evolutionary", g.get("random", {})).get("family", "")
        families.append(fam)

    # Colour by family
    fam_colors = {"GPT-2": COLORS["gpt2"], "Pythia": COLORS["pythia"], "OPT": COLORS["opt"]}
    bar_colors = [fam_colors.get(f, "#999999") for f in families]

    ax.bar(x - w / 2, rand_vals, w, label="Random Baseline",
           color=[c + "88" for c in bar_colors], edgecolor="white")  # lighter
    bars_evo = ax.bar(x + w / 2, evo_vals, w, label="Evolutionary Search",
                      color=bar_colors, edgecolor="white")
    ax.set_xticks(x)
    nice = [o.replace("_vs_", " → ").replace("_sanity", "\n(sanity)") for o in order]
    ax.set_xticklabels(nice, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Copyright Leakage Score (max LB)")
    ax.set_title("NAF Audit: Model Pair Comparison (3 Families)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # family legend patches
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=c, label=f) for f, c in fam_colors.items()
               if any(fam == f for fam in families)]
    if patches:
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:2],
                  loc="upper left", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "fig1_model_pairs")


# ------------------------------------------------------------------
# Fig 2: Dataset comparison (BookMIA vs WikiText)
# ------------------------------------------------------------------

def fig_dataset_comparison(summaries: List[Dict], out_dir: Path):
    items = [s for s in summaries if s.get("experiment") == "dataset"]
    if len(items) < 2:
        return

    groups = {}
    for s in items:
        groups.setdefault(s["dataset"], {})[s["method"]] = s

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ds_names = list(groups.keys())
    x = np.arange(len(ds_names))
    w = 0.35

    evo = [groups[d].get("evolutionary", {}).get("copyright_leakage_score", 0) for d in ds_names]
    rand = [groups[d].get("random", {}).get("copyright_leakage_score", 0) for d in ds_names]

    ax.bar(x - w / 2, rand, w, label="Random", color=COLORS["rand"], edgecolor="white")
    ax.bar(x + w / 2, evo, w, label="Evolutionary", color=COLORS["evo"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in ds_names])
    ax.set_ylabel("Copyright Leakage Score")
    ax.set_title("Dataset Comparison: BookMIA vs WikiText-103")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "fig2_dataset_comparison")


# ------------------------------------------------------------------
# Fig 3: Sample count ablation
# ------------------------------------------------------------------

def fig_sample_ablation(summaries: List[Dict], out_dir: Path):
    items = [s for s in summaries if s.get("experiment") == "sample_ablation"]
    if not items:
        return

    items.sort(key=lambda s: s["num_samples"])
    ns = [s["num_samples"] for s in items]
    lbs = [s["copyright_leakage_score"] for s in items]
    kls = [s["worst_kl_mean"] for s in items]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, kls, "s-", color=COLORS["evo"], linewidth=2, markersize=8,
            label="KL Mean (best prompt)")
    ax.plot(ns, lbs, "o-", color=COLORS["gpt2"], linewidth=2, markersize=8,
            label="KL Lower Bound")
    ax.fill_between(ns, lbs, kls, alpha=0.15, color=COLORS["fill"],
                    label="Bernstein gap")
    ax.set_xlabel("Number of Samples ($n$)")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Bernstein Bound Tightens with More Samples")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "fig3_sample_ablation")


# ------------------------------------------------------------------
# Fig 4: Round count ablation
# ------------------------------------------------------------------

def fig_round_ablation(summaries: List[Dict], out_dir: Path):
    items = [s for s in summaries if s.get("experiment") == "round_ablation"]
    if not items:
        return

    items.sort(key=lambda s: s["num_rounds_cfg"])
    rs = [s["num_rounds_cfg"] for s in items]
    lbs = [s["copyright_leakage_score"] for s in items]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rs, lbs, "D-", color=COLORS["evo"], linewidth=2, markersize=9)
    ax.set_xlabel("Number of Evolutionary Rounds")
    ax.set_ylabel("Max KL Lower Bound")
    ax.set_title("Evolutionary Search Improves with More Rounds")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "fig4_round_ablation")


# ------------------------------------------------------------------
# Fig 5: Mutation operator ablation
# ------------------------------------------------------------------

def fig_mutation_ablation(summaries: List[Dict], out_dir: Path):
    items = [s for s in summaries if s.get("experiment") == "mutation_ablation"]
    if not items:
        return

    items.sort(key=lambda s: s["copyright_leakage_score"])
    names = [s["mutation_set"] for s in items]
    lbs = [s["copyright_leakage_score"] for s in items]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))
    ax.barh(names, lbs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Max KL Lower Bound")
    ax.set_title("Mutation Operator Ablation")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(lbs):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir, "fig5_mutation_ablation")


# ------------------------------------------------------------------
# Fig 6: K threshold sweep
# ------------------------------------------------------------------

def fig_k_threshold(summaries: List[Dict], out_dir: Path):
    items = [s for s in summaries if s.get("experiment") == "K_threshold"]
    if not items:
        return

    items.sort(key=lambda s: s["claimed_K"])
    ks = [s["claimed_K"] for s in items]
    viols = [s["total_violations"] for s in items]
    lbs = [s["copyright_leakage_score"] for s in items]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.bar(range(len(ks)), viols, color=COLORS["gpt2"], alpha=0.7, label="Violations")
    ax1.set_xticks(range(len(ks)))
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel("Claimed $K$ (NAF bound)")
    ax1.set_ylabel("Number of Violations", color=COLORS["gpt2"])
    ax1.tick_params(axis="y", labelcolor=COLORS["gpt2"])

    ax2 = ax1.twinx()
    ax2.plot(range(len(ks)), lbs, "D-", color=COLORS["evo"], linewidth=2,
             markersize=8, label="Max LB")
    ax2.set_ylabel("Max KL Lower Bound", color=COLORS["evo"])
    ax2.tick_params(axis="y", labelcolor=COLORS["evo"])

    fig.suptitle("Violations Drop as Claimed $K$ Increases", fontsize=13, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig6_k_threshold")


# ------------------------------------------------------------------
# Fig 7: Delta (confidence) sweep
# ------------------------------------------------------------------

def fig_delta_sweep(summaries: List[Dict], out_dir: Path):
    items = [s for s in summaries if s.get("experiment") == "delta_sweep"]
    if not items:
        return

    items.sort(key=lambda s: s["delta_cfg"])
    deltas = [s["delta_cfg"] for s in items]
    lbs = [s["copyright_leakage_score"] for s in items]
    viols = [s["total_violations"] for s in items]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(deltas, lbs, "o-", color=COLORS["evo"], linewidth=2, markersize=8,
             label="Max Lower Bound")
    ax1.set_xlabel(r"Confidence parameter $\delta$")
    ax1.set_ylabel("Max KL Lower Bound", color=COLORS["evo"])
    ax1.tick_params(axis="y", labelcolor=COLORS["evo"])

    ax2 = ax1.twinx()
    ax2.bar(deltas, viols, width=0.008, color=COLORS["gpt2"], alpha=0.5, label="Violations")
    ax2.set_ylabel("Violations", color=COLORS["gpt2"])
    ax2.tick_params(axis="y", labelcolor=COLORS["gpt2"])

    fig.suptitle(r"Lower $\delta$ = Higher Confidence, Looser Bound",
                 fontsize=13, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_delta_sweep")


# ------------------------------------------------------------------
# Fig 8: Large-scale distribution (histogram + box)
# ------------------------------------------------------------------

def fig_largescale_distribution(exp_dir: Path, out_dir: Path):
    # Try both naming conventions
    for evo_name, rand_name in [
        ("large_evo", "large_rand"),
        ("largescale_evo", "largescale_random"),
    ]:
        evo_path = exp_dir / evo_name / "results.jsonl"
        rand_path = exp_dir / rand_name / "results.jsonl"
        if evo_path.exists() and rand_path.exists():
            break
    else:
        return

    evo = load_jsonl(evo_path)
    rand = load_jsonl(rand_path)
    if not evo or not rand:
        return

    evo_lb = [r["lower_bound"] for r in evo]
    rand_lb = [r["lower_bound"] for r in rand]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histograms
    ax = axes[0]
    lo = min(min(evo_lb), min(rand_lb))
    hi = max(max(evo_lb), max(rand_lb))
    bins = np.linspace(lo, hi, 40)
    ax.hist(rand_lb, bins=bins, alpha=0.6, color=COLORS["rand"],
            label="Random", edgecolor="white")
    ax.hist(evo_lb, bins=bins, alpha=0.6, color=COLORS["evo"],
            label="Evolutionary", edgecolor="white")
    claimed_K = evo[0].get("claimed_K", 1.0)
    ax.axvline(claimed_K, color="red", linestyle="--", linewidth=2,
               label=f"$K={claimed_K}$")
    ax.set_xlabel("KL Lower Bound")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of KL Lower Bounds")
    ax.legend()

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(
        [rand_lb, evo_lb],
        tick_labels=["Random", "Evolutionary"],
        patch_artist=True, widths=0.5,
    )
    bp["boxes"][0].set_facecolor(COLORS["rand"])
    bp["boxes"][1].set_facecolor(COLORS["evo"])
    ax.axhline(claimed_K, color="red", linestyle="--", linewidth=1.5,
               label=f"$K={claimed_K}$")
    ax.set_ylabel("KL Lower Bound")
    ax.set_title("Random vs Evolutionary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Large-Scale Audit ({len(rand)} base prompts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig8_largescale")


# ------------------------------------------------------------------
# Fig 9: Summary heatmap — all experiments at a glance
# ------------------------------------------------------------------

def fig_summary_heatmap(summaries: List[Dict], out_dir: Path):
    """Single figure that gives a bird's-eye view of all experiments."""
    if len(summaries) < 4:
        return

    tags = [s["tag"][:35] for s in summaries]
    lbs = [s["copyright_leakage_score"] for s in summaries]
    viols = [s["total_violations"] for s in summaries]

    fig, ax = plt.subplots(figsize=(10, max(6, len(tags) * 0.28)))
    y = np.arange(len(tags))
    colors = [COLORS["violation"] if v > 0 else COLORS["safe"] for v in viols]
    ax.barh(y, lbs, color=colors, edgecolor="white", height=0.7, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=7)
    ax.set_xlabel("Copyright Leakage Score (max KL lower bound)")
    ax.set_title("All Experiments — Overview", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Annotate violation counts
    for i, (lb, v) in enumerate(zip(lbs, viols)):
        txt = f" {v} viol." if v > 0 else ""
        ax.text(max(lb, 0) + 0.02, i, f"{lb:.2f}{txt}", va="center", fontsize=7)

    fig.tight_layout()
    _save(fig, out_dir, "fig9_summary")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from experiment results")
    parser.add_argument("--input", type=str, default="runs/full_experiments",
                        help="Experiment directory with all_experiments.json")
    args = parser.parse_args()

    exp_dir = Path(args.input)
    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_master(exp_dir)
    logger.info("Loaded %d experiment summaries", len(summaries))

    fig_model_pairs(summaries, out_dir)         # Fig 1
    fig_dataset_comparison(summaries, out_dir)   # Fig 2
    fig_sample_ablation(summaries, out_dir)      # Fig 3
    fig_round_ablation(summaries, out_dir)       # Fig 4
    fig_mutation_ablation(summaries, out_dir)     # Fig 5
    fig_k_threshold(summaries, out_dir)          # Fig 6
    fig_delta_sweep(summaries, out_dir)          # Fig 7
    fig_largescale_distribution(exp_dir, out_dir) # Fig 8
    fig_summary_heatmap(summaries, out_dir)      # Fig 9

    logger.info("All %d figures saved to %s", 9, out_dir)


if __name__ == "__main__":
    main()
