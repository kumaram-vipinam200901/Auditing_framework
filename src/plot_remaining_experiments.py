"""
Generate publication-quality figures from the remaining-work experiments.

Reads runs/remaining_overnight/remaining_experiments.json and resource_log.json.
Produces 6 figures (PNG + PDF):
  Fig R1: Scaling — model pair comparison (grouped bar)
  Fig R2: Anchored decoding — KL vs budget K (line + shaded)
  Fig R3: Sequence-level KL vs token horizon T (line)
  Fig R4: Curated vs BookMIA prompts (grouped bar)
  Fig R5: Resource usage timeline (GPU + RAM over experiment phases)
  Fig R6: Summary heatmap — all remaining experiments at a glance

Usage:
    python -m src.plot_remaining_experiments --input runs/remaining_overnight
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
    "anchored": "#756bb1",
    "seqkl": "#e7298a",
    "curated": "#d95f02",
    "bookmia": "#7570b3",
    "fill": "#fd8d3c",
    "violation": "#de2d26",
    "safe": "#31a354",
    "gpu": "#e6550d",
    "ram": "#3182bd",
}


def _save(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    logger.info("Saved %s", name)


def load_master(exp_dir: Path) -> List[Dict]:
    with open(exp_dir / "remaining_experiments.json") as f:
        return json.load(f)


def load_resource_log(exp_dir: Path) -> List[Dict]:
    p = exp_dir / "resource_log.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Fig R1: Scaling — model pair comparison
# ------------------------------------------------------------------

def fig_scaling(summaries: List[Dict], out_dir: Path):
    """Grouped bar: evolutionary vs random for each scaled model pair."""
    items = [s for s in summaries if s.get("experiment") == "scaling"]
    if not items:
        logger.warning("No scaling experiments found, skipping fig_scaling")
        return

    groups = {}
    order = []
    for s in items:
        label = s["tag"].replace("scale_evo_", "").replace("scale_rand_", "")
        if label not in groups:
            groups[label] = {}
            order.append(label)
        groups[label][s["method"]] = s

    fig, ax = plt.subplots(figsize=(max(9, len(order) * 2.4), 5.5))
    x = np.arange(len(order))
    w = 0.35

    evo_lb, rand_lb, evo_kl, rand_kl, families = [], [], [], [], []
    for label in order:
        g = groups[label]
        evo_lb.append(g.get("evolutionary", {}).get("copyright_leakage_score", 0))
        rand_lb.append(g.get("random", {}).get("copyright_leakage_score", 0))
        evo_kl.append(g.get("evolutionary", {}).get("worst_kl_mean", 0))
        rand_kl.append(g.get("random", {}).get("worst_kl_mean", 0))
        fam = g.get("evolutionary", g.get("random", {})).get("family", "")
        families.append(fam)

    # Distinct evo/rand colors; hatching encodes family
    fam_hatches = {"GPT-2": "", "Pythia": "//"}

    for i, label in enumerate(order):
        h = fam_hatches.get(families[i], "")
        ax.bar(x[i] - w / 2, rand_lb[i], w,
               color=COLORS["rand"], edgecolor="#333", linewidth=0.8, hatch=h)
        ax.bar(x[i] + w / 2, evo_lb[i], w,
               color=COLORS["evo"], edgecolor="#333", linewidth=0.8, hatch=h)

    # KL mean error caps
    for i in range(len(order)):
        ax.plot([x[i] - w / 2], [rand_kl[i]], marker="_", color="black",
                markersize=12, markeredgewidth=2, zorder=5)
        ax.plot([x[i] + w / 2], [evo_kl[i]], marker="_", color="black",
                markersize=12, markeredgewidth=2, zorder=5)
        # Bernstein gap line
        ax.plot([x[i] - w / 2, x[i] - w / 2], [rand_lb[i], rand_kl[i]],
                color="black", linewidth=1, alpha=0.5)
        ax.plot([x[i] + w / 2, x[i] + w / 2], [evo_lb[i], evo_kl[i]],
                color="black", linewidth=1, alpha=0.5)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="$K=1.0$")

    ax.set_xticks(x)
    nice = [o.replace("_vs_", " \u2192 ").replace("_", "-") for o in order]
    ax.set_xticklabels(nice, fontsize=10)
    ax.set_ylabel("KL Divergence")
    ax.set_title("Scaling to Larger Models: NAF Audit Results", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate violations above evo bars
    for i, label in enumerate(order):
        g = groups[label]
        ev = g.get("evolutionary", {}).get("total_violations", 0)
        ax.text(x[i] + w / 2, evo_lb[i] + 0.3, f"{ev} viol.",
                ha="center", fontsize=8, color=COLORS["violation"], fontweight="bold")

    # Build clean legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLORS["evo"], edgecolor="#333", label="Evolutionary"),
        Patch(facecolor=COLORS["rand"], edgecolor="#333", label="Random"),
        Patch(facecolor="#cccccc", edgecolor="#333", hatch="//", label="Pythia family"),
        Patch(facecolor="#cccccc", edgecolor="#333", label="GPT-2 family"),
        plt.Line2D([0], [0], color="black", marker="_", linestyle="None",
                   markersize=10, markeredgewidth=2, label="KL mean"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="$K=1.0$"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              framealpha=0.9, ncol=2)

    fig.tight_layout()
    _save(fig, out_dir, "figR1_scaling")


# ------------------------------------------------------------------
# Fig R2: Anchored decoding — KL vs budget K
# ------------------------------------------------------------------

def fig_anchored_decoding(summaries: List[Dict], out_dir: Path):
    """Line plot: measured KL LB vs claimed K budget. Should stay below y=x."""
    items = [s for s in summaries if s.get("experiment") == "anchored_decoding"]
    if not items:
        logger.warning("No anchored decoding experiments found")
        return

    # Group by K
    k_groups = {}
    for s in items:
        k = s.get("K_anchor", s.get("claimed_K"))
        if k is None:
            continue
        k_groups.setdefault(k, {})[s["method"]] = s

    ks = sorted(k_groups.keys())

    fig, ax = plt.subplots(figsize=(7, 5))

    # y=x line (budget boundary)
    k_range = np.linspace(0, max(ks) * 1.15, 100)
    ax.plot(k_range, k_range, "k--", linewidth=1.5, alpha=0.5, label="$y = K$ (budget)")
    ax.fill_between(k_range, k_range, max(ks) * 1.5, alpha=0.06, color="red")
    ax.fill_between(k_range, 0, k_range, alpha=0.06, color="green")

    evo_lb = [k_groups[k].get("evolutionary", {}).get("copyright_leakage_score", 0) for k in ks]
    rand_lb = [k_groups[k].get("random", {}).get("copyright_leakage_score", 0) for k in ks]
    evo_kl = [k_groups[k].get("evolutionary", {}).get("worst_kl_mean", 0) for k in ks]

    ax.plot(ks, evo_lb, "o-", color=COLORS["evo"], linewidth=2, markersize=8,
            label="Evo search (max LB)")
    ax.plot(ks, rand_lb, "s-", color=COLORS["rand"], linewidth=2, markersize=8,
            label="Random (max LB)")
    ax.plot(ks, evo_kl, "^--", color=COLORS["evo"], linewidth=1, markersize=6,
            alpha=0.6, label="Evo (KL mean)")

    ax.set_xlabel("Anchored Decoding Budget $K$")
    ax.set_ylabel("Measured KL Lower Bound")
    ax.set_title("Anchored Decoding Enforces NAF Guarantee")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax.text(max(ks) * 0.6, max(ks) * 0.2, "NAF-compliant\nregion",
            fontsize=10, color="green", alpha=0.7, ha="center")
    ax.text(max(ks) * 0.3, max(ks) * 0.9, "Violation\nregion",
            fontsize=10, color="red", alpha=0.7, ha="center")

    fig.tight_layout()
    _save(fig, out_dir, "figR2_anchored_decoding")


# ------------------------------------------------------------------
# Fig R3: Sequence-level KL vs token horizon T
# ------------------------------------------------------------------

def fig_sequence_kl(summaries: List[Dict], out_dir: Path):
    """Line plot: KL grows with sequence length T."""
    items = [s for s in summaries if s.get("experiment") == "sequence_kl"]
    if not items:
        logger.warning("No sequence KL experiments found")
        return

    # Sort by T (or by tag for token baseline)
    def sort_key(s):
        t = s.get("seq_len")
        if t is not None:
            return t
        if "token" in s["tag"]:
            return 1
        return 0

    items.sort(key=sort_key)

    ts, lbs, kls, viols = [], [], [], []
    for s in items:
        t = s.get("seq_len", 1)
        ts.append(t)
        lbs.append(s.get("max_lower_bound", s.get("copyright_leakage_score", 0)))
        kls.append(s.get("max_kl_mean", s.get("worst_kl_mean", 0)))
        viols.append(s.get("total_violations", 0))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ts, kls, "s-", color=COLORS["seqkl"], linewidth=2.5, markersize=9,
             label="KL Mean (best prompt)")
    ax1.plot(ts, lbs, "o-", color=COLORS["gpt2"], linewidth=2.5, markersize=9,
             label="KL Lower Bound")
    ax1.fill_between(ts, lbs, kls, alpha=0.15, color=COLORS["fill"],
                     label="Bernstein gap")

    ax1.set_xlabel("Sequence Length $T$ (tokens)")
    ax1.set_ylabel("KL Divergence")
    ax1.set_title("Sequence-Level KL Grows with Horizon $T$")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Secondary axis for violations
    ax2 = ax1.twinx()
    ax2.bar(ts, viols, width=[max(1, t * 0.15) for t in ts],
            alpha=0.25, color=COLORS["violation"], label="Violations")
    ax2.set_ylabel("Violations", color=COLORS["violation"])
    ax2.tick_params(axis="y", labelcolor=COLORS["violation"])

    # Annotate each point
    for i, (t, lb) in enumerate(zip(ts, lbs)):
        ax1.annotate(f"T={t}", (t, lb), textcoords="offset points",
                     xytext=(0, 12), fontsize=8, ha="center")

    fig.tight_layout()
    _save(fig, out_dir, "figR3_sequence_kl")


# ------------------------------------------------------------------
# Fig R4: Curated vs BookMIA prompts
# ------------------------------------------------------------------

def fig_curated_prompts(summaries: List[Dict], out_dir: Path):
    """Grouped bar comparing curated memorised prompts vs BookMIA."""
    items = [s for s in summaries
             if s.get("experiment") in ("curated_prompts", "curated_vs_bookmia")]
    if not items:
        logger.warning("No curated prompt experiments found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: curated evo vs rand
    curated_items = [s for s in items if s.get("experiment") == "curated_prompts"]
    if curated_items:
        ax = axes[0]
        methods = {s["method"]: s for s in curated_items}
        names = ["Evolutionary", "Random"]
        vals = [
            methods.get("evolutionary", {}).get("copyright_leakage_score", 0),
            methods.get("random", {}).get("copyright_leakage_score", 0),
        ]
        viol = [
            methods.get("evolutionary", {}).get("total_violations", 0),
            methods.get("random", {}).get("total_violations", 0),
        ]
        colors = [COLORS["evo"], COLORS["rand"]]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.5)
        for i, (v, vl) in enumerate(zip(vals, viol)):
            ax.text(i, v + 0.05, f"LB={v:.2f}\n{vl} viol.",
                    ha="center", fontsize=9)
        ax.set_ylabel("Copyright Leakage Score")
        ax.set_title("Curated Memorised Prompts")
        ax.grid(axis="y", alpha=0.3)

    # Right: curated vs bookmia head-to-head
    cmp_items = [s for s in items if s.get("experiment") == "curated_vs_bookmia"]
    if cmp_items:
        ax = axes[1]
        sources = {s["prompt_source"]: s for s in cmp_items}
        names = ["Curated\n(Carlini et al.)", "BookMIA"]
        vals = [
            sources.get("carlini", sources.get("curated", {})).get("copyright_leakage_score", 0),
            sources.get("bookmia", {}).get("copyright_leakage_score", 0),
        ]
        viol = [
            sources.get("carlini", sources.get("curated", {})).get("total_violations", 0),
            sources.get("bookmia", {}).get("total_violations", 0),
        ]
        colors = [COLORS["curated"], COLORS["bookmia"]]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.5)
        for i, (v, vl) in enumerate(zip(vals, viol)):
            ax.text(i, v + 0.03, f"LB={v:.2f}\n{vl} viol.",
                    ha="center", fontsize=9)
        ax.set_ylabel("Copyright Leakage Score")
        ax.set_title("Curated vs BookMIA (Random Baseline)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Curated Memorised Passages Stress-Test the Audit",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "figR4_curated_prompts")


# ------------------------------------------------------------------
# Fig R5: Resource usage timeline
# ------------------------------------------------------------------

def fig_resource_timeline(resource_log: List[Dict], out_dir: Path):
    """Timeline of GPU and RAM usage across experiment phases."""
    if not resource_log:
        logger.warning("No resource log found")
        return

    events = [e for e in resource_log if "wall_time_epoch" in e]
    if len(events) < 2:
        return

    t0 = events[0]["wall_time_epoch"]
    times_min = [(e["wall_time_epoch"] - t0) / 60 for e in events]
    gpu_mb = [e.get("gpu_mem_allocated_mb", 0) for e in events]
    ram_mb = [e.get("cpu_rss_mb", 0) for e in events]
    labels = [e.get("event", "") for e in events]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(times_min, gpu_mb, "-o", color=COLORS["gpu"], linewidth=2,
             markersize=5, label="GPU VRAM (MB)")
    ax1.set_xlabel("Wall Time (minutes)")
    ax1.set_ylabel("GPU VRAM (MB)", color=COLORS["gpu"])
    ax1.tick_params(axis="y", labelcolor=COLORS["gpu"])

    ax2 = ax1.twinx()
    ax2.plot(times_min, ram_mb, "-s", color=COLORS["ram"], linewidth=2,
             markersize=5, label="CPU RAM (MB)")
    ax2.set_ylabel("CPU RAM (MB)", color=COLORS["ram"])
    ax2.tick_params(axis="y", labelcolor=COLORS["ram"])

    # Shade experiment phases
    phase_colors = {
        "scaling": "#3182bd22",
        "anchored": "#756bb122",
        "sequence": "#e7298a22",
        "curated": "#d95f0222",
    }
    phase_labels_drawn = set()
    for i, lbl in enumerate(labels):
        if lbl.startswith("start_"):
            phase = lbl.replace("start_", "")
            end_lbl = f"end_{phase}"
            for j in range(i + 1, len(labels)):
                if labels[j] == end_lbl:
                    color = phase_colors.get(phase, "#cccccc22")
                    draw_label = phase not in phase_labels_drawn
                    ax1.axvspan(times_min[i], times_min[j], alpha=0.15,
                                color=color.replace("22", ""),
                                label=phase.title() if draw_label else None)
                    phase_labels_drawn.add(phase)
                    break

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_title("Resource Usage Across Experiment Phases")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "figR5_resource_timeline")


# ------------------------------------------------------------------
# Fig R6: Summary heatmap — all remaining experiments
# ------------------------------------------------------------------

def fig_summary_heatmap(summaries: List[Dict], out_dir: Path):
    """Horizontal bar chart overview of all remaining experiments."""
    if len(summaries) < 2:
        return

    tags = [s["tag"][:40] for s in summaries]
    lbs = [s.get("copyright_leakage_score", s.get("max_lower_bound", 0)) for s in summaries]
    viols = [s.get("total_violations", 0) for s in summaries]

    # Color by experiment type
    exp_colors = {
        "scaling": COLORS["gpt2"],
        "anchored_decoding": COLORS["anchored"],
        "sequence_kl": COLORS["seqkl"],
        "curated_prompts": COLORS["curated"],
        "curated_vs_bookmia": COLORS["bookmia"],
    }
    colors = [exp_colors.get(s.get("experiment", ""), "#999999") for s in summaries]
    edge = [COLORS["violation"] if v > 0 else "#ffffff" for v in viols]

    fig, ax = plt.subplots(figsize=(10, max(6, len(tags) * 0.32)))
    y = np.arange(len(tags))
    ax.barh(y, lbs, color=colors, edgecolor=edge, linewidth=1.5,
            height=0.7, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=7.5)
    ax.set_xlabel("Copyright Leakage Score (max KL lower bound)")
    ax.set_title("Remaining Experiments — Overview", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for i, (lb, v) in enumerate(zip(lbs, viols)):
        txt = f" {v}v" if v > 0 else ""
        ax.text(max(lb, 0) + 0.08, i, f"{lb:.2f}{txt}", va="center", fontsize=7)

    # Legend for experiment types
    from matplotlib.patches import Patch
    patches = []
    seen = set()
    for s in summaries:
        exp = s.get("experiment", "")
        if exp not in seen and exp in exp_colors:
            patches.append(Patch(facecolor=exp_colors[exp], label=exp.replace("_", " ").title()))
            seen.add(exp)
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, "figR6_summary")


# ------------------------------------------------------------------
# Fig R7: Per-prompt KL distributions (violin + strip)
# ------------------------------------------------------------------

def fig_kl_distributions(summaries: List[Dict], exp_dir: Path, out_dir: Path):
    """Violin plots showing full per-prompt KL distributions for scaling pairs."""
    items = [s for s in summaries if s.get("experiment") == "scaling"
             and s.get("method") == "evolutionary"]
    if not items:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    all_data = []
    labels = []

    for s in items:
        tag = s["tag"]
        jsonl_path = exp_dir / tag / "results.jsonl"
        if not jsonl_path.exists():
            continue
        recs = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        if not recs:
            continue
        lbs = [r["lower_bound"] for r in recs]
        all_data.append(lbs)
        nice = tag.replace("scale_evo_", "").replace("_vs_", " \u2192 ").replace("_", "-")
        labels.append(nice)

    if not all_data:
        return

    parts = ax.violinplot(all_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS["evo"])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color(COLORS["gpt2"])

    # Overlay strip (jittered dots)
    for i, data in enumerate(all_data):
        jitter = np.random.default_rng(42).normal(0, 0.04, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   alpha=0.3, s=8, color="#333")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="$K=1.0$")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("KL Lower Bound (per prompt)")
    ax.set_title("Per-Prompt KL Distributions Across Model Pairs", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "figR7_kl_distributions")


# ------------------------------------------------------------------
# Fig R8: Anchored decoding — budget utilization
# ------------------------------------------------------------------

def fig_anchored_utilization(summaries: List[Dict], out_dir: Path):
    """Bar chart: what fraction of K budget is actually used by anchored decoding."""
    items = [s for s in summaries if s.get("experiment") == "anchored_decoding"
             and s.get("method") == "evolutionary"]
    if not items:
        return

    items.sort(key=lambda s: s.get("K_anchor", 0))
    ks = [s["K_anchor"] for s in items]
    lbs = [s["copyright_leakage_score"] for s in items]
    utilization = [lb / k if k > 0 else 0 for lb, k in zip(lbs, ks)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(ks)), utilization, color=COLORS["anchored"],
                  edgecolor="white", width=0.6)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="100% = violation boundary")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"$K={k}$" for k in ks])
    ax.set_ylabel("Budget Utilization (max LB / $K$)")
    ax.set_title("Anchored Decoding: KL Budget Utilization", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for i, (u, lb) in enumerate(zip(utilization, lbs)):
        ax.text(i, u + 0.02, f"{u:.0%}\n(LB={lb:.2f})",
                ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, out_dir, "figR8_anchored_utilization")


# ------------------------------------------------------------------
# Fig R9: Model size gap vs KL divergence
# ------------------------------------------------------------------

MODEL_PARAMS = {
    "gpt2": 124, "gpt2-medium": 355, "gpt2-xl": 1558,
    "EleutherAI/pythia-70m": 70, "EleutherAI/pythia-160m": 160,
    "EleutherAI/pythia-1b": 1000,
}

def fig_model_size_vs_kl(summaries: List[Dict], out_dir: Path):
    """Scatter: parameter-count ratio (risky/safe) vs KL divergence."""
    items = [s for s in summaries if s.get("experiment") == "scaling"
             and s.get("method") == "evolutionary"]
    if not items:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for s in items:
        risky_name = s.get("risky", "")
        safe_name = s.get("safe", "")
        rp = MODEL_PARAMS.get(risky_name, 0)
        sp = MODEL_PARAMS.get(safe_name, 1)
        ratio = rp / sp if sp > 0 else 0
        lb = s["copyright_leakage_score"]
        fam = s.get("family", "")
        c = COLORS["gpt2"] if fam == "GPT-2" else COLORS["pythia"]
        ax.scatter(ratio, lb, s=120, color=c, edgecolor="#333", zorder=5)
        nice = s["tag"].replace("scale_evo_", "").replace("_vs_", "\u2192").replace("_", "-")
        ax.annotate(nice, (ratio, lb), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    ax.set_xlabel("Parameter Ratio (risky / safe)", fontsize=11)
    ax.set_ylabel("Max KL Lower Bound", fontsize=11)
    ax.set_title("Larger Model Gap \u2192 Higher KL Divergence", fontweight="bold")
    ax.grid(alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLORS["gpt2"], edgecolor="#333", label="GPT-2"),
        Patch(facecolor=COLORS["pythia"], edgecolor="#333", label="Pythia"),
    ], fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir, "figR9_model_size_vs_kl")


# ------------------------------------------------------------------
# Fig R10: Evolutionary convergence across rounds
# ------------------------------------------------------------------

def fig_evolutionary_convergence(summaries: List[Dict], exp_dir: Path, out_dir: Path):
    """Line plot: best KL lower bound per round for key experiments."""
    targets = [
        ("scale_evo_gpt2xl_vs_gpt2", "GPT2-XL \u2192 GPT2"),
        ("scale_evo_pythia1b_vs_70m", "Pythia-1B \u2192 70M"),
        ("anch_evo_anchored_K1.0", "Anchored K=1.0"),
        ("curated_evo", "Curated prompts"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_colors = [COLORS["gpt2"], COLORS["pythia"], COLORS["anchored"], COLORS["curated"]]
    plotted = False

    for (tag, label), c in zip(targets, plot_colors):
        jsonl_path = exp_dir / tag / "results.jsonl"
        if not jsonl_path.exists():
            continue
        recs = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        if not recs:
            continue

        # Track best LB per round
        rounds = sorted(set(r.get("round", 0) for r in recs))
        best_per_round = []
        running_best = -float("inf")
        for rd in rounds:
            rd_recs = [r for r in recs if r.get("round", 0) == rd]
            rd_best = max(r["lower_bound"] for r in rd_recs)
            running_best = max(running_best, rd_best)
            best_per_round.append(running_best)

        ax.plot(rounds, best_per_round, "o-", color=c, linewidth=2,
                markersize=7, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Evolutionary Round")
    ax.set_ylabel("Best KL Lower Bound (cumulative)")
    ax.set_title("Evolutionary Search Convergence", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "figR10_convergence")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from remaining-work experiment results")
    parser.add_argument("--input", type=str, default="runs/remaining_overnight",
                        help="Directory with remaining_experiments.json")
    args = parser.parse_args()

    exp_dir = Path(args.input)
    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_master(exp_dir)
    resource_log = load_resource_log(exp_dir)
    logger.info("Loaded %d experiment summaries, %d resource events",
                len(summaries), len(resource_log))

    fig_scaling(summaries, out_dir)                        # R1
    fig_anchored_decoding(summaries, out_dir)               # R2
    fig_sequence_kl(summaries, out_dir)                     # R3
    fig_curated_prompts(summaries, out_dir)                  # R4
    fig_resource_timeline(resource_log, out_dir)             # R5
    fig_summary_heatmap(summaries, out_dir)                  # R6
    fig_kl_distributions(summaries, exp_dir, out_dir)        # R7
    fig_anchored_utilization(summaries, out_dir)             # R8
    fig_model_size_vs_kl(summaries, out_dir)                 # R9
    fig_evolutionary_convergence(summaries, exp_dir, out_dir) # R10

    logger.info("All 10 figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
