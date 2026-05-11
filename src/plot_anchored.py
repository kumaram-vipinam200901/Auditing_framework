"""
Plot anchored-decoding tightness results.

Inputs:  runs/anchored_tightness/results.csv
Outputs: anchored_kl_box.pdf, anchored_budget_util.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "K": float(r["K"]),
                "kl_risky": float(r["kl_risky_vs_safe"]),
                "kl_anchored": float(r["kl_anchored_vs_safe"]),
                "audit_lb": float(r["audit_lb"]),
                "theta": float(r["theta"]),
            })
    return rows


def plot_kl_distributions(rows, out_path: Path):
    Ks = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.2))

    width = 0.25
    x = np.arange(len(Ks))

    # Three distributions per K: KL_risky (no protection), KL_anchored (exact), audit_LB
    risky_data = [[r["kl_risky"] for r in rows if r["K"] == K] for K in Ks]
    anch_data = [[r["kl_anchored"] for r in rows if r["K"] == K] for K in Ks]
    lb_data = [[r["audit_lb"] for r in rows if r["K"] == K] for K in Ks]

    bp1 = ax.boxplot(risky_data, positions=x - width, widths=width * 0.8,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="#cccccc", edgecolor="black"),
                     medianprops=dict(color="black"))
    bp2 = ax.boxplot(anch_data, positions=x, widths=width * 0.8,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="#3b6cb8", edgecolor="black"),
                     medianprops=dict(color="black"))
    bp3 = ax.boxplot(lb_data, positions=x + width, widths=width * 0.8,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="#f4a261", edgecolor="black"),
                     medianprops=dict(color="black"))

    # Budget line: y = K
    for i, K in enumerate(Ks):
        ax.hlines(K, x[i] - 1.5*width, x[i] + 1.5*width, color="red", lw=1.6, ls="--")
        ax.text(x[i] + 1.5*width + 0.03, K, f"$K={K:g}$", color="red",
                fontsize=8, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$K={K:g}$" for K in Ks])
    ax.set_ylabel("KL nats (vs. safe model)")
    ax.set_xlabel("Claimed NAF budget $K$")
    ax.set_title("Anchored decoding: per-prompt KL stays under budget across all $K$")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
              ["KL$_{\\mathrm{risky}}$ (no protection)",
               "KL$_{\\mathrm{anchored}}$ (exact)",
               "Audit LB ($n=1000$, $\\delta=0.05$)"],
              loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_budget_utilization(rows, out_path: Path):
    Ks = sorted({r["K"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 3.8))
    utils = [[r["kl_anchored"] / K * 100 for r in rows if r["K"] == K] for K in Ks]
    bp = ax.boxplot(utils, positions=range(len(Ks)), widths=0.5,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor="#3b6cb8", edgecolor="black", alpha=0.8),
                    medianprops=dict(color="black", linewidth=1.5))
    ax.axhline(100, color="red", ls="--", lw=1, label="$K$ (budget)")
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"$K={K:g}$" for K in Ks])
    ax.set_ylabel("KL$_{\\mathrm{anchored}}$ / $K$  (\\%)")
    ax.set_title("Budget utilization of anchored decoding")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="runs/anchored_tightness/results.csv")
    p.add_argument("--out_dir", default="runs/anchored_tightness/figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(Path(args.csv))
    plot_kl_distributions(rows, out_dir / "anchored_kl_box.pdf")
    plot_budget_utilization(rows, out_dir / "anchored_budget_util.pdf")


if __name__ == "__main__":
    main()
