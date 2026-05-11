"""
Plot DP-NAF audit results.

Inputs:
  runs/dp_naf_full/summary.json     (audit vs pretrained, 3 regimes)
  runs/dp_naf_full/trial_results.csv
  runs/dp_naf_loo/summary.json      (paired LOO audit, 2 regimes)
  runs/dp_naf_loo/trial_results.csv

Outputs:
  dp_naf_vs_pretrained.pdf  -- bar chart of measured KL vs eps^2/2
  dp_naf_loo.pdf            -- bar chart of paired-LOO KL vs eps^2/2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_vs_pretrained(summary_path: Path, csv_path: Path, out_path: Path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)["by_regime"]
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"regime": r["regime"],
                         "kl_exact": float(r["kl_exact"]),
                         "lb": float(r["lb"]),
                         "achieved_eps": float(r["achieved_eps"]) if r["achieved_eps"] != "inf" else float("inf")})

    regimes = list(summary.keys())
    kl_per_regime = {reg: [r["kl_exact"] for r in rows if r["regime"] == reg] for reg in regimes}
    lb_per_regime = {reg: [r["lb"] for r in rows if r["regime"] == reg] for reg in regimes}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(regimes))
    w = 0.35

    medians_kl = [np.median(kl_per_regime[r]) for r in regimes]
    maxes_kl = [np.max(kl_per_regime[r]) for r in regimes]
    medians_lb = [np.median(lb_per_regime[r]) for r in regimes]
    maxes_lb = [np.max(lb_per_regime[r]) for r in regimes]

    ax.bar(x - w/2, medians_kl, w, label="KL_exact (median)", color="#3b6cb8", alpha=0.85)
    ax.bar(x + w/2, medians_lb, w, label="Bernstein LB (median)", color="#f4a261", alpha=0.85)
    # Max as error-bar style markers
    ax.scatter(x - w/2, maxes_kl, marker="^", s=60, color="#1c3d77",
               label="KL_exact (max)", zorder=3)
    ax.scatter(x + w/2, maxes_lb, marker="^", s=60, color="#a76321",
               label="LB (max)", zorder=3)

    # Theoretical bound eps^2/2 as horizontal dashes per bar (only if finite)
    for i, reg in enumerate(regimes):
        bound = summary[reg]["theoretical_KL_bound_eps2_over_2"]
        try:
            b = float(bound)
        except (TypeError, ValueError):
            continue
        if b == float("inf") or b > 1e6:
            continue
        ax.hlines(b, i - 0.4, i + 0.4, color="red", lw=2, ls="--",
                  label=r"theory $\epsilon^2/2$" if i == 1 else None)
        ax.text(i, b, f"  $\\epsilon^2/2={b:.2f}$", color="red",
                fontsize=8, va="bottom", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("dp-", "DP, ").replace("eps", r"$\epsilon=$")
                        .replace("non-private", "Non-DP\n($\epsilon=\infty$)") for r in regimes])
    ax.set_ylabel(r"KL($P_{\mathrm{finetune}} \,\|\, P_{\mathrm{pretrain}}$)")
    ax.set_title("DP fine-tuning audit: KL to pretrained reference")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_loo(summary_path: Path, csv_path: Path, out_path: Path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)["by_eps"]
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"eps": float(r["target_eps"]),
                         "kl_exact": float(r["kl_exact"]),
                         "lb": float(r["lb"]),
                         "bound": float(r["bound_eps2_2"])})

    eps_values = sorted({r["eps"] for r in rows}, reverse=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(eps_values))
    w = 0.35

    kl_med = [np.median([r["kl_exact"] for r in rows if r["eps"] == e]) for e in eps_values]
    kl_max = [np.max([r["kl_exact"] for r in rows if r["eps"] == e]) for e in eps_values]
    bounds = [np.mean([r["bound"] for r in rows if r["eps"] == e]) for e in eps_values]

    ax.bar(x - w/2, kl_med, w, label=r"$\mathrm{KL}(M_{\mathrm{full}}\,\|\,M_{\mathrm{loo}})$ (median)",
           color="#3b6cb8", alpha=0.85)
    ax.bar(x + w/2, bounds, w, label=r"theory bound $\epsilon^2/2$",
           color="#e76f51", alpha=0.85)
    ax.scatter(x - w/2, kl_max, marker="^", s=60, color="#1c3d77",
               label=r"KL (max over prompts)", zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\epsilon = {e:g}$" for e in eps_values])
    ax.set_ylabel("KL nats (log scale)")
    ax.set_title("Paired LOO DP-NAF test:\nempirical KL vs $\\epsilon^2/2$ bound")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    # Annotate each bar with violation count
    for i, e in enumerate(eps_values):
        s = summary[f"eps={e}"]
        ax.text(i, max(kl_max[i], bounds[i]) * 1.4,
                f"{s['violations_kl_exceeds_bound']}/{s['n_prompts']} violations",
                ha="center", fontsize=8, color="darkred",
                fontweight="bold" if s["violations_kl_exceeds_bound"] > 0 else "normal")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vs_pretrained_dir", type=str, default="runs/dp_naf_full")
    p.add_argument("--loo_dir", type=str, default="runs/dp_naf_loo")
    p.add_argument("--out_dir", type=str, default="runs/dp_naf_full/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vp_dir = Path(args.vs_pretrained_dir)
    if (vp_dir / "summary.json").exists():
        plot_vs_pretrained(vp_dir / "summary.json", vp_dir / "trial_results.csv",
                           out_dir / "dp_naf_vs_pretrained.pdf")
    loo_dir = Path(args.loo_dir)
    if (loo_dir / "summary.json").exists():
        plot_loo(loo_dir / "summary.json", loo_dir / "trial_results.csv",
                 out_dir / "dp_naf_loo.pdf")


if __name__ == "__main__":
    main()
