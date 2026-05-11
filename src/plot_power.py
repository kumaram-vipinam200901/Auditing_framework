"""
Plot results of the power/calibration experiment.

Produces:
  power_curves.pdf    rejection rate vs K_factor, one line per sample size n
  calibration_scatter.pdf   per-trial LB vs true KL
  bernstein_gap.pdf   mean (KL_exact - LB) vs n, log-log
  exact_kl_dist.pdf   histogram of exact per-prompt KL

Usage:
  python -m src.plot_power --power_dir runs/power_full --null_dir runs/power_null --out_dir runs/power_full/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_trials(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "prompt_idx": int(r["prompt_idx"]),
                "kl_exact": float(r["kl_exact"]),
                "n": int(r["n"]),
                "z_bar": float(r["z_bar"]),
                "var": float(r["var"]),
                "lb": float(r["lb"]),
                "correction": float(r["correction"]),
            })
    return rows


def load_summary(power_dir: Path):
    with open(power_dir / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_power_curves(summary, out_path: Path, title: str):
    by_n = summary["by_n"]
    k_factors = [float(kf) for kf in summary["config"]["k_factors"]]
    n_values = sorted(int(n) for n in by_n.keys())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(n_values)))

    for color, n in zip(cmap, n_values):
        rates = [by_n[str(n)][f"{kf:.2f}"]["reject_rate"] for kf in k_factors]
        ax.plot(k_factors, rates, marker="o", lw=2, color=color, label=f"n = {n}")

    ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.7)
    ax.text(1.02, 0.05, "claimed K = true KL",
            color="grey", fontsize=9, transform=ax.get_yaxis_transform())

    # Type-I shaded region
    ax.axvspan(1.0, max(k_factors), alpha=0.08, color="red")
    ax.text(1.5, 0.95, "Type-I region\n(claimed K > true KL)",
            color="red", fontsize=9, ha="center", va="top", alpha=0.7)
    ax.text(0.5, 0.95, "Power region\n(claimed K < true KL)",
            color="green", fontsize=9, ha="center", va="top", alpha=0.7)
    ax.axhline(0.05, color="red", ls=":", lw=1, alpha=0.5)
    ax.text(max(k_factors), 0.07, r"$\delta=0.05$", color="red", fontsize=8, ha="right")

    ax.set_xlabel(r"Claimed $K$ as fraction of true $\mathrm{KL}$ (i.e. $K / \mathrm{KL}_{\mathrm{true}}$)")
    ax.set_ylabel("Rejection rate (Pr[LB > K])")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="MC samples", loc="center right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_calibration_scatter(rows, out_path: Path):
    """Per-trial scatter of LB vs true KL, validating LB <= KL_true with high prob."""
    fig, ax = plt.subplots(figsize=(6, 5))
    n_values = sorted({r["n"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(n_values)))

    for color, n in zip(cmap, n_values):
        sub = [r for r in rows if r["n"] == n]
        if not sub:
            continue
        kl_x = [r["kl_exact"] for r in sub]
        lb_y = [r["lb"] for r in sub]
        ax.scatter(kl_x, lb_y, s=4, color=color, alpha=0.4, label=f"n = {n}")

    # Diagonal: LB = KL_true (target if Bernstein were tight)
    lo = min(min(r["lb"] for r in rows), 0)
    hi = max(r["kl_exact"] for r in rows) * 1.05
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--",
            label=r"$LB = KL_{\mathrm{true}}$")

    ax.set_xlabel(r"True KL (exact, summed over vocabulary)")
    ax.set_ylabel(r"Bernstein lower bound $\widehat{\mathrm{LB}}_n(\delta=0.05)$")
    ax.set_title("Calibration: LB never exceeds true KL")
    ax.legend(loc="lower right", markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_bernstein_gap(rows, out_path: Path):
    """Mean Bernstein correction t_n(delta) vs n, and theoretical 1/sqrt(n) reference."""
    n_values = sorted({r["n"] for r in rows})
    mean_corr = []
    p25_corr = []
    p75_corr = []
    for n in n_values:
        cs = np.array([r["correction"] for r in rows if r["n"] == n])
        mean_corr.append(cs.mean())
        p25_corr.append(np.percentile(cs, 25))
        p75_corr.append(np.percentile(cs, 75))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(n_values, mean_corr, "o-", lw=2, color="steelblue", label=r"empirical $t_n(\delta)$")
    ax.fill_between(n_values, p25_corr, p75_corr, alpha=0.2, color="steelblue", label="IQR (25-75%)")

    # Theoretical 1/sqrt(n) trend through first point
    c0 = mean_corr[0] * math.sqrt(n_values[0])
    theo = [c0 / math.sqrt(n) for n in n_values]
    ax.plot(n_values, theo, "k--", lw=1, alpha=0.7, label=r"$\propto 1/\sqrt{n}$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel(r"Bernstein correction $t_n(\delta=0.05)$")
    ax.set_title("Confidence-bound width scales as $1/\sqrt{n}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_exact_kl_dist(power_dir: Path, out_path: Path):
    """Histogram of exact per-prompt KL values."""
    with open(power_dir / "exact_kl.csv", "r", encoding="utf-8") as f:
        kl = [float(row["kl_exact"]) for row in csv.DictReader(f)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(kl, bins=20, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(np.median(kl), color="red", ls="--", label=f"median = {np.median(kl):.3f}")
    ax.set_xlabel(r"True token-level $\mathrm{KL}(P_{\mathrm{risky}}\,\|\,P_{\mathrm{safe}})$")
    ax.set_ylabel("Number of prompts")
    ax.set_title(f"Distribution of true KL across {len(kl)} prompts")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--power_dir", type=str, required=True,
                   help="Power experiment dir (e.g. runs/power_full)")
    p.add_argument("--null_dir", type=str, default=None,
                   help="Optional null-pair dir (e.g. runs/power_null)")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    power_dir = Path(args.power_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (power_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(power_dir)
    plot_power_curves(summary, out_dir / "power_curves.pdf",
                      title="Auditor power & calibration (gpt2-medium vs gpt2)")

    rows = load_trials(power_dir / "trials.csv")
    plot_calibration_scatter(rows, out_dir / "calibration_scatter.pdf")
    plot_bernstein_gap(rows, out_dir / "bernstein_gap.pdf")
    plot_exact_kl_dist(power_dir, out_dir / "exact_kl_dist.pdf")

    if args.null_dir:
        null_summary = load_summary(Path(args.null_dir))
        plot_power_curves(null_summary, out_dir / "power_curves_null.pdf",
                          title="Auditor on null pair (gpt2 vs gpt2): Type-I sanity")


if __name__ == "__main__":
    main()
