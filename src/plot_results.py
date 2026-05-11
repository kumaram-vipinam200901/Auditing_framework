"""
Plotting utilities for NAF audit results.

Reads results.jsonl and produces:
  1. KL distribution histogram with lower bounds and violation threshold
  2. Per-round best-LB progression (evolutionary search)
  3. Comparison of random baseline vs evolutionary search
  4. Violin / box plot of KL estimates across rounds

Usage:
    python -m src.plot_results --input runs/<run_id>/results.jsonl
    python -m src.plot_results --input runs/<run_id>/results.jsonl --baseline runs/baseline_<id>/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------------
# Plot 1: KL distribution histogram
# ------------------------------------------------------------------

def plot_kl_distribution(
    records: List[Dict],
    claimed_K: float,
    out_path: Path,
    title: str = "Distribution of KL Estimates",
):
    """Histogram of KL means and lower bounds with the claimed-K threshold."""
    kl_means = [r["kl_mean"] for r in records]
    lower_bounds = [r["lower_bound"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KL means
    ax = axes[0]
    ax.hist(kl_means, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(claimed_K, color="red", linestyle="--", linewidth=2, label=f"Claimed K={claimed_K}")
    ax.set_xlabel("KL Divergence (mean estimate)")
    ax.set_ylabel("Count")
    ax.set_title("KL Mean Estimates")
    ax.legend()

    # Lower bounds
    ax = axes[1]
    violations = [lb for lb in lower_bounds if lb > claimed_K]
    non_violations = [lb for lb in lower_bounds if lb <= claimed_K]
    ax.hist(non_violations, bins=30, color="steelblue", edgecolor="white", alpha=0.8, label="No violation")
    if violations:
        ax.hist(violations, bins=max(1, len(violations) // 3 + 1), color="crimson",
                edgecolor="white", alpha=0.8, label="Violation")
    ax.axvline(claimed_K, color="red", linestyle="--", linewidth=2, label=f"K={claimed_K}")
    ax.set_xlabel("KL Lower Bound (Bernstein)")
    ax.set_ylabel("Count")
    ax.set_title("KL Lower Bounds (auditing)")
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ------------------------------------------------------------------
# Plot 2: Per-round progression
# ------------------------------------------------------------------

def plot_round_progression(
    records: List[Dict],
    claimed_K: float,
    out_path: Path,
):
    """Line plot of best lower bound and mean KL per round."""
    rounds = sorted(set(r.get("round", 0) for r in records))
    best_lb = []
    best_mean = []
    for rd in rounds:
        rd_recs = [r for r in records if r.get("round", 0) == rd]
        best_lb.append(max(r["lower_bound"] for r in rd_recs))
        best_mean.append(max(r["kl_mean"] for r in rd_recs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, best_lb, "o-", color="crimson", linewidth=2, markersize=8, label="Best Lower Bound")
    ax.plot(rounds, best_mean, "s--", color="steelblue", linewidth=2, markersize=7, label="Best KL Mean")
    ax.axhline(claimed_K, color="gray", linestyle=":", linewidth=1.5, label=f"Claimed K={claimed_K}")
    ax.set_xlabel("Round")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Evolutionary Search: Per-Round Progression")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ------------------------------------------------------------------
# Plot 3: Baseline vs Evolutionary comparison
# ------------------------------------------------------------------

def plot_baseline_comparison(
    evo_records: List[Dict],
    baseline_records: List[Dict],
    claimed_K: float,
    out_path: Path,
):
    """Side-by-side comparison of KL lower bounds."""
    evo_lb = [r["lower_bound"] for r in evo_records]
    base_lb = [r["lower_bound"] for r in baseline_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [base_lb, evo_lb],
        labels=["Random Baseline", "Evolutionary Search"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("salmon")
    ax.axhline(claimed_K, color="red", linestyle="--", linewidth=1.5, label=f"Claimed K={claimed_K}")
    ax.set_ylabel("KL Lower Bound")
    ax.set_title("Random vs Evolutionary: KL Lower Bounds")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ------------------------------------------------------------------
# Plot 4: Per-round violin plot
# ------------------------------------------------------------------

def plot_round_violin(
    records: List[Dict],
    claimed_K: float,
    out_path: Path,
):
    """Violin plot of KL lower bounds per round."""
    rounds = sorted(set(r.get("round", 0) for r in records))
    data = []
    for rd in rounds:
        rd_lbs = [r["lower_bound"] for r in records if r.get("round", 0) == rd]
        data.append(rd_lbs)

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=rounds, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.axhline(claimed_K, color="red", linestyle="--", linewidth=1.5, label=f"Claimed K={claimed_K}")
    ax.set_xlabel("Round")
    ax.set_ylabel("KL Lower Bound")
    ax.set_title("KL Lower Bounds per Evolutionary Round")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot NAF audit results")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to results.jsonl from an audit run")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to results.jsonl from a baseline run (for comparison)")
    parser.add_argument("--claimed_K", type=float, default=None,
                        help="Override claimed K (default: read from data)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_jsonl(input_path)
    if not records:
        logger.error("No records found in %s", input_path)
        sys.exit(1)

    claimed_K = args.claimed_K if args.claimed_K is not None else records[0].get("claimed_K", 1.0)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d records, claimed_K=%.4f", len(records), claimed_K)

    # Plot 1: distribution
    plot_kl_distribution(records, claimed_K, out_dir / "kl_distribution.png")

    # Plot 2: round progression (if multi-round)
    rounds = set(r.get("round", 0) for r in records)
    if len(rounds) > 1:
        plot_round_progression(records, claimed_K, out_dir / "round_progression.png")
        plot_round_violin(records, claimed_K, out_dir / "round_violin.png")

    # Plot 3: baseline comparison
    if args.baseline:
        baseline_records = load_jsonl(args.baseline)
        if baseline_records:
            plot_baseline_comparison(
                records, baseline_records, claimed_K,
                out_dir / "baseline_comparison.png",
            )

    logger.info("All plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
