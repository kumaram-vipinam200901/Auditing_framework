"""
Anchored decoding tightness analysis.

The mid-report headline says "anchored decoding has 0 violations across all K."
This script makes that claim much stronger by quantifying *how much room* it
leaves under the budget K.

For each (prompt, K) we measure three quantities:
  (a) KL_risky    = KL(P_risky ||  P_safe)        -- exact, summed over V
                    (the violation if no protection were applied)
  (b) KL_anchored = KL(q_theta || P_safe)         -- exact, summed over V
                    (the actual divergence the algorithm achieves)
  (c) LB_anchored = Bernstein lower bound from auditing q_theta vs P_safe
                    (what our auditor would report)

A correct anchored decoding implementation must satisfy KL_anchored <= K.
The "tightness" is K - KL_anchored. Small => uses full budget; large => over-anchored.

The "audit slack" is K - LB_anchored. Comparing this to the auditor's known
detection power (from power_experiment.py) tells us the strongest violation
that could possibly hide undetected.

Usage:
  python -m src.anchored_tightness --num_prompts 30 --num_samples 1000 \\
      --K_grid 0.5 1.0 2.0 5.0 --output_dir runs/anchored_tightness
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .config import HardwareConfig, set_seed
from .kl_estimator import empirical_bernstein_bound
from .models import AnchoredDecodingModel, LanguageModel, load_model_pair
from .power_experiment import exact_token_kl

logger = logging.getLogger(__name__)


@torch.inference_mode()
def exact_kl_anchored(
    prompt: str,
    risky: LanguageModel,
    safe: LanguageModel,
    anchored: AnchoredDecodingModel,
) -> Tuple[float, float, float]:
    """Return (kl_risky_vs_safe, kl_anchored_vs_safe, theta).

    All KLs are computed exactly by summing over the full vocabulary
    (since the next-token distributions are explicit softmax).
    """
    ids = risky.encode(prompt)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    log_p_risky = risky.next_token_log_probs(ids).float()  # (1, V)
    log_p_safe = safe.next_token_log_probs(ids).float()    # (1, V)

    # KL(P_risky || P_safe) exact
    p = log_p_risky.exp().double()
    diff_r = log_p_risky.double() - log_p_safe.double()
    kl_risky = (p * diff_r).sum(dim=-1).item()

    # Solve for theta and form q
    w_d = anchored._solve_theta(log_p_safe, log_p_risky, anchored.K)  # (1,1)
    log_q_unnorm = (1 - w_d) * log_p_safe + w_d * log_p_risky
    log_q = torch.log_softmax(log_q_unnorm, dim=-1)

    # KL(q || P_safe) exact
    q = log_q.exp().double()
    diff_q = log_q.double() - log_p_safe.double()
    kl_q_safe = (q * diff_q).sum(dim=-1).item()

    return float(kl_risky), float(kl_q_safe), float(w_d.item())


@torch.inference_mode()
def audit_anchored(
    prompt: str,
    safe: LanguageModel,
    anchored: AnchoredDecodingModel,
    n_samples: int,
    delta: float,
    seed: int,
) -> Tuple[float, float, float]:
    """Run the auditor on (q_theta, P_safe). Returns (z_bar, var, lb)."""
    g = torch.Generator(device=anchored.device).manual_seed(seed)
    ids = anchored.encode(prompt)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    log_q = anchored._fused_log_probs(ids).squeeze(0)            # (V,)
    log_safe = safe.next_token_log_probs(ids).squeeze(0)          # (V,)

    probs = log_q.exp()
    sampled = torch.multinomial(probs, num_samples=n_samples, replacement=True, generator=g)
    z = (log_q[sampled] - log_safe[sampled]).cpu()
    z_bar, var, lb, ub, corr = empirical_bernstein_bound(z, delta)
    return float(z_bar), float(var), float(lb)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(output_dir / "log.txt"), mode="w", encoding="utf-8"),
        ],
    )

    set_seed(args.seed)
    hw = HardwareConfig(device=args.device, mixed_precision="none", torch_compile=False)
    risky, safe = load_model_pair(args.risky, args.safe, hw)

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()][: args.num_prompts]
    logger.info("Loaded %d prompts", len(prompts))

    rows: List[Dict] = []
    t0 = time.time()
    for K in args.K_grid:
        logger.info("=== Auditing anchored decoding at K = %.2f ===", K)
        anchored = AnchoredDecodingModel(risky, safe, K=K)
        for i, p in enumerate(prompts):
            kl_risky, kl_anchored, theta = exact_kl_anchored(p, risky, safe, anchored)
            z_bar, var, lb = audit_anchored(p, safe, anchored, args.num_samples, args.delta, args.seed * 1000 + i)
            rows.append({
                "K": K,
                "prompt_idx": i,
                "kl_risky_vs_safe": kl_risky,
                "kl_anchored_vs_safe": kl_anchored,
                "theta": theta,
                "audit_zbar": z_bar,
                "audit_var": var,
                "audit_lb": lb,
                "violates_exact": kl_anchored > K + 1e-6,
                "violates_audit": lb > K,
                "budget_utilization": kl_anchored / K if K > 0 else 0.0,
            })
        logger.info("  K=%.2f done. Median KL_anchored=%.4f / K=%.2f (utilization=%.2f%%)",
                    K,
                    sorted(r["kl_anchored_vs_safe"] for r in rows if r["K"] == K)[len(prompts)//2],
                    K,
                    100 * sorted(r["budget_utilization"] for r in rows if r["K"] == K)[len(prompts)//2])
    logger.info("Total time: %.1fs", time.time() - t0)

    # Save results
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate summary
    summary = {"config": vars(args), "by_K": {}}
    for K in args.K_grid:
        subset = [r for r in rows if r["K"] == K]
        kl_anch = [r["kl_anchored_vs_safe"] for r in subset]
        kl_risk = [r["kl_risky_vs_safe"] for r in subset]
        lbs = [r["audit_lb"] for r in subset]
        utils = [r["budget_utilization"] for r in subset]
        summary["by_K"][f"{K:.2f}"] = {
            "kl_risky_vs_safe": {
                "median": sorted(kl_risk)[len(kl_risk)//2],
                "max": max(kl_risk),
                "fraction_above_K": sum(1 for v in kl_risk if v > K) / len(kl_risk),
            },
            "kl_anchored_vs_safe": {
                "median": sorted(kl_anch)[len(kl_anch)//2],
                "max": max(kl_anch),
                "fraction_above_K": sum(1 for v in kl_anch if v > K + 1e-6) / len(kl_anch),
            },
            "budget_utilization_median": sorted(utils)[len(utils)//2],
            "audit_lb": {
                "median": sorted(lbs)[len(lbs)//2],
                "max": max(lbs),
                "fraction_above_K": sum(1 for v in lbs if v > K) / len(lbs),
            },
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n========== ANCHORED DECODING TIGHTNESS ==========")
    for K in args.K_grid:
        s = summary["by_K"][f"{K:.2f}"]
        logger.info("K = %.2f", K)
        logger.info("  KL_risky        median=%.3f  max=%.3f  frac>K=%.2f%%",
                    s["kl_risky_vs_safe"]["median"], s["kl_risky_vs_safe"]["max"],
                    100*s["kl_risky_vs_safe"]["fraction_above_K"])
        logger.info("  KL_anchored     median=%.3f  max=%.3f  frac>K=%.2f%% (should be 0)",
                    s["kl_anchored_vs_safe"]["median"], s["kl_anchored_vs_safe"]["max"],
                    100*s["kl_anchored_vs_safe"]["fraction_above_K"])
        logger.info("  Budget util.    median=%.2f%%", 100*s["budget_utilization_median"])
        logger.info("  Audit LB        median=%.3f  max=%.3f  frac>K=%.2f%% (auditor flags)",
                    s["audit_lb"]["median"], s["audit_lb"]["max"],
                    100*s["audit_lb"]["fraction_above_K"])
    logger.info("\nSaved %s and %s", output_dir / "results.csv", output_dir / "summary.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--risky", type=str, default="gpt2-medium")
    p.add_argument("--safe", type=str, default="gpt2")
    p.add_argument("--num_prompts", type=int, default=30)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--K_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--prompts_file", type=str, default="data/sample_prompts.txt")
    p.add_argument("--output_dir", type=str, default="runs/anchored_tightness")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
