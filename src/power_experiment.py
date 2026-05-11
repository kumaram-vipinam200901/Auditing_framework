"""
Power / calibration experiment for the NAF auditor.

This is the standard auditing-paper sanity check, modelled on Jagielski et al.
(NeurIPS 2020) for DP. An auditor's "0 violations on anchored decoding" result
is only meaningful if we know the auditor *can* detect violations when they
exist. Here we measure:

  (T1) Type-I error (false-positive rate)  Pr[ LB > K | KL_true <= K ]
        Should be <= delta by Bernstein. Validates correctness.

  (T2) Power (true-positive rate)          Pr[ LB > K | KL_true >  K ]
        As a function of (n, KL_true - K). Measures how strong the auditor is.

Crucial trick: at the next-token level, both P_risky(.|x) and P_safe(.|x) are
explicit softmax distributions over the full vocabulary (~50K tokens for GPT-2).
So we can compute the EXACT token-level KL once, and use it as an oracle
ground-truth, vs the n-sample auditor's lower bound.

Outputs:
  runs/power/exact_vs_lb.csv        per-prompt true KL, mean LB, mean correction
  runs/power/trial_results.csv      per-trial (prompt, n, seed, LB, mean, var)
  runs/power/calibration.json       Type-I error per (n, K) cell
  runs/power/power.json             Power per (n, K) cell

Usage:
  python -m src.power_experiment --num_prompts 20 --num_trials 100 \\
      --risky gpt2-medium --safe gpt2 \\
      --output_dir runs/power
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import HardwareConfig, resolve_device, set_seed
from .kl_estimator import empirical_bernstein_bound
from .models import LanguageModel, load_model_pair

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exact token-level KL (the oracle)
# -----------------------------------------------------------------------------

@torch.inference_mode()
def exact_token_kl(prompt: str, risky: LanguageModel, safe: LanguageModel) -> Tuple[float, float]:
    """Compute the EXACT next-token KL(P_risky || P_safe) by summing over V.

    Returns (kl, var_under_p) where var_under_p is Var_{Y~P_risky}[log P_risky(Y) - log P_safe(Y)].
    The variance is needed because the Bernstein correction scales as sqrt(V_n / n),
    so prompts with large variance produce looser LBs.
    """
    ids = risky.encode(prompt)
    log_p = risky.next_token_log_probs(ids).squeeze(0).double()   # (V,)
    log_q = safe.next_token_log_probs(ids).squeeze(0).double()    # (V,)
    p = log_p.exp()
    diff = log_p - log_q
    # KL = sum_y p(y) * diff(y)
    kl = (p * diff).sum().item()
    # Var_{Y~p}[diff(Y)] = E_p[diff^2] - (E_p[diff])^2
    e_diff2 = (p * diff * diff).sum().item()
    var = e_diff2 - kl * kl
    return float(kl), float(max(var, 0.0))


# -----------------------------------------------------------------------------
# Sampling-based audit (one trial)
# -----------------------------------------------------------------------------

@torch.inference_mode()
def audit_trial_cached(
    log_p_full: torch.Tensor,   # (V,) on device
    log_q_full: torch.Tensor,   # (V,) on device
    probs: torch.Tensor,        # (V,) on device, == log_p_full.exp()
    n: int,
    delta: float,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Run one Monte-Carlo audit trial using cached per-prompt distributions.
    Returns (z_bar, var, lower_bound, correction).
    """
    g = torch.Generator(device=log_p_full.device).manual_seed(seed)
    sampled = torch.multinomial(probs, num_samples=n, replacement=True, generator=g)
    z = (log_p_full[sampled] - log_q_full[sampled]).cpu()
    z_bar, var, lb, ub, corr = empirical_bernstein_bound(z, delta)
    return float(z_bar), float(var), float(lb), float(corr)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(output_dir / "power_log.txt"), mode="w", encoding="utf-8"),
        ],
    )

    set_seed(args.seed)

    # Map fp32 -> "none" (full precision) for HardwareConfig
    precision = "none" if args.precision in ("fp32", "none") else args.precision
    hw = HardwareConfig(
        device=args.device,
        mixed_precision=precision,
        torch_compile=False,
    )
    logger.info("Loading models: risky=%s, safe=%s", args.risky, args.safe)
    risky, safe = load_model_pair(args.risky, args.safe, hw)

    # Load prompts
    logger.info("Loading prompts from %s", args.prompts_file)
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()][: args.num_prompts]
    logger.info("Loaded %d prompts", len(prompts))

    # Sample sizes and claimed K values to evaluate
    n_grid = args.n_grid
    delta = args.delta

    # ----- Step 1: compute exact KL for every prompt (the oracle) -----
    logger.info("Computing exact token-level KL for %d prompts (this is fast)", len(prompts))
    t0 = time.time()
    exact_records = []
    for i, p in enumerate(prompts):
        kl, var = exact_token_kl(p, risky, safe)
        exact_records.append({"prompt_idx": i, "prompt": p[:80], "kl_exact": kl, "var_under_p": var})
        if (i + 1) % 5 == 0:
            logger.info("  exact KL [%d/%d]: median so far = %.3f", i + 1, len(prompts),
                        sorted(r["kl_exact"] for r in exact_records)[len(exact_records) // 2])
    logger.info("Exact KL phase done in %.1fs", time.time() - t0)

    with open(output_dir / "exact_kl.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "prompt", "kl_exact", "var_under_p"])
        w.writeheader()
        w.writerows(exact_records)

    # ----- Step 2: Monte-Carlo trials (per-prompt cache => 1 fwd pass per prompt) -----
    logger.info("Running %d MC trials per prompt across %d sample sizes",
                args.num_trials, len(n_grid))
    trial_path = output_dir / "trials.csv"
    with open(trial_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "kl_exact", "n", "seed", "z_bar", "var", "lb", "correction"])
        t0 = time.time()
        total = len(prompts) * len(n_grid) * args.num_trials
        done = 0
        for i, p in enumerate(prompts):
            kl_ex = exact_records[i]["kl_exact"]
            ids = risky.encode(p)
            log_p_full = risky.next_token_log_probs(ids).squeeze(0)
            log_q_full = safe.next_token_log_probs(ids).squeeze(0)
            probs = log_p_full.exp()
            for n in n_grid:
                for trial in range(args.num_trials):
                    seed_t = args.seed * 1_000_000 + i * 10_000 + n * 100 + trial
                    z_bar, var, lb, corr = audit_trial_cached(
                        log_p_full, log_q_full, probs, n, delta, seed_t,
                    )
                    writer.writerow([i, f"{kl_ex:.6f}", n, seed_t,
                                     f"{z_bar:.6f}", f"{var:.6f}",
                                     f"{lb:.6f}", f"{corr:.6f}"])
                    done += 1
                    if done % 1000 == 0:
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 1e-9)
                        eta = (total - done) / max(rate, 1e-9)
                        logger.info("  trials [%d/%d]  rate=%.1f/s  ETA=%.0fs", done, total, rate, eta)
            f.flush()
    logger.info("MC trials done in %.1fs", time.time() - t0)

    # ----- Step 3: aggregate calibration & power -----
    logger.info("Aggregating calibration and power")
    # Group by prompt and n
    import collections
    grouped: Dict[Tuple[int, int], List[Tuple[float, float]]] = collections.defaultdict(list)
    with open(trial_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pi = int(row["prompt_idx"])
            n = int(row["n"])
            lb = float(row["lb"])
            kl = float(row["kl_exact"])
            grouped[(pi, n)].append((lb, kl))

    # K-grid: relative to each prompt's exact KL
    # K_factor in {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
    # K = K_factor * kl_exact  (K=0 means audit always rejects; K>>kl_exact means never reject)
    k_factors = [0.0, 0.25, 0.5, 0.75, 0.95, 1.05, 1.25, 1.5, 2.0]

    summary = {
        "config": {
            "risky": args.risky,
            "safe": args.safe,
            "delta": delta,
            "num_prompts": len(prompts),
            "num_trials": args.num_trials,
            "n_grid": n_grid,
            "k_factors": k_factors,
        },
        "by_n": {},
    }

    for n in n_grid:
        # aggregated power curve: for each K_factor, fraction of trials where LB > K_factor * kl_exact
        # split into "true K-NAF" cells (K_factor >= 1) for Type-I, and "violating" (K_factor < 1) for power
        cell_counts = {f"{kf:.2f}": [0, 0] for kf in k_factors}  # [n_reject, n_total]
        for (pi, n_p), trials in grouped.items():
            if n_p != n:
                continue
            for lb, kl in trials:
                for kf in k_factors:
                    K = kf * kl
                    cell_counts[f"{kf:.2f}"][1] += 1
                    if lb > K:
                        cell_counts[f"{kf:.2f}"][0] += 1
        rates = {}
        for kf, (rej, tot) in cell_counts.items():
            rates[kf] = {"reject_rate": rej / max(tot, 1), "n_total": tot, "n_reject": rej}
        summary["by_n"][str(n)] = rates

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print headline numbers
    logger.info("\n========== CALIBRATION & POWER SUMMARY ==========")
    logger.info("delta = %.3f  (Bernstein guarantee: Pr[LB > KL_true] <= delta)", delta)
    logger.info("False-positive cells: K_factor >= 1.0 (claimed K >= true KL)")
    logger.info("Power cells:         K_factor <  1.0 (claimed K <  true KL)")
    for n in n_grid:
        logger.info("\n--- n = %d ---", n)
        for kf in k_factors:
            r = summary["by_n"][str(n)][f"{kf:.2f}"]
            tag = "Type-I" if kf >= 1.0 else "Power"
            logger.info("  K = %.2f * KL_true   reject = %4d/%4d = %.3f   [%s]",
                        kf, r["n_reject"], r["n_total"], r["reject_rate"], tag)

    logger.info("\nWrote: %s", output_dir / "summary.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--risky", type=str, default="gpt2-medium")
    p.add_argument("--safe", type=str, default="gpt2")
    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--num_trials", type=int, default=100)
    p.add_argument("--n_grid", type=int, nargs="+", default=[50, 100, 500, 1000, 5000])
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--precision", type=str, default="fp32")  # fp32 for numerical stability
    p.add_argument("--prompts_file", type=str, default="data/sample_prompts.txt")
    p.add_argument("--output_dir", type=str, default="runs/power")
    return p.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
