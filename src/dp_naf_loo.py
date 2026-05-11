"""
Paired leave-one-out DP-NAF test.

The mid-report Eq. 5 claim is: if a fine-tuning mechanism M is eps-DP w.r.t.
a leave-one-out swap of one training document C, then for any prompt x,

    KL( P_M(D)(. | x)  ||  P_M(D \\ {C})(. | x) )  <=  eps^2 / 2.

This is the CORRECT pairing -- M is run twice on adjacent datasets.  The
'plain' DP-NAF script (dp_naf_experiment.py) instead audits KL to the
pretrained baseline, which is NOT what eps-DP bounds; we use this script to
also report the right quantity.

Implementation:
  * Set torch.manual_seed(K) before each training run so the noise samples
    are paired between the two runs.
  * Train M_full on D (all blocks) with DP-SGD.
  * Train M_loo  on D \\ {first block} with DP-SGD, identical seed.
  * For 20 prompts, audit KL(M_full || M_loo) with the Bernstein bound and
    compare to eps^2 / 2.

Usage:
  python -m src.dp_naf_loo --output_dir runs/dp_naf_loo --epsilons 8 2 --n_epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dp_naf_experiment import TextChunkDataset, collate, train_dp_sgd, audit_kl

logger = logging.getLogger(__name__)


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out / "log.txt", mode="w", encoding="utf-8")],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text = Path(args.text_file).read_text(encoding="utf-8")
    full_ds = TextChunkDataset(text, tokenizer, block_size=args.block_size)
    loo_ds = Subset(full_ds, list(range(1, len(full_ds))))  # remove first block
    logger.info("Full dataset: %d blocks; LOO: %d blocks", len(full_ds), len(loo_ds))

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        audit_prompts = [line.strip() for line in f if line.strip()][: args.num_audit_prompts]

    rows: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for eps in args.epsilons:
        logger.info("\n========== eps = %g ==========", eps)
        # --- Train M_full ---
        torch.manual_seed(args.seed)
        m_full = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        dl_full = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        t0 = time.time()
        losses_f, eps_f, m_full = train_dp_sgd(
            m_full, dl_full, device, args.n_epochs, args.lr, eps, args.delta_dp,
            max_grad_norm=args.max_grad_norm,
        )
        logger.info("M_full   final loss=%.3f  achieved_eps=%.3f  (%.1fs)",
                    losses_f[-1], eps_f, time.time() - t0)

        # --- Train M_loo with paired seed ---
        torch.manual_seed(args.seed)
        m_loo = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        dl_loo = DataLoader(loo_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        t0 = time.time()
        losses_l, eps_l, m_loo = train_dp_sgd(
            m_loo, dl_loo, device, args.n_epochs, args.lr, eps, args.delta_dp,
            max_grad_norm=args.max_grad_norm,
        )
        logger.info("M_loo    final loss=%.3f  achieved_eps=%.3f  (%.1fs)",
                    losses_l[-1], eps_l, time.time() - t0)

        # --- Audit KL(M_full || M_loo) ---
        bound = (max(eps_f, eps_l) ** 2) / 2
        logger.info("Auditing KL(M_full || M_loo); theoretical bound eps^2/2 = %.3f", bound)
        for i, p in enumerate(audit_prompts):
            r = audit_kl(m_full, m_loo, tokenizer, p,
                         n_samples=args.audit_samples, delta=args.audit_delta,
                         seed=2000 + i, device=device)
            r.update({"prompt_idx": i, "target_eps": eps,
                      "eps_full": eps_f, "eps_loo": eps_l, "bound_eps2_2": bound})
            rows.append(r)

        kls = [r["kl_exact"] for r in rows if r["target_eps"] == eps]
        lbs = [r["lb"] for r in rows if r["target_eps"] == eps]
        s = {
            "target_eps": eps,
            "achieved_eps_full": eps_f,
            "achieved_eps_loo": eps_l,
            "theoretical_bound_eps2_2": bound,
            "kl_exact_median": float(sorted(kls)[len(kls)//2]),
            "kl_exact_max": float(max(kls)),
            "lb_median": float(sorted(lbs)[len(lbs)//2]),
            "lb_max": float(max(lbs)),
            "violations_kl_exceeds_bound": int(sum(1 for v in kls if v > bound)),
            "violations_lb_exceeds_bound": int(sum(1 for v in lbs if v > bound)),
            "n_prompts": len(audit_prompts),
        }
        summary[f"eps={eps}"] = s
        logger.info("  Bound eps^2/2 = %.3f", bound)
        logger.info("  KL_exact      median=%.4f max=%.4f  (#violations: %d/%d)",
                    s["kl_exact_median"], s["kl_exact_max"],
                    s["violations_kl_exceeds_bound"], len(audit_prompts))
        logger.info("  Audit LB      median=%.4f max=%.4f  (#audit-violations: %d/%d)",
                    s["lb_median"], s["lb_max"],
                    s["violations_lb_exceeds_bound"], len(audit_prompts))

        del m_full, m_loo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rows:
        with open(out / "trial_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "by_eps": summary}, f, indent=2, default=str)
    logger.info("Wrote %s", out / "summary.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--text_file", type=str, default="data/dp_finetune_text.txt")
    p.add_argument("--prompts_file", type=str, default="data/sample_prompts.txt")
    p.add_argument("--num_audit_prompts", type=int, default=20)
    p.add_argument("--epsilons", type=float, nargs="+", default=[8.0, 2.0])
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--delta_dp", type=float, default=1e-5)
    p.add_argument("--audit_samples", type=int, default=1000)
    p.add_argument("--audit_delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="runs/dp_naf_loo")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
