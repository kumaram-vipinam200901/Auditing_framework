"""
DP -> NAF empirical study.

Theory (mid-report Eq. 5):
  If a fine-tuning mechanism M is eps-DP w.r.t. the leave-one-out swap,
  then for any prompt x:
      KL( P_finetune(. | x)  ||  P_pretrain(. | x) )  <=  eps^2 / 2.
  (eps-DP -> eps^2/2-zCDP; combined with post-processing this gives the
  per-prompt KL bound.)

So a DP-fine-tuned model is automatically (eps^2/2)-NAF.  This script
*audits* that prediction:

  1. Fine-tune GPT-2 on a small "copyright-prone" dataset (famous opening
     lines repeated many epochs) under three regimes:
         (a) NON-PRIVATE: standard SGD.  KL can be arbitrarily large.
         (b) DP-SGD with eps ~ 8.
         (c) DP-SGD with eps ~ 2.
     The "safe" baseline is the original pretrained GPT-2.
  2. For each fine-tuned model, run the auditor on a fixed prompt set.
  3. Compare measured KL lower bound to the theoretical bound eps^2/2.

Usage:
  python -m src.dp_naf_experiment --output_dir runs/dp_naf
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kl_estimator import empirical_bernstein_bound

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Tokenized dataset
# ---------------------------------------------------------------------

class TextChunkDataset(Dataset):
    """Tokenize text into fixed-length chunks for causal-LM fine-tuning."""

    def __init__(self, text: str, tokenizer, block_size: int = 64):
        ids = tokenizer.encode(text)
        # Pack into non-overlapping blocks
        self.blocks = []
        for i in range(0, len(ids) - block_size, block_size):
            self.blocks.append(torch.tensor(ids[i: i + block_size], dtype=torch.long))
        if not self.blocks:
            # text shorter than block_size -- pad with eos
            self.blocks = [torch.tensor(ids + [tokenizer.eos_token_id] * (block_size - len(ids)),
                                        dtype=torch.long)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def collate(batch):
    return torch.stack(batch, dim=0)


# ---------------------------------------------------------------------
# Training (non-private and DP-SGD)
# ---------------------------------------------------------------------

def train_non_private(
    model,
    dataloader,
    device,
    n_epochs: int,
    lr: float,
):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    losses = []
    step = 0
    for ep in range(n_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 20 == 0:
                logger.info("  [non-DP] step %d  loss=%.3f", step, sum(losses[-20:]) / 20)
    return losses


def train_dp_sgd(
    model,
    dataloader,
    device,
    n_epochs: int,
    lr: float,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float = 1.0,
):
    """DP-SGD via opacus. Returns (loss_history, achieved_epsilon, base_model)."""
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # GPT-2 has tied embeddings which break per-sample gradients in opacus.
    # Untie them by cloning lm_head weights.
    if hasattr(model, "lm_head") and hasattr(model, "transformer"):
        if model.lm_head.weight is model.transformer.wte.weight:
            model.lm_head.weight = torch.nn.Parameter(
                model.transformer.wte.weight.detach().clone()
            )
            logger.info("Untied lm_head from wte for opacus compatibility.")

    # Freeze embeddings to keep training tractable on small data and avoid
    # the high-dim noise blowup from huge embedding matrices.
    for n, p in model.named_parameters():
        if "wte" in n or "wpe" in n:
            p.requires_grad = False

    model = ModuleValidator.fix(model)
    model.train()
    ModuleValidator.validate(model, strict=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr, momentum=0.0)

    pe = PrivacyEngine()
    model, opt, dataloader = pe.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=dataloader,
        epochs=n_epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    losses = []
    step = 0
    for ep in range(n_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            if batch.numel() == 0 or batch.size(0) == 0:
                # Opacus Poisson sampling can produce empty batches on tiny datasets.
                continue
            opt.zero_grad()
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 20 == 0:
                logger.info("  [DP eps=%.1f] step %d  loss=%.3f", target_epsilon, step,
                            sum(losses[-20:]) / 20)
    achieved = pe.get_epsilon(target_delta)
    # Unwrap module so we can use the underlying model directly
    base = model._module if hasattr(model, "_module") else model
    return losses, achieved, base


# ---------------------------------------------------------------------
# Auditing
# ---------------------------------------------------------------------

@torch.inference_mode()
def audit_kl(
    finetuned,
    pretrained,
    tokenizer,
    prompt: str,
    n_samples: int,
    delta: float,
    seed: int,
    device,
) -> Dict[str, float]:
    """Audit KL(P_finetuned(.|x) || P_pretrained(.|x)) for one prompt.

    Returns exact KL (summed over V) and Bernstein lower bound.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Distributions over full vocab
    finetuned.eval()
    pretrained.eval()
    log_p = F.log_softmax(finetuned(input_ids=ids).logits[:, -1, :].float(), dim=-1).squeeze(0)
    log_q = F.log_softmax(pretrained(input_ids=ids).logits[:, -1, :].float(), dim=-1).squeeze(0)

    # Exact KL
    p = log_p.exp().double()
    diff = log_p.double() - log_q.double()
    kl_exact = (p * diff).sum().item()

    # Bernstein audit
    probs = log_p.exp()
    sampled = torch.multinomial(probs, num_samples=n_samples, replacement=True, generator=g)
    z = (log_p[sampled] - log_q[sampled]).cpu()
    z_bar, var, lb, ub, corr = empirical_bernstein_bound(z, delta)
    return {
        "kl_exact": float(kl_exact),
        "z_bar": float(z_bar),
        "var": float(var),
        "lb": float(lb),
        "correction": float(corr),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def fresh_model(model_name: str, device):
    m = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    m.train()
    return m


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
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tune text
    text = Path(args.text_file).read_text(encoding="utf-8")
    ds = TextChunkDataset(text, tokenizer, block_size=args.block_size)
    logger.info("Dataset: %d blocks of %d tokens each", len(ds), args.block_size)

    # Pretrained reference (the "safe" model)
    pretrained = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    pretrained.eval()

    # Audit prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        audit_prompts = [line.strip() for line in f if line.strip()][: args.num_audit_prompts]
    logger.info("Auditing on %d prompts", len(audit_prompts))

    all_audit_rows: List[Dict] = []
    eps_results: Dict[str, Dict] = {}

    for eps in args.epsilons:
        regime = "non-private" if eps <= 0 else f"dp-eps{eps:g}"
        logger.info("\n========== Training regime: %s (epochs=%d, lr=%g) ==========",
                    regime, args.n_epochs, args.lr)
        # Re-init model from pretrained
        ft = fresh_model(args.model_name, device)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        t0 = time.time()
        if eps <= 0:
            losses = train_non_private(ft, dl, device, args.n_epochs, args.lr)
            achieved_eps = float("inf")
        else:
            losses, achieved_eps, ft = train_dp_sgd(
                ft, dl, device, args.n_epochs, args.lr,
                target_epsilon=eps, target_delta=args.delta_dp,
                max_grad_norm=args.max_grad_norm,
            )
        train_time = time.time() - t0
        logger.info("Trained %s in %.1fs (final loss=%.3f, achieved_eps=%.3f)",
                    regime, train_time, losses[-1] if losses else float("nan"), achieved_eps)

        # Audit
        audit_rows = []
        for i, p in enumerate(audit_prompts):
            r = audit_kl(ft, pretrained, tokenizer, p,
                        n_samples=args.audit_samples, delta=args.audit_delta,
                        seed=1000 + i, device=device)
            r["prompt_idx"] = i
            r["regime"] = regime
            r["target_eps"] = eps
            r["achieved_eps"] = achieved_eps
            audit_rows.append(r)
            all_audit_rows.append(r)

        # Aggregate
        kl_exact = [r["kl_exact"] for r in audit_rows]
        lb = [r["lb"] for r in audit_rows]
        thy_bound = (achieved_eps ** 2) / 2 if achieved_eps != float("inf") else float("inf")
        eps_results[regime] = {
            "target_eps": eps,
            "achieved_eps": achieved_eps,
            "theoretical_KL_bound_eps2_over_2": thy_bound,
            "final_loss": float(losses[-1]) if losses else None,
            "train_time_sec": train_time,
            "kl_exact_median": float(sorted(kl_exact)[len(kl_exact) // 2]),
            "kl_exact_max": float(max(kl_exact)),
            "lb_median": float(sorted(lb)[len(lb) // 2]),
            "lb_max": float(max(lb)),
            "any_violates_thy_bound": bool(any(v > thy_bound for v in kl_exact)) if math.isfinite(thy_bound) else False,
        }

        logger.info("Audit summary [%s]:", regime)
        logger.info("  Theoretical bound eps^2/2 = %.4f", thy_bound)
        logger.info("  KL_exact   median=%.4f  max=%.4f", eps_results[regime]["kl_exact_median"],
                    eps_results[regime]["kl_exact_max"])
        logger.info("  Audit LB   median=%.4f  max=%.4f", eps_results[regime]["lb_median"],
                    eps_results[regime]["lb_max"])
        logger.info("  Any KL_exact > eps^2/2 ?  %s", eps_results[regime]["any_violates_thy_bound"])

        # Free model memory
        del ft
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save
    with open(out / "trial_results.csv", "w", newline="", encoding="utf-8") as f:
        if all_audit_rows:
            w = csv.DictWriter(f, fieldnames=list(all_audit_rows[0].keys()))
            w.writeheader()
            w.writerows(all_audit_rows)

    summary = {
        "config": vars(args),
        "by_regime": eps_results,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote %s", out / "summary.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--text_file", type=str, default="data/dp_finetune_text.txt")
    p.add_argument("--prompts_file", type=str, default="data/sample_prompts.txt")
    p.add_argument("--num_audit_prompts", type=int, default=20)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 8.0, 2.0],
                   help="Target epsilons; 0 = non-private")
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--delta_dp", type=float, default=1e-5)
    p.add_argument("--audit_samples", type=int, default=1000)
    p.add_argument("--audit_delta", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="runs/dp_naf")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
