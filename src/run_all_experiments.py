"""
Comprehensive experiment suite for conference-quality NAF auditing paper.

Runs 8 experiment groups across 3 model families, 2 datasets, and multiple
ablation axes.  Total: ~50 sub-experiments, estimated 1-3 hours on CPU.

IMPORTANT — Tokeniser compatibility
------------------------------------
KL estimation requires that both models share the same vocabulary so that
token IDs have identical meaning.  We therefore group model pairs *within*
the same tokeniser family:
  • GPT-2 family   : gpt2, gpt2-medium, gpt2-large, gpt2-xl
  • Pythia family   : EleutherAI/pythia-70m … pythia-1b  (GPT-NeoX tokeniser)
  • OPT family      : facebook/opt-125m … opt-1.3b      (OPT tokeniser)
Cross-family pairs are INVALID and intentionally excluded.

Usage:
    python -m src.run_all_experiments --config configs/comprehensive.yaml
    python -m src.run_all_experiments --config configs/comprehensive.yaml --quick
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import AuditConfig, DatasetConfig, load_config, resolve_device, set_seed
from .dataset import load_prompts
from .evolutionary_search import evaluate_pool, run_evolutionary_search, _estimate_to_dict
from .models import LanguageModel

logger = logging.getLogger(__name__)


def _setup_logging(log_file: str | None, output_dir: Path):
    """Configure logging to console + file."""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    if log_file is None:
        log_file = str(output_dir / "experiment_log.txt")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    logger.info("Logging to console + %s", log_file)


# ======================================================================
# Model pair definitions  (risky, safe, tag, family)
# Models within a family share the same tokeniser → cross-family is invalid
# ======================================================================

GPT2_PAIRS: List[Tuple[str, str, str]] = [
    ("gpt2",        "gpt2",        "gpt2_vs_gpt2_sanity"),      # sanity: KL≈0
    ("gpt2-medium", "gpt2",        "gpt2med_vs_gpt2"),          # 355M vs 124M
    ("gpt2-large",  "gpt2",        "gpt2large_vs_gpt2"),        # 774M vs 124M
    ("gpt2-large",  "gpt2-medium", "gpt2large_vs_gpt2med"),     # 774M vs 355M
]

PYTHIA_PAIRS: List[Tuple[str, str, str]] = [
    ("EleutherAI/pythia-160m", "EleutherAI/pythia-70m",  "pythia160m_vs_70m"),
    ("EleutherAI/pythia-410m", "EleutherAI/pythia-70m",  "pythia410m_vs_70m"),
    ("EleutherAI/pythia-410m", "EleutherAI/pythia-160m", "pythia410m_vs_160m"),
]

OPT_PAIRS: List[Tuple[str, str, str]] = [
    ("facebook/opt-350m", "facebook/opt-125m", "opt350m_vs_125m"),
]

ALL_FAMILIES = [
    ("GPT-2",  GPT2_PAIRS),
    ("Pythia", PYTHIA_PAIRS),
    ("OPT",    OPT_PAIRS),
]

# Ablation axes
SAMPLE_COUNTS       = [50, 100, 200, 500, 1000, 2000, 5000]
ROUND_COUNTS        = [1, 2, 3, 5, 10, 15]
K_THRESHOLDS        = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DELTA_VALUES        = [0.01, 0.05, 0.10, 0.20]
MUTATION_ABLATIONS  = {
    "all":            ["verbatim_instruction", "quote_wrap", "style_tweak",
                       "punctuation_tweak", "crossover", "prefix_instruction", "format_change"],
    "no_crossover":   ["verbatim_instruction", "quote_wrap", "style_tweak",
                       "punctuation_tweak", "prefix_instruction", "format_change"],
    "verbatim_only":  ["verbatim_instruction"],
    "format_only":    ["quote_wrap", "format_change", "prefix_instruction"],
    "tweak_only":     ["style_tweak", "punctuation_tweak"],
    "crossover_only": ["crossover"],
}

# Quick-mode subsets
SAMPLE_COUNTS_QUICK = [50, 500, 2000]
ROUND_COUNTS_QUICK  = [1, 5]
K_THRESHOLDS_QUICK  = [0.5, 1.0, 5.0]


# ======================================================================
# Helper: run one sub-experiment
# ======================================================================

def _run_single(
    tag: str,
    cfg: AuditConfig,
    risky: LanguageModel,
    safe: LanguageModel,
    prompts: List[str],
    master_dir: Path,
    do_evolutionary: bool = True,
) -> Dict[str, Any]:
    """Run a single sub-experiment and return summary dict."""
    run_dir = master_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.output.run_id = tag

    t0 = time.time()

    if do_evolutionary:
        result = run_evolutionary_search(prompts, risky, safe, cfg_copy, run_dir)
        summary = {
            "tag": tag,
            "copyright_leakage_score": result.copyright_leakage_score,
            "worst_kl_mean": result.worst_kl_mean,
            "total_violations": result.total_violations,
            "num_rounds": len(result.rounds),
            "num_prompts_evaluated": sum(rr.pool_size for rr in result.rounds),
            "elapsed_sec": time.time() - t0,
        }
    else:
        estimates = evaluate_pool(
            prompts, risky, safe,
            n_samples=cfg_copy.sampling.num_samples,
            delta=cfg_copy.statistical.delta,
            claimed_K=cfg_copy.statistical.claimed_K,
            temperature=cfg_copy.sampling.temperature,
            batch_size=cfg_copy.sampling.batch_size,
        )
        estimates.sort(key=lambda e: e.lower_bound, reverse=True)
        best = estimates[0] if estimates else None

        with open(run_dir / "results.jsonl", "w") as f:
            for e in estimates:
                d = _estimate_to_dict(e)
                d["round"] = 0
                f.write(json.dumps(d) + "\n")

        num_v = sum(1 for e in estimates if e.violates)
        summary = {
            "tag": tag,
            "copyright_leakage_score": best.lower_bound if best else 0.0,
            "worst_kl_mean": best.kl_mean if best else 0.0,
            "total_violations": num_v,
            "num_rounds": 0,
            "num_prompts_evaluated": len(estimates),
            "elapsed_sec": time.time() - t0,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    logger.info("  [%s] LB=%.4f  KL=%.4f  viol=%d  (%.1fs)",
                tag, summary["copyright_leakage_score"],
                summary["worst_kl_mean"], summary["total_violations"],
                summary["elapsed_sec"])
    return summary


def _free_models(*models):
    """Delete models and reclaim memory."""
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================
# Main experiment driver
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive NAF audit experiments for paper")
    parser.add_argument("--config", type=str, default="configs/comprehensive.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced matrix (~15 min instead of ~2 hrs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs/full_experiments")
    parser.add_argument("--skip_large_models", action="store_true",
                        help="Skip GPT-2 Large / Pythia-410M to save time/memory")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (default: <output_dir>/experiment_log.txt)")
    args = parser.parse_args()

    master_dir = Path(args.output_dir)
    master_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file, master_dir)

    cfg = load_config(args.config)
    cfg.seed = args.seed
    set_seed(cfg.seed)
    device = resolve_device(cfg.hardware)

    all_summaries: List[Dict] = []
    total_t0 = time.time()

    # ==================================================================
    # EXPERIMENT 1 — Model pairs across 3 families
    #   This is the most important experiment: shows the framework works
    #   on diverse architectures and that KL grows with model gap.
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 1: MODEL PAIR COMPARISON (3 families)")
    logger.info("=" * 70)

    for family_name, pairs in ALL_FAMILIES:
        for risky_name, safe_name, tag in pairs:
            if args.skip_large_models and ("large" in risky_name or "1b" in risky_name
                                           or "1.3b" in risky_name):
                logger.info("  Skipping %s (--skip_large_models)", tag)
                continue

            logger.info("  [%s] Loading %s vs %s …", family_name, risky_name, safe_name)
            try:
                risky = LanguageModel(risky_name, device, cfg.hardware)
                safe = LanguageModel(safe_name, device, cfg.hardware)
            except Exception as e:
                logger.warning("  SKIP %s: %s", tag, e)
                continue

            prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
            use_n = min(200, len(prompts))

            cfg_exp = copy.deepcopy(cfg)
            cfg_exp.sampling.num_samples = 1000
            cfg_exp.search.num_rounds = 5
            cfg_exp.search.top_k = 15

            # Evolutionary
            s = _run_single(f"mp_evo_{tag}", cfg_exp, risky, safe,
                            prompts[:use_n], master_dir)
            s.update(experiment="model_pair", family=family_name,
                     risky=risky_name, safe=safe_name, method="evolutionary")
            all_summaries.append(s)

            # Random baseline
            s = _run_single(f"mp_rand_{tag}", cfg_exp, risky, safe,
                            prompts[:use_n], master_dir, do_evolutionary=False)
            s.update(experiment="model_pair", family=family_name,
                     risky=risky_name, safe=safe_name, method="random")
            all_summaries.append(s)

            _free_models(risky, safe)

    # ==================================================================
    # Load default pair for remaining ablation experiments
    # ==================================================================
    logger.info("Loading default pair (%s vs %s) for ablations…",
                cfg.model.risky_model, cfg.model.safe_model)
    risky = LanguageModel(cfg.model.risky_model, device, cfg.hardware)
    safe = LanguageModel(cfg.model.safe_model, device, cfg.hardware)
    bookmia_prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)

    # ==================================================================
    # EXPERIMENT 2 — Dataset comparison: BookMIA vs WikiText-103
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 2: DATASET COMPARISON (BookMIA vs WikiText-103)")
    logger.info("=" * 70)

    for ds_name, ds_cfg_overrides in [
        ("bookmia", {"name": "bookmia", "num_prompts": 200, "member_only": True}),
        ("wikitext", {"name": "wikitext", "num_prompts": 200}),
    ]:
        cfg_exp = copy.deepcopy(cfg)
        for k, v in ds_cfg_overrides.items():
            setattr(cfg_exp.dataset, k, v)
        cfg_exp.sampling.num_samples = 1000
        cfg_exp.search.num_rounds = 5

        try:
            ds_prompts = load_prompts(cfg_exp.dataset, tokenizer=risky.tokenizer)
        except Exception as e:
            logger.warning("  SKIP dataset %s: %s", ds_name, e)
            continue

        # Evolutionary
        s = _run_single(f"ds_evo_{ds_name}", cfg_exp, risky, safe,
                        ds_prompts, master_dir)
        s.update(experiment="dataset", dataset=ds_name, method="evolutionary")
        all_summaries.append(s)

        # Random
        s = _run_single(f"ds_rand_{ds_name}", cfg_exp, risky, safe,
                        ds_prompts, master_dir, do_evolutionary=False)
        s.update(experiment="dataset", dataset=ds_name, method="random")
        all_summaries.append(s)

    # ==================================================================
    # EXPERIMENT 3 — Sample count ablation
    #   Shows: Bernstein bound tightens as n increases (theory → practice)
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 3: SAMPLE COUNT ABLATION")
    logger.info("=" * 70)

    counts = SAMPLE_COUNTS_QUICK if args.quick else SAMPLE_COUNTS
    for n in counts:
        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.sampling.num_samples = n
        cfg_exp.search.num_rounds = 3
        s = _run_single(f"abl_samples_{n}", cfg_exp, risky, safe,
                        bookmia_prompts[:100], master_dir)
        s.update(experiment="sample_ablation", num_samples=n)
        all_summaries.append(s)

    # ==================================================================
    # EXPERIMENT 4 — Round count ablation
    #   Shows: evolutionary search improves with more rounds
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 4: EVOLUTIONARY ROUND ABLATION")
    logger.info("=" * 70)

    rounds = ROUND_COUNTS_QUICK if args.quick else ROUND_COUNTS
    for r in rounds:
        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.search.num_rounds = r
        cfg_exp.sampling.num_samples = 1000
        s = _run_single(f"abl_rounds_{r}", cfg_exp, risky, safe,
                        bookmia_prompts[:100], master_dir)
        s.update(experiment="round_ablation", num_rounds_cfg=r)
        all_summaries.append(s)

    # ==================================================================
    # EXPERIMENT 5 — Mutation operator ablation
    #   Shows: which mutation types are most effective
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 5: MUTATION OPERATOR ABLATION")
    logger.info("=" * 70)

    for mname, mtypes in MUTATION_ABLATIONS.items():
        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.search.mutation_types = mtypes
        cfg_exp.search.num_rounds = 5
        cfg_exp.sampling.num_samples = 500
        s = _run_single(f"abl_mut_{mname}", cfg_exp, risky, safe,
                        bookmia_prompts[:100], master_dir)
        s.update(experiment="mutation_ablation", mutation_set=mname)
        all_summaries.append(s)

    # ==================================================================
    # EXPERIMENT 6 — K threshold sweep
    #   Shows: violation count drops cleanly as K increases
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 6: NAF THRESHOLD (K) SWEEP")
    logger.info("=" * 70)

    ks = K_THRESHOLDS_QUICK if args.quick else K_THRESHOLDS
    for K in ks:
        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.statistical.claimed_K = K
        cfg_exp.search.num_rounds = 5
        cfg_exp.sampling.num_samples = 1000
        s = _run_single(f"sweep_K_{K}", cfg_exp, risky, safe,
                        bookmia_prompts[:200], master_dir)
        s.update(experiment="K_threshold", claimed_K=K)
        all_summaries.append(s)

    # ==================================================================
    # EXPERIMENT 7 — Confidence parameter (δ) sweep
    #   Shows: trade-off between confidence and bound tightness
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 7: CONFIDENCE (δ) SWEEP")
    logger.info("=" * 70)

    for delta in DELTA_VALUES:
        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.statistical.delta = delta
        cfg_exp.search.num_rounds = 5
        cfg_exp.sampling.num_samples = 1000
        s = _run_single(f"sweep_delta_{delta}", cfg_exp, risky, safe,
                        bookmia_prompts[:100], master_dir)
        s.update(experiment="delta_sweep", delta_cfg=delta)
        all_summaries.append(s)

    # ==================================================================
    # EXPERIMENT 8 — Large-scale headline run
    #   Uses all available prompts, many samples, many rounds
    # ==================================================================
    logger.info("=" * 70)
    logger.info("EXP 8: LARGE-SCALE (%d prompts, 2000 samples, 10 rounds)",
                len(bookmia_prompts))
    logger.info("=" * 70)

    cfg_exp = copy.deepcopy(cfg)
    cfg_exp.sampling.num_samples = 2000
    cfg_exp.search.num_rounds = 10
    cfg_exp.search.top_k = 20
    cfg_exp.search.pool_size = 100

    s = _run_single("large_evo", cfg_exp, risky, safe,
                    bookmia_prompts, master_dir)
    s.update(experiment="large_scale", method="evolutionary",
             total_prompts=len(bookmia_prompts))
    all_summaries.append(s)

    s = _run_single("large_rand", cfg_exp, risky, safe,
                    bookmia_prompts, master_dir, do_evolutionary=False)
    s.update(experiment="large_scale", method="random",
             total_prompts=len(bookmia_prompts))
    all_summaries.append(s)

    _free_models(risky, safe)

    # ==================================================================
    # Save master results
    # ==================================================================
    total_elapsed = time.time() - total_t0
    master_path = master_dir / "all_experiments.json"
    with open(master_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Summary table
    logger.info("=" * 80)
    logger.info("FULL SUMMARY TABLE  (%d experiments in %.0f min)",
                len(all_summaries), total_elapsed / 60)
    logger.info("=" * 80)
    logger.info("%-45s  %8s  %8s  %6s  %7s",
                "Tag", "Max LB", "Max KL", "Viol.", "Time")
    logger.info("-" * 82)
    for s in all_summaries:
        logger.info("%-45s  %8.4f  %8.4f  %6d  %6.1fs",
                    s["tag"][:45],
                    s["copyright_leakage_score"],
                    s["worst_kl_mean"],
                    s["total_violations"],
                    s["elapsed_sec"])
    logger.info("=" * 80)
    logger.info("Total wall time: %.1f min", total_elapsed / 60)
    logger.info("Results saved to: %s", master_dir)


if __name__ == "__main__":
    main()
