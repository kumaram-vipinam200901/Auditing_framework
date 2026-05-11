"""
Remaining-work experiments for the final report.

Implements the four items listed in §Discussion of the mid-report:
  (i)   Scaling to larger models (GPT-2 XL, Pythia-1B)
  (ii)  Auditing anchored decoding (He et al. 2026)
  (iii) Sequence-level KL estimation (vs token-level)
  (iv)  Curated memorised passages from Carlini et al. as seed prompts

Usage:
    python -m src.run_remaining_experiments --config configs/overnight.yaml
    python -m src.run_remaining_experiments --config configs/overnight.yaml --quick
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import platform
import psutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import AuditConfig, load_config, resolve_device, set_seed
from .dataset import load_prompts
from .evolutionary_search import evaluate_pool, run_evolutionary_search, _estimate_to_dict
from .kl_estimator import estimate_kl, estimate_kl_sequence, KLEstimate
from .models import AnchoredDecodingModel, LanguageModel

logger = logging.getLogger(__name__)


# ======================================================================
# Logging
# ======================================================================

def _setup_logging(output_dir: Path):
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler(str(output_dir / "remaining_log.txt"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _snapshot_resources() -> Dict[str, Any]:
    """Capture a snapshot of current resource usage."""
    snap: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_epoch": time.time(),
    }
    # CPU / RAM
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    snap["cpu_rss_mb"] = round(mem.rss / (1024 ** 2), 1)
    snap["cpu_vms_mb"] = round(mem.vms / (1024 ** 2), 1)
    snap["cpu_percent"] = proc.cpu_percent(interval=0.1)
    snap["system_ram_used_pct"] = psutil.virtual_memory().percent
    # GPU
    if torch.cuda.is_available():
        snap["gpu_name"] = torch.cuda.get_device_name(0)
        snap["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
        snap["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
        snap["gpu_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
        snap["gpu_utilization_pct"] = None  # nvidia-smi needed; logged at start
    else:
        snap["gpu_name"] = "N/A (CPU only)"
        snap["gpu_mem_allocated_mb"] = 0
        snap["gpu_mem_reserved_mb"] = 0
    return snap


def _log_resources(tag: str, snap: Dict[str, Any]):
    """Log a resource snapshot."""
    gpu_str = (
        f"GPU={snap['gpu_mem_allocated_mb']:.0f}/{snap.get('gpu_mem_total_mb', '?')}MB"
        if snap.get("gpu_mem_allocated_mb", 0) > 0
        else "GPU=N/A"
    )
    logger.info(
        "  [RESOURCES %s] RAM=%.0fMB  %s  CPU=%.1f%%  SysRAM=%.1f%%",
        tag, snap["cpu_rss_mb"], gpu_str,
        snap["cpu_percent"], snap["system_ram_used_pct"],
    )


def _free_models(*models):
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_single(
    tag: str,
    cfg: AuditConfig,
    risky,
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
    res_before = _snapshot_resources()

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

    # Resource tracking
    res_after = _snapshot_resources()
    summary["resources"] = {
        "before": res_before,
        "after": res_after,
        "peak_gpu_mem_mb": (
            round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 1)
            if torch.cuda.is_available() else 0
        ),
        "cpu_rss_delta_mb": round(
            res_after["cpu_rss_mb"] - res_before["cpu_rss_mb"], 1
        ),
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    logger.info("  [%s] LB=%.4f  KL=%.4f  viol=%d  (%.1fs)",
                tag, summary["copyright_leakage_score"],
                summary["worst_kl_mean"], summary["total_violations"],
                summary["elapsed_sec"])
    _log_resources(tag, res_after)
    return summary


# ======================================================================
# EXPERIMENT (i): Larger model pairs
# ======================================================================

LARGE_PAIRS: List[Tuple[str, str, str, str]] = [
    # (risky, safe, tag, family)
    ("gpt2-xl",                    "gpt2",                     "gpt2xl_vs_gpt2",       "GPT-2"),
    ("gpt2-xl",                    "gpt2-medium",              "gpt2xl_vs_gpt2med",    "GPT-2"),
    ("EleutherAI/pythia-1b",       "EleutherAI/pythia-70m",    "pythia1b_vs_70m",      "Pythia"),
    ("EleutherAI/pythia-1b",       "EleutherAI/pythia-160m",   "pythia1b_vs_160m",     "Pythia"),
]

LARGE_PAIRS_QUICK: List[Tuple[str, str, str, str]] = [
    ("gpt2-xl",  "gpt2",  "gpt2xl_vs_gpt2", "GPT-2"),
]


def run_scaling_experiments(
    cfg: AuditConfig,
    device: torch.device,
    master_dir: Path,
    quick: bool = False,
) -> List[Dict]:
    logger.info("=" * 70)
    logger.info("REMAINING (i): SCALING TO LARGER MODELS")
    logger.info("=" * 70)

    pairs = LARGE_PAIRS_QUICK if quick else LARGE_PAIRS
    summaries = []

    for risky_name, safe_name, tag, family in pairs:
        logger.info("  Loading %s vs %s …", risky_name, safe_name)
        try:
            risky = LanguageModel(risky_name, device, cfg.hardware)
            safe = LanguageModel(safe_name, device, cfg.hardware)
        except Exception as e:
            logger.warning("  SKIP %s: %s", tag, e)
            continue

        prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
        use_n = min(100 if quick else 200, len(prompts))

        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.sampling.num_samples = 500 if quick else 1000
        cfg_exp.search.num_rounds = 3 if quick else 5
        cfg_exp.search.top_k = 10

        # Evolutionary
        s = _run_single(f"scale_evo_{tag}", cfg_exp, risky, safe,
                        prompts[:use_n], master_dir)
        s.update(experiment="scaling", family=family,
                 risky=risky_name, safe=safe_name, method="evolutionary")
        summaries.append(s)

        # Random baseline
        s = _run_single(f"scale_rand_{tag}", cfg_exp, risky, safe,
                        prompts[:use_n], master_dir, do_evolutionary=False)
        s.update(experiment="scaling", family=family,
                 risky=risky_name, safe=safe_name, method="random")
        summaries.append(s)

        _free_models(risky, safe)

    return summaries


# ======================================================================
# EXPERIMENT (ii): Anchored decoding audit
# ======================================================================

ANCHORED_K_VALUES = [0.5, 1.0, 2.0, 5.0]
ANCHORED_K_QUICK  = [1.0, 5.0]


def run_anchored_decoding_experiments(
    cfg: AuditConfig,
    device: torch.device,
    master_dir: Path,
    quick: bool = False,
) -> List[Dict]:
    logger.info("=" * 70)
    logger.info("REMAINING (ii): AUDITING ANCHORED DECODING")
    logger.info("=" * 70)

    summaries = []
    k_values = ANCHORED_K_QUICK if quick else ANCHORED_K_VALUES

    risky_name = cfg.model.risky_model
    safe_name = cfg.model.safe_model

    logger.info("  Loading base models: %s, %s", risky_name, safe_name)
    risky = LanguageModel(risky_name, device, cfg.hardware)
    safe = LanguageModel(safe_name, device, cfg.hardware)

    prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
    use_n = min(50 if quick else 100, len(prompts))

    for K_anchor in k_values:
        logger.info("  Anchored decoding with K_anchor=%.2f", K_anchor)

        # Build the anchored model (wraps risky + safe)
        anchored = AnchoredDecodingModel(risky, safe, K=K_anchor)

        cfg_exp = copy.deepcopy(cfg)
        cfg_exp.sampling.num_samples = 500 if quick else 1000
        cfg_exp.search.num_rounds = 3 if quick else 5
        cfg_exp.search.top_k = 10

        # Audit: anchored model (as "risky") vs safe model
        # If anchored decoding is truly K-NAF, the auditor should NOT
        # find violations with claimed_K = K_anchor
        cfg_exp.statistical.claimed_K = K_anchor

        tag = f"anchored_K{K_anchor}"

        # Evolutionary search
        s = _run_single(f"anch_evo_{tag}", cfg_exp, anchored, safe,
                        prompts[:use_n], master_dir)
        s.update(experiment="anchored_decoding", K_anchor=K_anchor,
                 method="evolutionary")
        summaries.append(s)

        # Random baseline
        s = _run_single(f"anch_rand_{tag}", cfg_exp, anchored, safe,
                        prompts[:use_n], master_dir, do_evolutionary=False)
        s.update(experiment="anchored_decoding", K_anchor=K_anchor,
                 method="random")
        summaries.append(s)

    _free_models(risky, safe)
    return summaries


# ======================================================================
# EXPERIMENT (iii): Sequence-level KL vs token-level
# ======================================================================

SEQ_LENGTHS = [4, 8, 16, 32]
SEQ_LENGTHS_QUICK = [8, 16]


def run_sequence_kl_experiments(
    cfg: AuditConfig,
    device: torch.device,
    master_dir: Path,
    quick: bool = False,
) -> List[Dict]:
    logger.info("=" * 70)
    logger.info("REMAINING (iii): SEQUENCE-LEVEL KL ESTIMATION")
    logger.info("=" * 70)

    summaries = []
    seq_lens = SEQ_LENGTHS_QUICK if quick else SEQ_LENGTHS

    risky_name = cfg.model.risky_model
    safe_name = cfg.model.safe_model

    risky = LanguageModel(risky_name, device, cfg.hardware)
    safe = LanguageModel(safe_name, device, cfg.hardware)

    prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
    use_n = min(20 if quick else 50, len(prompts))
    n_seq = 50 if quick else 100

    # First: token-level baseline for comparison
    logger.info("  Token-level KL baseline (%d prompts)", use_n)
    token_estimates = evaluate_pool(
        prompts[:use_n], risky, safe,
        n_samples=cfg.sampling.num_samples,
        delta=cfg.statistical.delta,
        claimed_K=cfg.statistical.claimed_K,
        temperature=cfg.sampling.temperature,
        batch_size=cfg.sampling.batch_size,
    )
    token_estimates.sort(key=lambda e: e.lower_bound, reverse=True)

    token_best = token_estimates[0] if token_estimates else None
    token_violations = sum(1 for e in token_estimates if e.violates)
    token_kl_values = [e.kl_mean for e in token_estimates]
    avg_token_kl = sum(token_kl_values) / len(token_kl_values) if token_kl_values else 0.0

    summaries.append({
        "tag": "seqkl_token_baseline",
        "experiment": "sequence_kl",
        "method": "token_level",
        "seq_len": 1,
        "avg_kl_mean": avg_token_kl,
        "max_lower_bound": token_best.lower_bound if token_best else 0.0,
        "max_kl_mean": token_best.kl_mean if token_best else 0.0,
        "total_violations": token_violations,
        "num_prompts": use_n,
    })

    # Now: sequence-level for each length
    for T in seq_lens:
        logger.info("  Sequence-level KL (T=%d, %d seqs × %d prompts)", T, n_seq, use_n)

        run_dir = master_dir / f"seqkl_T{T}"
        run_dir.mkdir(parents=True, exist_ok=True)

        seq_estimates: List[KLEstimate] = []
        for i, prompt in enumerate(prompts[:use_n]):
            logger.info("    Prompt %d/%d (T=%d)", i + 1, use_n, T)
            try:
                est = estimate_kl_sequence(
                    prompt_text=prompt,
                    risky=risky,
                    safe=safe,
                    n_sequences=n_seq,
                    seq_len=T,
                    delta=cfg.statistical.delta,
                    claimed_K=cfg.statistical.claimed_K,
                    temperature=cfg.sampling.temperature,
                )
                seq_estimates.append(est)
            except Exception as e:
                logger.warning("    Failed prompt %d: %s", i, e)

        # Save results
        with open(run_dir / "results.jsonl", "w") as f:
            for e in seq_estimates:
                f.write(json.dumps(_estimate_to_dict(e)) + "\n")

        seq_estimates.sort(key=lambda e: e.lower_bound, reverse=True)
        best = seq_estimates[0] if seq_estimates else None
        n_viol = sum(1 for e in seq_estimates if e.violates)
        kl_vals = [e.kl_mean for e in seq_estimates]
        avg_kl = sum(kl_vals) / len(kl_vals) if kl_vals else 0.0

        s = {
            "tag": f"seqkl_T{T}",
            "experiment": "sequence_kl",
            "method": "sequence_level",
            "seq_len": T,
            "avg_kl_mean": avg_kl,
            "max_lower_bound": best.lower_bound if best else 0.0,
            "max_kl_mean": best.kl_mean if best else 0.0,
            "total_violations": n_viol,
            "num_prompts": use_n,
        }
        summaries.append(s)
        logger.info("  T=%d: avg_KL=%.4f  max_LB=%.4f  violations=%d",
                     T, avg_kl, s["max_lower_bound"], n_viol)

    _free_models(risky, safe)
    return summaries


# ======================================================================
# EXPERIMENT (iv): Curated memorised prompts
# ======================================================================

def run_curated_prompt_experiments(
    cfg: AuditConfig,
    device: torch.device,
    master_dir: Path,
    quick: bool = False,
) -> List[Dict]:
    logger.info("=" * 70)
    logger.info("REMAINING (iv): CURATED MEMORISED PROMPTS (Carlini-style)")
    logger.info("=" * 70)

    summaries = []

    # Load curated prompts from local file
    curated_path = Path(__file__).parent.parent / "data" / "carlini_memorised_prompts.txt"
    if not curated_path.exists():
        logger.warning("Curated prompts file not found at %s. Skipping.", curated_path)
        return summaries

    with open(curated_path, "r", encoding="utf-8") as f:
        curated_prompts = [line.strip() for line in f if line.strip()]

    logger.info("  Loaded %d curated prompts", len(curated_prompts))

    risky_name = cfg.model.risky_model
    safe_name = cfg.model.safe_model

    risky = LanguageModel(risky_name, device, cfg.hardware)
    safe = LanguageModel(safe_name, device, cfg.hardware)

    cfg_exp = copy.deepcopy(cfg)
    cfg_exp.sampling.num_samples = 500 if quick else 1000
    cfg_exp.search.num_rounds = 3 if quick else 5
    cfg_exp.search.top_k = 10

    # Evolutionary with curated prompts
    s = _run_single("curated_evo", cfg_exp, risky, safe,
                    curated_prompts, master_dir)
    s.update(experiment="curated_prompts", prompt_source="carlini",
             method="evolutionary", num_seed_prompts=len(curated_prompts))
    summaries.append(s)

    # Random baseline with curated prompts
    s = _run_single("curated_rand", cfg_exp, risky, safe,
                    curated_prompts, master_dir, do_evolutionary=False)
    s.update(experiment="curated_prompts", prompt_source="carlini",
             method="random", num_seed_prompts=len(curated_prompts))
    summaries.append(s)

    # Also compare: curated vs BookMIA on same model pair
    logger.info("  Comparing curated vs BookMIA prompts…")
    bookmia_prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
    use_n = min(len(curated_prompts), len(bookmia_prompts))

    s = _run_single("curated_vs_bookmia_curated", cfg_exp, risky, safe,
                    curated_prompts[:use_n], master_dir, do_evolutionary=False)
    s.update(experiment="curated_vs_bookmia", prompt_source="carlini",
             method="random", num_prompts=use_n)
    summaries.append(s)

    s = _run_single("curated_vs_bookmia_bookmia", cfg_exp, risky, safe,
                    bookmia_prompts[:use_n], master_dir, do_evolutionary=False)
    s.update(experiment="curated_vs_bookmia", prompt_source="bookmia",
             method="random", num_prompts=use_n)
    summaries.append(s)

    _free_models(risky, safe)
    return summaries


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run remaining-work experiments for final report")
    parser.add_argument("--config", type=str, default="configs/overnight.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced scope (~30 min instead of overnight)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs/remaining_work")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Skip experiments: scaling, anchored, sequence, curated")
    args = parser.parse_args()

    master_dir = Path(args.output_dir)
    master_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(master_dir)

    cfg = load_config(args.config)
    cfg.seed = args.seed
    set_seed(cfg.seed)
    device = resolve_device(cfg.hardware)

    all_summaries: List[Dict] = []
    resource_log: List[Dict] = []
    total_t0 = time.time()

    # Log system info once
    sys_info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_mem_total_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            if torch.cuda.is_available() else 0
        ),
        "cpu_count": os.cpu_count(),
        "total_ram_mb": round(psutil.virtual_memory().total / (1024 ** 2), 1),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.info("System: %s", json.dumps(sys_info, indent=2))
    resource_log.append({"event": "system_info", **sys_info})

    # (i) Scaling experiments
    if "scaling" not in args.skip:
        resource_log.append({"event": "start_scaling", **_snapshot_resources()})
        sums = run_scaling_experiments(cfg, device, master_dir, quick=args.quick)
        all_summaries.extend(sums)
        resource_log.append({"event": "end_scaling", **_snapshot_resources()})

    # (ii) Anchored decoding
    if "anchored" not in args.skip:
        resource_log.append({"event": "start_anchored", **_snapshot_resources()})
        sums = run_anchored_decoding_experiments(cfg, device, master_dir, quick=args.quick)
        all_summaries.extend(sums)
        resource_log.append({"event": "end_anchored", **_snapshot_resources()})

    # (iii) Sequence-level KL
    if "sequence" not in args.skip:
        resource_log.append({"event": "start_sequence", **_snapshot_resources()})
        sums = run_sequence_kl_experiments(cfg, device, master_dir, quick=args.quick)
        all_summaries.extend(sums)
        resource_log.append({"event": "end_sequence", **_snapshot_resources()})

    # (iv) Curated memorised prompts
    if "curated" not in args.skip:
        resource_log.append({"event": "start_curated", **_snapshot_resources()})
        sums = run_curated_prompt_experiments(cfg, device, master_dir, quick=args.quick)
        all_summaries.extend(sums)
        resource_log.append({"event": "end_curated", **_snapshot_resources()})

    # Save master results
    total_elapsed = time.time() - total_t0
    master_path = master_dir / "remaining_experiments.json"
    with open(master_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Save resource log
    resource_log.append({"event": "finished", "total_elapsed_min": round(total_elapsed / 60, 2), **_snapshot_resources()})
    with open(master_dir / "resource_log.json", "w") as f:
        json.dump(resource_log, f, indent=2)

    # Summary table
    logger.info("=" * 80)
    logger.info("REMAINING WORK SUMMARY  (%d experiments in %.0f min)",
                len(all_summaries), total_elapsed / 60)
    logger.info("=" * 80)
    logger.info("%-40s  %8s  %8s  %6s  %7s  %10s  %10s",
                "Tag", "Max LB", "Max KL", "Viol.", "Time", "PeakGPU MB", "RAM MB")
    logger.info("-" * 100)
    for s in all_summaries:
        lb = s.get("copyright_leakage_score", s.get("max_lower_bound", 0.0))
        kl = s.get("worst_kl_mean", s.get("max_kl_mean", 0.0))
        res = s.get("resources", {})
        peak_gpu = res.get("peak_gpu_mem_mb", 0)
        ram = res.get("after", {}).get("cpu_rss_mb", 0)
        logger.info("%-40s  %8.4f  %8.4f  %6d  %6.1fs  %10.0f  %10.0f",
                    s["tag"][:40], lb, kl,
                    s["total_violations"],
                    s.get("elapsed_sec", 0.0),
                    peak_gpu, ram)
    logger.info("=" * 80)
    logger.info("Total wall time: %.1f min", total_elapsed / 60)
    logger.info("Results saved to: %s", master_dir)


if __name__ == "__main__":
    main()
