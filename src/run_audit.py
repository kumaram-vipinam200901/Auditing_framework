"""
Main entrypoint: run the full NAF audit with evolutionary prompt search.

Usage:
    python -m src.run_audit --config configs/mvp.yaml
    python -m src.run_audit --config configs/fast.yaml --num_samples 100
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

from .config import (
    AuditConfig,
    build_cli_parser,
    cli_overrides,
    load_config,
    make_run_dir,
    set_seed,
)
from .dataset import load_prompts
from .evolutionary_search import run_evolutionary_search
from .models import load_model_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_cli_parser("NAF Audit — Evolutionary Prompt Search")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides(args))
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)

    logger.info("=" * 60)
    logger.info("NAF AUDIT — Evolutionary Search")
    logger.info("Run dir: %s", run_dir)
    logger.info("Risky model: %s", cfg.model.risky_model)
    logger.info("Safe model:  %s", cfg.model.safe_model)
    logger.info("Claimed K:   %s", cfg.statistical.claimed_K)
    logger.info("Delta:       %s", cfg.statistical.delta)
    logger.info("Samples:     %s", cfg.sampling.num_samples)
    logger.info("Rounds:      %s", cfg.search.num_rounds)
    logger.info("=" * 60)

    # Save config
    import yaml
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(json.loads(json.dumps({
            "seed": cfg.seed,
            "model": {"risky_model": cfg.model.risky_model, "safe_model": cfg.model.safe_model},
            "dataset": {"name": cfg.dataset.name, "num_prompts": cfg.dataset.num_prompts},
            "sampling": {"num_samples": cfg.sampling.num_samples, "batch_size": cfg.sampling.batch_size},
            "search": {"num_rounds": cfg.search.num_rounds, "top_k": cfg.search.top_k},
            "statistical": {"delta": cfg.statistical.delta, "claimed_K": cfg.statistical.claimed_K},
            "hardware": {"device": cfg.hardware.device, "mixed_precision": cfg.hardware.mixed_precision},
        })), f, default_flow_style=False)

    # Load models
    t0 = time.time()
    risky, safe = load_model_pair(cfg.model.risky_model, cfg.model.safe_model, cfg.hardware)
    logger.info("Models loaded in %.1fs", time.time() - t0)

    # Load prompts
    prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
    logger.info("Loaded %d initial prompts", len(prompts))

    # Run evolutionary search
    result = run_evolutionary_search(prompts, risky, safe, cfg, run_dir)

    # Write CSV summary
    if cfg.output.save_csv:
        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "prompt_text", "kl_mean", "sample_var",
                "lower_bound", "upper_bound", "bernstein_correction",
                "n_samples", "delta", "violates", "claimed_K",
            ])
            for rr in result.rounds:
                for est in rr.all_estimates:
                    writer.writerow([
                        est.get("round", rr.round_idx),
                        est["prompt_text"][:200],  # truncate for CSV
                        est["kl_mean"],
                        est["sample_var"],
                        est["lower_bound"],
                        est["upper_bound"],
                        est["bernstein_correction"],
                        est["n_samples"],
                        est["delta"],
                        est["violates"],
                        est["claimed_K"],
                    ])
        logger.info("CSV results saved to %s", csv_path)

    # Final report
    logger.info("=" * 60)
    logger.info("AUDIT COMPLETE")
    logger.info("Copyright Leakage Score (max LB): %.4f", result.copyright_leakage_score)
    logger.info("Worst-case KL mean:               %.4f", result.worst_kl_mean)
    logger.info("Claimed K:                        %.4f", result.claimed_K)
    logger.info("Total violations found:           %d", result.total_violations)
    if result.copyright_leakage_score > result.claimed_K:
        logger.info("*** VIOLATION DETECTED with confidence 1 - δ = %.3f ***",
                     1.0 - result.delta)
    else:
        logger.info("No violations detected at confidence level 1 - δ = %.3f",
                     1.0 - result.delta)
    logger.info("Worst-case prompt:\n%s", result.worst_prompt[:500])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
