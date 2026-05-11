"""
Baseline: evaluate KL on random (un-optimised) prompts — no evolutionary search.

This serves as the control against which the evolutionary search is compared.
The baseline simply evaluates the initial pool of prompts (no mutation rounds)
and reports the same statistics.

Usage:
    python -m src.run_baseline_random --config configs/mvp.yaml
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from .config import (
    build_cli_parser,
    cli_overrides,
    load_config,
    make_run_dir,
    set_seed,
)
from .dataset import load_prompts
from .evolutionary_search import evaluate_pool, _estimate_to_dict
from .models import load_model_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_cli_parser("NAF Audit — Random Prompt Baseline")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides(args))
    # Force run_id suffix so baseline doesn't overwrite audit runs
    if cfg.output.run_id is None:
        import datetime
        cfg.output.run_id = "baseline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    elif not cfg.output.run_id.startswith("baseline"):
        cfg.output.run_id = "baseline_" + cfg.output.run_id

    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)

    logger.info("=" * 60)
    logger.info("NAF AUDIT — Random Prompt Baseline")
    logger.info("Run dir: %s", run_dir)
    logger.info("=" * 60)

    # Load models
    risky, safe = load_model_pair(cfg.model.risky_model, cfg.model.safe_model, cfg.hardware)

    # Load prompts
    prompts = load_prompts(cfg.dataset, tokenizer=risky.tokenizer)
    logger.info("Loaded %d prompts", len(prompts))

    # Evaluate all prompts (single round, no mutations)
    t0 = time.time()
    estimates = evaluate_pool(
        prompts, risky, safe,
        n_samples=cfg.sampling.num_samples,
        delta=cfg.statistical.delta,
        claimed_K=cfg.statistical.claimed_K,
        temperature=cfg.sampling.temperature,
        batch_size=cfg.sampling.batch_size,
    )
    elapsed = time.time() - t0

    estimates.sort(key=lambda e: e.lower_bound, reverse=True)

    # JSONL output
    jsonl_path = run_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for est in estimates:
            d = _estimate_to_dict(est)
            d["round"] = 0
            f.write(json.dumps(d) + "\n")

    # CSV output
    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "prompt_text", "kl_mean", "sample_var",
            "lower_bound", "upper_bound", "bernstein_correction",
            "n_samples", "delta", "violates", "claimed_K",
        ])
        for est in estimates:
            writer.writerow([
                0, est.prompt_text[:200], est.kl_mean, est.sample_var,
                est.lower_bound, est.upper_bound, est.bernstein_correction,
                est.n_samples, est.delta, est.violates, est.claimed_K,
            ])

    # Summary
    num_violations = sum(1 for e in estimates if e.violates)
    best = estimates[0] if estimates else None
    summary = {
        "copyright_leakage_score": best.lower_bound if best else 0.0,
        "worst_prompt": best.prompt_text if best else "",
        "worst_kl_mean": best.kl_mean if best else 0.0,
        "total_violations": num_violations,
        "claimed_K": cfg.statistical.claimed_K,
        "delta": cfg.statistical.delta,
        "num_prompts": len(estimates),
        "elapsed_sec": elapsed,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("BASELINE COMPLETE")
    logger.info("Max KL lower bound: %.4f", summary["copyright_leakage_score"])
    logger.info("Max KL mean:        %.4f", summary["worst_kl_mean"])
    logger.info("Violations:         %d / %d", num_violations, len(estimates))
    logger.info("Elapsed:            %.1fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
