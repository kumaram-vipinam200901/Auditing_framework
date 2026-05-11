"""
Evolutionary adversarial prompt search for NAF auditing.

Algorithm (from the proposal, §1 — "Part 2: Finding the worst-case prompts"):
  1. Initialise a pool of candidate prompts (e.g. book openings from BookMIA).
  2. Evaluate each prompt's KL lower bound.
  3. Keep the top-k highest-scoring prompts.
  4. Generate new candidates by mutating the top-k parents.
  5. Merge survivors + children → new pool.  Repeat for R rounds.

The "Copyright Leakage Score" is the maximum lower bound found across all
prompts over all rounds, along with the worst-case prompt.

The elimination / selection step is analogous to LCB (lower confidence bound)
in bandits, as noted in the course PDF (Research_DP.pdf §1.1.2).
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import AuditConfig, SearchConfig, StatisticalConfig
from .kl_estimator import KLEstimate, estimate_kl
from .models import LanguageModel
from .prompt_mutations import generate_mutations

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Summary of one evolutionary round."""
    round_idx: int
    best_prompt: str
    best_kl_mean: float
    best_lower_bound: float
    best_violates: bool
    pool_size: int
    num_violations: int
    elapsed_sec: float
    all_estimates: List[Dict] = field(default_factory=list)


@dataclass
class SearchResult:
    """Aggregate result over all rounds."""
    rounds: List[RoundResult]
    copyright_leakage_score: float  # max LB across all prompts and rounds
    worst_prompt: str
    worst_kl_mean: float
    total_violations: int
    claimed_K: float
    delta: float


def _estimate_to_dict(est: KLEstimate) -> Dict:
    """Serialise a KLEstimate to a JSON-friendly dict (drop tensor)."""
    d = {
        "prompt_text": est.prompt_text,
        "kl_mean": est.kl_mean,
        "sample_var": est.sample_var,
        "lower_bound": est.lower_bound,
        "upper_bound": est.upper_bound,
        "bernstein_correction": est.bernstein_correction,
        "n_samples": est.n_samples,
        "delta": est.delta,
        "violates": est.violates,
        "claimed_K": est.claimed_K,
    }
    return d


def evaluate_pool(
    prompts: List[str],
    risky: LanguageModel,
    safe: LanguageModel,
    n_samples: int,
    delta: float,
    claimed_K: float,
    temperature: float,
    batch_size: int,
) -> List[KLEstimate]:
    """Evaluate KL estimates for every prompt in the pool."""
    estimates: List[KLEstimate] = []
    for i, prompt in enumerate(prompts):
        logger.info(
            "  Evaluating prompt %d/%d (len=%d chars)", i + 1, len(prompts), len(prompt)
        )
        try:
            est = estimate_kl(
                prompt_text=prompt,
                risky=risky,
                safe=safe,
                n_samples=n_samples,
                delta=delta,
                claimed_K=claimed_K,
                temperature=temperature,
                batch_size=batch_size,
            )
            estimates.append(est)
        except Exception as e:
            logger.warning("  Failed on prompt %d: %s", i, e)
    return estimates


def run_evolutionary_search(
    initial_prompts: List[str],
    risky: LanguageModel,
    safe: LanguageModel,
    cfg: AuditConfig,
    run_dir: Optional[Path] = None,
) -> SearchResult:
    """Execute the full evolutionary prompt search.

    Parameters
    ----------
    initial_prompts : seed prompts (e.g. from BookMIA)
    risky, safe : loaded language models
    cfg : full experiment config
    run_dir : directory for per-round outputs (optional)

    Returns
    -------
    SearchResult with all rounds, best prompt, and copyright leakage score.
    """
    search = cfg.search
    stat = cfg.statistical
    samp = cfg.sampling

    rng = random.Random(cfg.seed)
    pool = list(initial_prompts)

    # Track global bests
    global_best_lb = float("-inf")
    global_best_prompt = ""
    global_best_kl = 0.0
    total_violations = 0
    all_rounds: List[RoundResult] = []

    # JSONL writer
    jsonl_path = run_dir / "results.jsonl" if run_dir else None
    jsonl_file = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

    for r in range(search.num_rounds):
        t0 = time.time()
        logger.info("=== Round %d/%d  (pool size=%d) ===", r + 1, search.num_rounds, len(pool))

        # --- Evaluate current pool ---
        estimates = evaluate_pool(
            pool, risky, safe,
            n_samples=samp.num_samples,
            delta=stat.delta,
            claimed_K=stat.claimed_K,
            temperature=samp.temperature,
            batch_size=samp.batch_size,
        )

        if not estimates:
            logger.error("No valid estimates in round %d. Stopping.", r)
            break

        # Sort by lower_bound descending
        estimates.sort(key=lambda e: e.lower_bound, reverse=True)

        round_best = estimates[0]
        round_violations = sum(1 for e in estimates if e.violates)
        total_violations += round_violations

        # Update global best
        if round_best.lower_bound > global_best_lb:
            global_best_lb = round_best.lower_bound
            global_best_prompt = round_best.prompt_text
            global_best_kl = round_best.kl_mean

        # Write JSONL
        est_dicts = [_estimate_to_dict(e) for e in estimates]
        if jsonl_file:
            for d in est_dicts:
                d["round"] = r
                jsonl_file.write(json.dumps(d) + "\n")
            jsonl_file.flush()

        elapsed = time.time() - t0
        rr = RoundResult(
            round_idx=r,
            best_prompt=round_best.prompt_text,
            best_kl_mean=round_best.kl_mean,
            best_lower_bound=round_best.lower_bound,
            best_violates=round_best.violates,
            pool_size=len(pool),
            num_violations=round_violations,
            elapsed_sec=elapsed,
            all_estimates=est_dicts,
        )
        all_rounds.append(rr)

        logger.info(
            "  Round %d: best LB=%.4f  mean_KL=%.4f  violations=%d/%d  (%.1fs)",
            r, round_best.lower_bound, round_best.kl_mean,
            round_violations, len(estimates), elapsed,
        )

        # Save best prompts this round
        if run_dir and cfg.output.save_prompts:
            _save_round_prompts(run_dir, r, estimates[:search.top_k])

        # --- Selection: keep top-k ---
        survivors_count = search.top_k
        survivors = [e.prompt_text for e in estimates[:survivors_count]]

        # --- Mutation: generate children ---
        if r < search.num_rounds - 1:  # no mutation after last round
            children = generate_mutations(
                parents=survivors,
                mutation_types=search.mutation_types,
                mutations_per_parent=search.mutations_per_parent,
                rng=rng,
                use_paraphrase=search.use_local_paraphrase,
                paraphrase_model=search.paraphrase_model,
            )
            # Merge: survivors + children → new pool (cap at pool_size)
            pool = survivors + children
            if len(pool) > search.pool_size:
                pool = pool[:search.pool_size]

            logger.info("  New pool: %d survivors + %d children = %d",
                        len(survivors), len(children), len(pool))
        else:
            pool = survivors

    if jsonl_file:
        jsonl_file.close()

    result = SearchResult(
        rounds=all_rounds,
        copyright_leakage_score=global_best_lb,
        worst_prompt=global_best_prompt,
        worst_kl_mean=global_best_kl,
        total_violations=total_violations,
        claimed_K=stat.claimed_K,
        delta=stat.delta,
    )

    # Save summary
    if run_dir and cfg.output.save_summary:
        _save_summary(run_dir, result)

    return result


def _save_round_prompts(run_dir: Path, round_idx: int, top_estimates: List[KLEstimate]):
    """Save the top-k prompts for a round."""
    out = run_dir / f"round_{round_idx:02d}_top_prompts.json"
    data = [
        {"rank": i, "prompt": e.prompt_text, "kl_mean": e.kl_mean,
         "lower_bound": e.lower_bound, "violates": e.violates}
        for i, e in enumerate(top_estimates)
    ]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_summary(run_dir: Path, result: SearchResult):
    """Write a JSON summary of the full search."""
    summary = {
        "copyright_leakage_score": result.copyright_leakage_score,
        "worst_prompt": result.worst_prompt,
        "worst_kl_mean": result.worst_kl_mean,
        "total_violations": result.total_violations,
        "claimed_K": result.claimed_K,
        "delta": result.delta,
        "num_rounds": len(result.rounds),
        "per_round": [
            {
                "round": rr.round_idx,
                "best_lower_bound": rr.best_lower_bound,
                "best_kl_mean": rr.best_kl_mean,
                "num_violations": rr.num_violations,
                "pool_size": rr.pool_size,
                "elapsed_sec": rr.elapsed_sec,
            }
            for rr in result.rounds
        ],
    }
    out = run_dir / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", out)
