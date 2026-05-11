"""
Smoke test: runs a minimal end-to-end audit to verify correctness.

Uses tiny sample counts and few prompts to complete in < 2 minutes on CPU.
Validates:
  1. Models load and produce valid log-probs
  2. KL estimator returns a valid KLEstimate
  3. Empirical Bernstein bound computes correctly
  4. Mutation operators work
  5. Evolutionary search loop completes
  6. Output files are created

Usage:
    python -m pytest tests/test_smoke.py -v
    python tests/test_smoke.py          # standalone
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AuditConfig, HardwareConfig, load_config, set_seed
from src.kl_estimator import empirical_bernstein_bound, estimate_kl
from src.models import LanguageModel
from src.prompt_mutations import (
    apply_mutation,
    generate_mutations,
    mutate_crossover,
    mutate_punctuation_tweak,
    mutate_style_tweak,
    mutate_verbatim_instruction,
)


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

_DEVICE = torch.device("cpu")
_HW = HardwareConfig(device="cpu", mixed_precision="none", torch_compile=False)


def _get_models():
    """Load gpt2 (smallest) as both risky and safe for speed."""
    risky = LanguageModel("gpt2", _DEVICE, _HW)
    safe = LanguageModel("gpt2", _DEVICE, _HW)
    return risky, safe


# ------------------------------------------------------------------
# Test: Empirical Bernstein bound
# ------------------------------------------------------------------

def test_bernstein_bound():
    """Verify the Bernstein bound formula on synthetic data."""
    torch.manual_seed(0)
    # Known distribution: Z ~ N(2.0, 1.0)
    z = torch.randn(200) + 2.0
    kl_mean, var, lb, ub, corr = empirical_bernstein_bound(z, delta=0.05)

    assert abs(kl_mean - 2.0) < 0.5, f"Mean {kl_mean} too far from 2.0"
    assert var > 0, "Variance must be positive"
    assert lb < kl_mean, "Lower bound must be below mean"
    assert ub > kl_mean, "Upper bound must be above mean"
    assert corr > 0, "Correction must be positive"

    # Check formula:  lb = mean - sqrt(2*var*ln(2/d)/n) - 7*ln(2/d)/(3*(n-1))
    n = 200
    delta = 0.05
    ln_term = math.log(2.0 / delta)
    expected_corr = math.sqrt(2 * var * ln_term / n) + 7 * ln_term / (3 * (n - 1))
    assert abs(corr - expected_corr) < 1e-10, f"Correction mismatch: {corr} vs {expected_corr}"

    print("✓ test_bernstein_bound passed")


# ------------------------------------------------------------------
# Test: Mutation operators
# ------------------------------------------------------------------

def test_mutations():
    rng = random.Random(42)
    prompt = "It was the best of times, it was the worst of times"

    # Single mutations
    v = mutate_verbatim_instruction(prompt, rng)
    assert prompt in v or prompt.lower() in v.lower()

    s = mutate_style_tweak(prompt, rng)
    assert len(s) > 0

    p = mutate_punctuation_tweak(prompt, rng)
    assert len(p) > 0

    # Crossover
    prompt_b = "Call me Ishmael. Some years ago"
    c = mutate_crossover(prompt, prompt_b, rng)
    assert len(c) > 0

    # Batch generation
    parents = [prompt, prompt_b]
    children = generate_mutations(
        parents,
        mutation_types=["verbatim_instruction", "punctuation_tweak", "crossover"],
        mutations_per_parent=2,
        rng=rng,
    )
    assert len(children) == 4  # 2 parents × 2 mutations

    print("✓ test_mutations passed")


# ------------------------------------------------------------------
# Test: Model log-probs
# ------------------------------------------------------------------

def test_model_logprobs():
    """Verify that log-probs sum to ~1 (in prob space)."""
    risky, _ = _get_models()
    ids = risky.encode("The quick brown fox")
    lp = risky.next_token_log_probs(ids)  # (1, V)
    assert lp.shape[-1] == risky.vocab_size
    # log-softmax should sum to 1 in prob space
    total = lp.exp().sum().item()
    assert abs(total - 1.0) < 1e-4, f"Probs sum to {total}, expected ~1.0"

    print("✓ test_model_logprobs passed")


# ------------------------------------------------------------------
# Test: KL estimator end-to-end
# ------------------------------------------------------------------

def test_kl_estimator():
    """Estimate KL for a single prompt and validate output structure."""
    risky, safe = _get_models()
    est = estimate_kl(
        prompt_text="Once upon a time in a land far away",
        risky=risky,
        safe=safe,
        n_samples=20,
        delta=0.05,
        claimed_K=1.0,
        temperature=1.0,
        batch_size=32,
    )
    assert est.n_samples == 20
    assert est.delta == 0.05
    assert isinstance(est.kl_mean, float)
    assert isinstance(est.lower_bound, float)
    assert est.lower_bound <= est.kl_mean
    # Since risky == safe here, KL should be ~0
    assert abs(est.kl_mean) < 0.5, f"Expected KL≈0 for same model, got {est.kl_mean}"

    print("✓ test_kl_estimator passed")


# ------------------------------------------------------------------
# Test: Mini evolutionary run
# ------------------------------------------------------------------

def test_mini_evolutionary():
    """Run 1 round with 3 prompts and 10 samples — just check it completes."""
    from src.evolutionary_search import run_evolutionary_search

    risky, safe = _get_models()
    cfg = AuditConfig(seed=42)
    cfg.sampling.num_samples = 10
    cfg.search.num_rounds = 1
    cfg.search.top_k = 2
    cfg.search.pool_size = 5
    cfg.search.mutations_per_parent = 1
    cfg.statistical.claimed_K = 1.0

    prompts = [
        "It was the best of times",
        "Call me Ishmael",
        "In a hole in the ground there lived a hobbit",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.output.run_dir = tmpdir
        cfg.output.run_id = "smoke"
        run_dir = Path(tmpdir) / "smoke"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_evolutionary_search(prompts, risky, safe, cfg, run_dir)

        assert result.copyright_leakage_score is not None
        assert len(result.rounds) == 1
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "summary.json").exists()

        # Validate JSON
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        assert "copyright_leakage_score" in summary

    print("✓ test_mini_evolutionary passed")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    test_bernstein_bound()
    test_mutations()
    test_model_logprobs()
    test_kl_estimator()
    test_mini_evolutionary()
    print("\n=== All smoke tests passed ===")
