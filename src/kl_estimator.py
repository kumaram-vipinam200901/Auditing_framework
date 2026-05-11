"""
Monte-Carlo KL divergence estimator with Empirical Bernstein lower bound.

Theory (from the proposal, citing Maurer & Pontil [2009], Theorem 4)
=====================================================================

For a given prefix x, define:
    p(·) = P_risky(·|x)      (risky / target model)
    q(·) = P_safe(·|x)       (safe / baseline model)

The KL divergence is:
    KL(p ∥ q) = E_{y ~ p} [ log p(y) - log q(y) ]

Define the *privacy loss* random variable:
    Z = log p(Y) - log q(Y),    Y ~ p(·)

Then KL(p ∥ q) = E[Z].

Estimation procedure
--------------------
1. Draw i.i.d. samples  y_1, …, y_n ~ p(·).
2. Compute  Z_i = log p(y_i) - log q(y_i)  for each sample.
3. The sample mean  K̂_n = (1/n) Σ Z_i  is an unbiased estimator of KL.

High-confidence LOWER bound via Empirical Bernstein inequality
--------------------------------------------------------------
Maurer & Pontil (2009), Theorem 4 gives a one-sided bound.  With
probability at least 1 − δ:

    E[Z] ≥ Z̄_n − √( 2 · V_n · ln(2/δ) / n ) − 7 · ln(2/δ) / (3(n−1))

where:
    Z̄_n  = (1/n) Σ_i Z_i                         (sample mean)
    V_n   = 1/(n−1) Σ_i (Z_i − Z̄_n)²             (unbiased sample variance)

The lower bound is:
    LB(δ) = Z̄_n − t_n(δ)
    t_n(δ) = √(2 · V_n · ln(2/δ) / n) + 7 · ln(2/δ) / (3·(n−1))

Auditing criterion
------------------
If LB(δ) > K  (the claimed NAF level), we declare a violation with
confidence 1 − δ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .models import LanguageModel


@dataclass
class KLEstimate:
    """Container for a single prompt's KL estimation result."""
    prompt_text: str
    kl_mean: float          # Z̄_n  (point estimate)
    sample_var: float       # V_n
    lower_bound: float      # LB(δ)
    upper_bound: float      # UB(δ) — for informational completeness
    bernstein_correction: float  # t_n(δ)
    n_samples: int
    delta: float
    violates: bool          # lower_bound > claimed_K
    claimed_K: float
    # raw samples kept for downstream analysis / ablations
    z_samples: Optional[torch.Tensor] = None


def empirical_bernstein_bound(
    z: torch.Tensor,
    delta: float = 0.05,
) -> tuple[float, float, float, float, float]:
    """Compute the empirical Bernstein lower and upper bounds.

    Parameters
    ----------
    z : (n,) tensor of privacy-loss samples Z_i = log p(y_i) - log q(y_i)
    delta : confidence parameter (default 0.05)

    Returns
    -------
    kl_mean, sample_var, lower_bound, upper_bound, correction
    """
    n = z.numel()
    assert n >= 2, "Need at least 2 samples for variance estimate"
    z_f = z.double()  # high precision

    kl_mean = z_f.mean().item()
    sample_var = z_f.var(unbiased=True).item()  # 1/(n-1) Σ(Z_i - Z̄)²

    ln_term = math.log(2.0 / delta)
    # Bernstein correction:
    #   t_n(δ) = √(2 V_n ln(2/δ) / n) + 7 ln(2/δ) / (3(n-1))
    correction = math.sqrt(2.0 * max(sample_var, 0.0) * ln_term / n) + \
                 7.0 * ln_term / (3.0 * (n - 1))

    lower_bound = kl_mean - correction
    upper_bound = kl_mean + correction  # symmetric for info

    return kl_mean, sample_var, lower_bound, upper_bound, correction


def estimate_kl(
    prompt_text: str,
    risky: LanguageModel,
    safe: LanguageModel,
    n_samples: int = 500,
    delta: float = 0.05,
    claimed_K: float = 1.0,
    temperature: float = 1.0,
    batch_size: int = 64,
    keep_samples: bool = False,
) -> KLEstimate:
    """Estimate KL(P_risky(·|x) ∥ P_safe(·|x)) for a single prompt.

    Procedure:
        1. Tokenise prompt → input_ids
        2. Sample n tokens y_i ~ P_risky(·|x)  (and get log P_risky(y_i|x))
        3. Evaluate log P_safe(y_i|x) for each sampled token
        4. Compute Z_i = log P_risky(y_i|x) − log P_safe(y_i|x)
        5. Apply Empirical Bernstein inequality for the lower bound
    """
    # Encode prompt (use risky model's tokenizer; assume shared vocab for GPT-2 family)
    input_ids = risky.encode(prompt_text)

    # Step 2: sample from risky model
    sampled_ids, log_p_risky = risky.sample_next_tokens(
        input_ids, n=n_samples, temperature=temperature, batch_size=batch_size,
    )

    # Step 3: evaluate safe model's log-probs for the same tokens
    log_p_safe = safe.log_probs_of_tokens(input_ids, sampled_ids)

    # Step 4: privacy loss samples
    z = (log_p_risky - log_p_safe).cpu()  # (n,)

    # Step 5: Bernstein bound
    kl_mean, sample_var, lb, ub, corr = empirical_bernstein_bound(z, delta)

    return KLEstimate(
        prompt_text=prompt_text,
        kl_mean=kl_mean,
        sample_var=sample_var,
        lower_bound=lb,
        upper_bound=ub,
        bernstein_correction=corr,
        n_samples=n_samples,
        delta=delta,
        violates=(lb > claimed_K),
        claimed_K=claimed_K,
        z_samples=z if keep_samples else None,
    )


def estimate_kl_sequence(
    prompt_text: str,
    risky: LanguageModel,
    safe: LanguageModel,
    n_sequences: int = 100,
    seq_len: int = 16,
    delta: float = 0.05,
    claimed_K: float = 1.0,
    temperature: float = 1.0,
    keep_samples: bool = False,
) -> KLEstimate:
    """Estimate *sequence-level* KL(P_risky ∥ P_safe) for a prompt.

    Instead of drawing single next-tokens, this generates full sequences
    of length *seq_len* autoregressively from the risky model, then
    scores each sequence under both models.  The privacy-loss sample is:

        Z_i = Σ_{t=1}^{T} [ log P_risky(y_t | x, y_{<t})
                           - log P_safe (y_t | x, y_{<t}) ]

    This addresses assumption A5 from the report: token-level KL
    lower-bounds sequence-level KL by the chain rule, but this
    estimator measures the actual sequence-level quantity.
    """
    import torch.nn.functional as F

    input_ids = risky.encode(prompt_text)  # (T_prompt,)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # (1, T_prompt)

    z_samples_list = []

    for _ in range(n_sequences):
        cur_ids = input_ids.clone()
        seq_log_ratio = 0.0

        for _t in range(seq_len):
            # Risky model: get distribution & sample one token
            with torch.inference_mode():
                r_logits = risky.model(input_ids=cur_ids).logits[:, -1, :].float()
                if temperature != 1.0:
                    r_logits = r_logits / temperature
                r_log_probs = F.log_softmax(r_logits, dim=-1).squeeze(0)  # (V,)

                s_logits = safe.model(input_ids=cur_ids).logits[:, -1, :].float()
                if temperature != 1.0:
                    s_logits = s_logits / temperature
                s_log_probs = F.log_softmax(s_logits, dim=-1).squeeze(0)  # (V,)

            # Sample from risky
            probs = r_log_probs.exp()
            y_t = torch.multinomial(probs, num_samples=1)  # (1,)

            seq_log_ratio += (r_log_probs[y_t] - s_log_probs[y_t]).item()

            # Extend the sequence
            cur_ids = torch.cat([cur_ids, y_t.unsqueeze(0)], dim=-1)

        z_samples_list.append(seq_log_ratio)

    z = torch.tensor(z_samples_list, dtype=torch.float64)
    kl_mean, sample_var, lb, ub, corr = empirical_bernstein_bound(z, delta)

    return KLEstimate(
        prompt_text=prompt_text,
        kl_mean=kl_mean,
        sample_var=sample_var,
        lower_bound=lb,
        upper_bound=ub,
        bernstein_correction=corr,
        n_samples=n_sequences,
        delta=delta,
        violates=(lb > claimed_K),
        claimed_K=claimed_K,
        z_samples=z if keep_samples else None,
    )
