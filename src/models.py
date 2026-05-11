"""
Model loading and next-token log-probability computation.

Supports any HuggingFace causal-LM.  Key design choices:
  - Uses log-softmax for numerical stability (avoids underflow).
  - Batched forward passes with torch.inference_mode().
  - Optional mixed-precision via torch.autocast.
  - Optional torch.compile for additional speed.

The KL estimator needs, for a given prefix x:
  1.  Samples  y_i ~ P_risky(·|x)   (tokens drawn from the risky model)
  2.  log P_risky(y_i | x)  and  log P_safe(y_i | x)

We provide both sampling and log-prob evaluation in a single batched call
to minimize redundant forward passes.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HardwareConfig, resolve_device

logger = logging.getLogger(__name__)


class LanguageModel:
    """Thin wrapper around a HuggingFace CausalLM for next-token operations."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        hw: HardwareConfig,
    ):
        self.model_name = model_name
        self.device = device
        self.hw = hw

        logger.info("Loading model %s → %s", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = self._resolve_dtype()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device)
        self.model.eval()

        if hw.torch_compile and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)

        self.vocab_size = self.model.config.vocab_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self) -> torch.dtype:
        if self.hw.mixed_precision == "fp16":
            return torch.float16
        if self.hw.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _autocast_ctx(self):
        if self.hw.mixed_precision == "none":
            import contextlib
            return contextlib.nullcontext()
        dtype = torch.float16 if self.hw.mixed_precision == "fp16" else torch.bfloat16
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        return torch.autocast(device_type=device_type, dtype=dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return input_ids tensor (1-D, on device)."""
        ids = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
        return ids.to(self.device)

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def next_token_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return log P(· | input_ids) over the full vocabulary.

        Parameters
        ----------
        input_ids : (B, T) or (T,)  token IDs on self.device

        Returns
        -------
        log_probs : (B, V)  log-softmax of the last-position logits
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with self._autocast_ctx():
            outputs = self.model(input_ids=input_ids)
        # logits at the last position → (B, V)
        logits = outputs.logits[:, -1, :]
        return F.log_softmax(logits.float(), dim=-1)

    @torch.inference_mode()
    def sample_next_tokens(
        self,
        input_ids: torch.Tensor,
        n: int,
        temperature: float = 1.0,
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample *n* next tokens and return (sampled_ids, log_probs_of_sampled).

        Parameters
        ----------
        input_ids : (T,)  1-D prefix token ids
        n : number of i.i.d. samples to draw
        temperature : sampling temperature (1.0 = unmodified)
        batch_size : forward-pass batch size

        Returns
        -------
        sampled_ids : (n,)  sampled token indices
        log_probs   : (n,)  log P(sampled_id | prefix)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (1, T)

        # single forward pass to get the distribution
        with self._autocast_ctx():
            outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :].float()  # (1, V)
        if temperature != 1.0:
            logits = logits / temperature
        log_probs_full = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

        # sample n tokens from the categorical distribution
        probs = log_probs_full.exp()
        sampled = torch.multinomial(probs, num_samples=n, replacement=True)  # (n,)
        log_probs_sampled = log_probs_full[sampled]  # (n,)

        return sampled, log_probs_sampled

    @torch.inference_mode()
    def log_probs_of_tokens(
        self,
        input_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate log P(token_id | prefix) for each token in *token_ids*.

        Parameters
        ----------
        input_ids : (T,)   prefix
        token_ids : (n,)   tokens whose probability we want

        Returns
        -------
        log_probs : (n,)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with self._autocast_ctx():
            outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :].float().squeeze(0)  # (V,)
        log_probs_full = F.log_softmax(logits, dim=-1)
        return log_probs_full[token_ids]


class AnchoredDecodingModel:
    """Implements anchored decoding (He et al., 2026).

    Faithful reimplementation of the Newton solver from the official repo:
      https://github.com/jacqueline-he/anchored-decoding

    The fused distribution at each token position is:
        q_θ(y|x) ∝ P_safe(y|x)^{1-θ} · P_risky(y|x)^{θ}

    where θ ∈ [0,1] is found via Newton-Raphson (with bisection safeguard)
    such that KL(q_θ ∥ P_safe) ≤ K (the NAF budget).

    θ=0 → q = P_safe (full anchoring, KL=0)
    θ=1 → q = P_risky (no anchoring, maximum KL)

    This wrapper exposes the same API as LanguageModel so the existing
    audit pipeline can treat it as a drop-in replacement.
    """

    def __init__(
        self,
        risky: LanguageModel,
        safe: LanguageModel,
        K: float = 1.0,
        solver_max_iter: int = 20,
    ):
        self.risky = risky
        self.safe = safe
        self.K = K
        self.solver_max_iter = solver_max_iter

        # Expose attributes that the rest of the pipeline expects
        self.model_name = f"anchored({risky.model_name},{safe.model_name},K={K})"
        self.device = risky.device
        self.hw = risky.hw
        self.tokenizer = risky.tokenizer
        self.vocab_size = risky.vocab_size

    # ------------------------------------------------------------------
    # Newton solver (matches reference _solve_theta_newton)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        """KL(P || Q) computed safely in fp32.  Returns shape (B,)."""
        p = log_p.exp()
        diff = log_p - log_q
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        kl = (p * diff).sum(dim=-1)
        return kl.clamp(min=0.0)

    def _solve_theta(
        self,
        log_pc: torch.Tensor,   # (B, V) fp32 log P_safe
        log_pd: torch.Tensor,   # (B, V) fp32 log P_risky
        k_radius: float,
    ) -> torch.Tensor:
        """Solve for θ = w_d such that KL(q_θ ∥ p_c) ≤ k_radius.

        Returns (B, 1) tensor of risky-model weights θ.
        """
        B, V = log_pc.shape
        device = log_pc.device

        k_t = torch.full((B,), k_radius, device=device, dtype=torch.float32)

        # Corner: k=0 → use safe model only
        if k_radius <= 0.0:
            return torch.zeros((B, 1), device=device, dtype=torch.float32)

        # Corner: risky model already within budget → use risky directly
        KL_pd_pc = self._safe_kl(log_pd, log_pc)
        mask_use_pd = KL_pd_pc <= k_t

        w_d = torch.empty((B, 1), device=device, dtype=torch.float32)
        w_d[mask_use_pd] = 1.0

        active = ~mask_use_pd
        if not active.any():
            return w_d

        # Active subset needs Newton solve
        log_pc_a = log_pc[active]
        log_pd_a = log_pd[active]
        k_a = k_t[active]
        Ba = log_pc_a.size(0)

        a = log_pd_a - log_pc_a
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        # Bracket θ ∈ [0, 1]
        lo = torch.zeros(Ba, device=device, dtype=torch.float32)
        hi = torch.ones(Ba, device=device, dtype=torch.float32)
        theta = torch.clamp(k_a / (k_a + 1.0), 1e-4, 1.0 - 1e-4)

        eps = 1e-9
        for _ in range(self.solver_max_iter):
            # q_θ ∝ p_c · exp(θ·a)  →  log q = log_pc + θ·a − logZ
            q_unnorm = log_pc_a + theta[:, None] * a
            logZ = torch.logsumexp(q_unnorm, dim=-1)
            q_unnorm.sub_(logZ[:, None])
            q_unnorm.exp_()                    # now holds q (probabilities)

            mean_a = (q_unnorm * a).sum(dim=-1)
            mean_a2 = (q_unnorm * (a * a)).sum(dim=-1)
            var_a = (mean_a2 - mean_a * mean_a).clamp_min(0.0)

            KL = theta * mean_a - logZ
            KL = torch.nan_to_num(KL, nan=float("inf"), posinf=float("inf"), neginf=0.0)

            f = KL - k_a

            # Update bracket
            hi = torch.where(f > 0, theta, hi)
            lo = torch.where(f <= 0, theta, lo)

            # Newton step with bisection fallback
            fp = (theta * var_a).clamp_min(eps)
            theta_new = theta - f / fp

            bad = (theta_new <= lo) | (theta_new >= hi) | ~torch.isfinite(theta_new)
            theta = torch.where(bad, 0.5 * (lo + hi), theta_new)

            if (hi - lo).max() < 1e-6:
                break

        # Final bisection projection for numerical feasibility
        def kl_theta(th):
            q_log = log_pc_a + th[:, None] * a
            lZ = torch.logsumexp(q_log, dim=-1)
            log_q = q_log - lZ[:, None]
            return self._safe_kl(log_q, log_pc_a)

        for _ in range(12):
            mid = 0.5 * (lo + hi)
            KL_mid = kl_theta(mid)
            feas = KL_mid <= k_a
            lo = torch.where(feas, mid, lo)
            hi = torch.where(feas, hi, mid)

        theta = lo  # feasible by construction
        w_d[active] = theta[:, None]
        return w_d

    # ------------------------------------------------------------------
    # Core: compute the fused log-distribution
    # ------------------------------------------------------------------

    def _fused_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return log q_θ(· | input_ids) over the full vocabulary."""
        log_pd = self.risky.next_token_log_probs(input_ids).float()  # (B, V)
        log_pc = self.safe.next_token_log_probs(input_ids).float()   # (B, V)

        w_d = self._solve_theta(log_pc, log_pd, self.K)  # (B, 1)
        w_c = 1.0 - w_d

        # Geometric blend: log q = w_c * log_pc + w_d * log_pd (unnormalised)
        term_c = w_c * log_pc
        term_d = w_d * log_pd
        term_c = torch.nan_to_num(term_c, nan=0.0)
        term_d = torch.nan_to_num(term_d, nan=0.0)
        log_fused = F.log_softmax(term_c + term_d, dim=-1)

        return log_fused

    # ------------------------------------------------------------------
    # Public API (mirrors LanguageModel)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        return self.risky.encode(text)

    def decode(self, ids: torch.Tensor) -> str:
        return self.risky.decode(ids)

    @torch.inference_mode()
    def sample_next_tokens(
        self,
        input_ids: torch.Tensor,
        n: int,
        temperature: float = 1.0,
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n tokens from the fused distribution."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        log_fused = self._fused_log_probs(input_ids).squeeze(0)  # (V,)
        if temperature != 1.0:
            log_fused = log_fused / temperature
            log_fused = log_fused - torch.logsumexp(log_fused, dim=-1, keepdim=True)

        probs = log_fused.exp()
        sampled = torch.multinomial(probs, num_samples=n, replacement=True)
        log_probs_sampled = log_fused[sampled]
        return sampled, log_probs_sampled

    @torch.inference_mode()
    def log_probs_of_tokens(
        self,
        input_ids: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate log P_fused(token_id | prefix) for each token."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        log_fused = self._fused_log_probs(input_ids).squeeze(0)  # (V,)
        return log_fused[token_ids]

    @torch.inference_mode()
    def next_token_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return log P_fused(· | input_ids) over the full vocabulary."""
        return self._fused_log_probs(input_ids)


def load_model_pair(
    risky_name: str,
    safe_name: str,
    hw: HardwareConfig,
) -> Tuple[LanguageModel, LanguageModel]:
    """Convenience: load both models on the resolved device."""
    device = resolve_device(hw)
    logger.info("Resolved device: %s", device)
    risky = LanguageModel(risky_name, device, hw)
    safe = LanguageModel(safe_name, device, hw)
    return risky, safe
