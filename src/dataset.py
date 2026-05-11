"""
Dataset loading for NAF audit prompts.

Supports three modes (selected via config.dataset.name):
  1. "bookmia"  – load from HuggingFace  swj0419/BookMIA  (Shi et al. 2024)
  2. "wikitext" – load from HuggingFace  wikitext  (wikitext-103-raw-v1)
  3. "local"    – read prompts from a local .txt or .jsonl file

BookMIA contains passages from books; label==1 marks member (training) texts.
We take the first `max_prompt_tokens` tokens of each passage as the prompt.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List

from .config import DatasetConfig

logger = logging.getLogger(__name__)


def _truncate_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate *text* to at most *max_tokens* tokens, then decode back."""
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _is_clean_text(text: str) -> bool:
    """Return False for corrupted entries (null bytes, control chars, etc.)."""
    if "\x00" in text:
        return False
    # Reject if >10% non-printable characters
    non_printable = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
    if non_printable > 0.1 * len(text):
        return False
    return True


def load_prompts(cfg: DatasetConfig, tokenizer=None) -> List[str]:
    """Return a list of prompt strings according to *cfg*."""
    if cfg.name == "bookmia":
        return _load_bookmia(cfg, tokenizer)
    elif cfg.name == "wikitext":
        return _load_wikitext(cfg, tokenizer)
    elif cfg.name == "local":
        return _load_local(cfg, tokenizer)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.name}")


# ------------------------------------------------------------------
# BookMIA  (HuggingFace)
# ------------------------------------------------------------------

def _load_bookmia(cfg: DatasetConfig, tokenizer) -> List[str]:
    try:
        from datasets import load_dataset
        logger.info("Loading BookMIA from HuggingFace: %s", cfg.hf_dataset)
        ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
    except Exception as e:
        logger.warning("HuggingFace load failed (%s). Trying local fallback.", e)
        if cfg.local_path:
            return _load_local(cfg, tokenizer)
        raise RuntimeError(
            "Cannot load BookMIA and no local_path configured. "
            "Provide a local .txt/.jsonl via dataset.local_path in your config."
        ) from e

    # Auto-detect text field if the configured one doesn't exist
    sample_keys = ds.column_names
    text_field = cfg.hf_text_field
    if text_field not in sample_keys:
        # BookMIA uses 'snippet'; try common alternatives
        for candidate in ["snippet", "text", "passage", "content", "sentence"]:
            if candidate in sample_keys:
                text_field = candidate
                break
        else:
            text_field = sample_keys[0]
        logger.info("Text field '%s' not found; using '%s' instead", cfg.hf_text_field, text_field)

    texts: List[str] = []
    for row in ds:
        if cfg.member_only and row.get(cfg.hf_label_field) != 1:
            continue
        text = row[text_field].strip()
        if not text or not _is_clean_text(text):
            continue
        if tokenizer and cfg.max_prompt_tokens:
            text = _truncate_to_tokens(text, cfg.max_prompt_tokens, tokenizer)
        texts.append(text)
        if len(texts) >= cfg.num_prompts:
            break

    if len(texts) < cfg.num_prompts:
        logger.warning(
            "Only found %d prompts (requested %d). Using all available.",
            len(texts), cfg.num_prompts,
        )

    logger.info("Loaded %d prompts from BookMIA", len(texts))
    return texts


# ------------------------------------------------------------------
# WikiText-103
# ------------------------------------------------------------------

def _load_wikitext(cfg: DatasetConfig, tokenizer) -> List[str]:
    try:
        from datasets import load_dataset
        logger.info("Loading WikiText-103 from HuggingFace")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        logger.warning("HuggingFace load failed (%s). Trying local fallback.", e)
        if cfg.local_path:
            return _load_local(cfg, tokenizer)
        raise

    texts: List[str] = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 40:  # skip very short lines / headers
            continue
        if tokenizer and cfg.max_prompt_tokens:
            text = _truncate_to_tokens(text, cfg.max_prompt_tokens, tokenizer)
        texts.append(text)
        if len(texts) >= cfg.num_prompts:
            break

    logger.info("Loaded %d prompts from WikiText-103", len(texts))
    return texts


# ------------------------------------------------------------------
# Local file (.txt or .jsonl)
# ------------------------------------------------------------------

def _load_local(cfg: DatasetConfig, tokenizer) -> List[str]:
    path = Path(cfg.local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local prompt file not found: {path}")

    logger.info("Loading prompts from local file: %s", path)
    texts: List[str] = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", obj.get("prompt", "")).strip()
                if text:
                    if tokenizer and cfg.max_prompt_tokens:
                        text = _truncate_to_tokens(text, cfg.max_prompt_tokens, tokenizer)
                    texts.append(text)
                    if len(texts) >= cfg.num_prompts:
                        break
    else:
        # plain text: one prompt per line
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    if tokenizer and cfg.max_prompt_tokens:
                        text = _truncate_to_tokens(text, cfg.max_prompt_tokens, tokenizer)
                    texts.append(text)
                    if len(texts) >= cfg.num_prompts:
                        break

    logger.info("Loaded %d prompts from %s", len(texts), path)
    return texts
