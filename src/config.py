"""
Configuration loading and validation for NAF auditing experiments.

Supports YAML config files with CLI overrides. All paths are resolved
relative to the project root (parent of src/).
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    risky_model: str = "gpt2-medium"
    safe_model: str = "gpt2"
    # optional: additional risky models for the full experiment
    extra_risky_models: List[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    name: str = "bookmia"  # "bookmia" | "wikitext" | "local"
    hf_dataset: str = "swj0419/BookMIA"
    hf_split: str = "train"
    hf_text_field: str = "text"
    hf_label_field: str = "label"  # 1 = member (book text)
    local_path: Optional[str] = None  # fallback .txt / .jsonl
    num_prompts: int = 50
    max_prompt_tokens: int = 64
    member_only: bool = True  # only use member texts (label==1)


@dataclass
class SamplingConfig:
    num_samples: int = 500
    batch_size: int = 64
    temperature: float = 1.0


@dataclass
class SearchConfig:
    num_rounds: int = 5
    top_k: int = 10
    pool_size: int = 50
    mutations_per_parent: int = 2
    keep_fraction: float = 0.5
    mutation_types: List[str] = field(
        default_factory=lambda: [
            "verbatim_instruction",
            "quote_wrap",
            "style_tweak",
            "punctuation_tweak",
            "crossover",
            "prefix_instruction",
            "format_change",
        ]
    )
    use_local_paraphrase: bool = False  # optional HF paraphraser
    paraphrase_model: str = "tuner007/pegasus_paraphrase"


@dataclass
class StatisticalConfig:
    delta: float = 0.05  # confidence parameter
    claimed_K: float = 1.0  # claimed NAF bound to audit against


@dataclass
class HardwareConfig:
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    mixed_precision: str = "none"  # "fp16" | "bf16" | "none"
    num_workers: int = 0
    torch_compile: bool = False
    cache_kv: bool = False


@dataclass
class OutputConfig:
    run_dir: str = "runs"
    run_id: Optional[str] = None  # auto-generated if None
    save_prompts: bool = True
    save_jsonl: bool = True
    save_csv: bool = True
    save_summary: bool = True


@dataclass
class AuditConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_dataclass(d: dict, cls):
    """Recursively convert nested dicts to the corresponding dataclass."""
    import dataclasses as dc
    field_types = {f.name: f.type for f in dc.fields(cls)}
    kwargs = {}
    for fname, ftype in field_types.items():
        if fname not in d:
            continue
        val = d[fname]
        # resolve string type annotations
        actual_type = _resolve_type(ftype, cls)
        if dc.is_dataclass(actual_type) and isinstance(val, dict):
            kwargs[fname] = _dict_to_dataclass(val, actual_type)
        else:
            kwargs[fname] = val
    return cls(**kwargs)


def _resolve_type(type_hint, parent_cls):
    """Resolve forward-reference / string annotations."""
    import dataclasses as dc
    mapping = {
        "ModelConfig": ModelConfig,
        "DatasetConfig": DatasetConfig,
        "SamplingConfig": SamplingConfig,
        "SearchConfig": SearchConfig,
        "StatisticalConfig": StatisticalConfig,
        "HardwareConfig": HardwareConfig,
        "OutputConfig": OutputConfig,
    }
    if isinstance(type_hint, str):
        return mapping.get(type_hint, str)
    return type_hint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> AuditConfig:
    """Load a YAML config and return an AuditConfig dataclass."""
    with open(yaml_path, "r") as f:
        raw: dict = yaml.safe_load(f) or {}
    if overrides:
        _deep_update(raw, overrides)
    return _dict_to_dataclass(raw, AuditConfig)


def resolve_device(cfg: HardwareConfig) -> torch.device:
    """Select the best available device."""
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set deterministic seeds everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(cfg: AuditConfig) -> Path:
    """Create and return the run output directory."""
    if cfg.output.run_id is None:
        import datetime
        cfg.output.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(cfg.output.run_dir) / cfg.output.run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def build_cli_parser(description: str = "NAF Audit") -> argparse.ArgumentParser:
    """Return a CLI parser with --config and common overrides."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_prompts", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--claimed_K", type=float, default=None)
    p.add_argument("--run_id", type=str, default=None)
    return p


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI args to a nested dict suitable for _deep_update."""
    ov: Dict[str, Any] = {}
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.device is not None:
        ov.setdefault("hardware", {})["device"] = args.device
    if args.num_prompts is not None:
        ov.setdefault("dataset", {})["num_prompts"] = args.num_prompts
    if args.num_samples is not None:
        ov.setdefault("sampling", {})["num_samples"] = args.num_samples
    if args.num_rounds is not None:
        ov.setdefault("search", {})["num_rounds"] = args.num_rounds
    if args.claimed_K is not None:
        ov.setdefault("statistical", {})["claimed_K"] = args.claimed_K
    if args.run_id is not None:
        ov.setdefault("output", {})["run_id"] = args.run_id
    return ov
