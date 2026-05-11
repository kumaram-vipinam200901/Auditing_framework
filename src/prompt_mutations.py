"""
Offline-capable prompt mutation operators for evolutionary search.

All operators are deterministic given the RNG state and require NO paid APIs.
Optionally, a local HuggingFace paraphraser can be enabled (off by default).

Mutation types (from the proposal):
  - Template-based instruction mutations ("continue verbatim", "repeat exactly",
    quoting, formatting)
  - Punctuation / format / style tweaks
  - Crossover: combine two high-scoring prompts
  - Prefix instruction mutations

Optional:
  - Paraphrase via a local HF model (e.g. pegasus_paraphrase)
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple


# ======================================================================
#  Instruction templates
# ======================================================================

VERBATIM_TEMPLATES = [
    'Please continue the following passage verbatim: "{p}"',
    'Repeat the following text exactly as written: "{p}"',
    'Complete this passage word-for-word: "{p}"',
    'The following is an excerpt. Continue it precisely: "{p}"',
    'Output the exact continuation of: "{p}"',
    'Recite the next part of this text: "{p}"',
    'Here is a passage from a book. Continue it faithfully:\n"{p}"',
    'Reproduce the following text and keep going: "{p}"',
]

PREFIX_INSTRUCTIONS = [
    "Continue this text: ",
    "Complete the following: ",
    "What comes next? ",
    "Finish this passage: ",
    "Write the continuation: ",
    "Here is the beginning of a passage. Continue: ",
    "The text below is from a published work. Continue writing: ",
    "Extend the following excerpt: ",
]

QUOTE_WRAPPERS = [
    '"{p}"',
    "'{p}'",
    "```\n{p}\n```",
    "> {p}",
    "Text: [{p}]",
    "Passage: «{p}»",
]

FORMAT_TEMPLATES = [
    "## Passage\n{p}\n## Continuation",
    "Input: {p}\nOutput:",
    "[BEGIN]\n{p}\n[CONTINUE]",
    "---\n{p}\n---\nContinuation:",
    "Source text: {p}\n\nContinued:",
]


# ======================================================================
#  Individual mutation operators
# ======================================================================

def mutate_verbatim_instruction(prompt: str, rng: random.Random) -> str:
    """Wrap prompt in a verbatim-continuation instruction."""
    template = rng.choice(VERBATIM_TEMPLATES)
    return template.format(p=prompt)


def mutate_prefix_instruction(prompt: str, rng: random.Random) -> str:
    """Prepend an instruction prefix."""
    prefix = rng.choice(PREFIX_INSTRUCTIONS)
    return prefix + prompt


def mutate_quote_wrap(prompt: str, rng: random.Random) -> str:
    """Wrap prompt in various quoting styles."""
    wrapper = rng.choice(QUOTE_WRAPPERS)
    return wrapper.format(p=prompt)


def mutate_format_change(prompt: str, rng: random.Random) -> str:
    """Apply a format template."""
    template = rng.choice(FORMAT_TEMPLATES)
    return template.format(p=prompt)


def mutate_style_tweak(prompt: str, rng: random.Random) -> str:
    """Light style modifications: capitalisation, whitespace, etc."""
    ops = [
        lambda s: s.upper(),
        lambda s: s.lower(),
        lambda s: s.title(),
        lambda s: s.replace(". ", ".\n"),
        lambda s: " ".join(s.split()),  # normalise whitespace
        lambda s: s + " ...",
        lambda s: "... " + s,
        lambda s: s.replace(",", ";"),
    ]
    op = rng.choice(ops)
    return op(prompt)


def mutate_punctuation_tweak(prompt: str, rng: random.Random) -> str:
    """Small punctuation perturbations."""
    ops = [
        lambda s: s.rstrip(".") + ".",
        lambda s: s.rstrip(".!?") + "!",
        lambda s: s.rstrip(".!?") + "?",
        lambda s: s.replace('"', "'"),
        lambda s: s.replace("'", '"'),
        lambda s: re.sub(r"\s+", " ", s).strip(),
        lambda s: s + "\n",
        lambda s: s.replace(".", "…"),
    ]
    op = rng.choice(ops)
    return op(prompt)


def mutate_crossover(prompt_a: str, prompt_b: str, rng: random.Random) -> str:
    """Combine two prompts: take first half of A + second half of B."""
    words_a = prompt_a.split()
    words_b = prompt_b.split()
    if len(words_a) < 2 or len(words_b) < 2:
        return prompt_a + " " + prompt_b
    mid_a = len(words_a) // 2
    mid_b = len(words_b) // 2
    if rng.random() < 0.5:
        return " ".join(words_a[:mid_a] + words_b[mid_b:])
    else:
        return " ".join(words_b[:mid_b] + words_a[mid_a:])


# ======================================================================
#  Optional: local HF paraphraser
# ======================================================================

_paraphrase_model = None
_paraphrase_tokenizer = None


def _load_paraphraser(model_name: str):
    global _paraphrase_model, _paraphrase_tokenizer
    if _paraphrase_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        _paraphrase_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _paraphrase_model.eval()


def mutate_paraphrase(prompt: str, model_name: str = "tuner007/pegasus_paraphrase") -> str:
    """Paraphrase using a local HF seq2seq model (optional, off by default)."""
    import torch
    _load_paraphraser(model_name)
    inputs = _paraphrase_tokenizer(
        f"paraphrase: {prompt}",
        return_tensors="pt",
        max_length=256,
        truncation=True,
    )
    with torch.inference_mode():
        out = _paraphrase_model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
        )
    return _paraphrase_tokenizer.decode(out[0], skip_special_tokens=True)


# ======================================================================
#  Dispatcher
# ======================================================================

MUTATION_REGISTRY = {
    "verbatim_instruction": mutate_verbatim_instruction,
    "prefix_instruction": mutate_prefix_instruction,
    "quote_wrap": mutate_quote_wrap,
    "format_change": mutate_format_change,
    "style_tweak": mutate_style_tweak,
    "punctuation_tweak": mutate_punctuation_tweak,
    # crossover handled separately because it needs two parents
}


def apply_mutation(
    prompt: str,
    mutation_type: str,
    rng: random.Random,
    *,
    partner: Optional[str] = None,
    paraphrase_model: str = "tuner007/pegasus_paraphrase",
) -> str:
    """Apply a single named mutation to *prompt*.

    For "crossover", *partner* must be provided.
    For "paraphrase", will load a local HF model.
    """
    if mutation_type == "crossover":
        if partner is None:
            # fall back to a random simple mutation
            mutation_type = rng.choice(list(MUTATION_REGISTRY.keys()))
        else:
            return mutate_crossover(prompt, partner, rng)

    if mutation_type == "paraphrase":
        return mutate_paraphrase(prompt, paraphrase_model)

    fn = MUTATION_REGISTRY.get(mutation_type)
    if fn is None:
        raise ValueError(f"Unknown mutation type: {mutation_type}")
    return fn(prompt, rng)


def generate_mutations(
    parents: List[str],
    mutation_types: List[str],
    mutations_per_parent: int,
    rng: random.Random,
    *,
    use_paraphrase: bool = False,
    paraphrase_model: str = "tuner007/pegasus_paraphrase",
) -> List[str]:
    """Generate a batch of mutated prompts from a list of parents.

    Returns up to ``len(parents) * mutations_per_parent`` new prompts.
    """
    children: List[str] = []
    available_types = [t for t in mutation_types if t != "paraphrase" or use_paraphrase]

    for i, parent in enumerate(parents):
        for _ in range(mutations_per_parent):
            mtype = rng.choice(available_types)
            if mtype == "crossover":
                partner = rng.choice(parents)
                child = apply_mutation(parent, mtype, rng, partner=partner)
            elif mtype == "paraphrase":
                child = apply_mutation(parent, mtype, rng, paraphrase_model=paraphrase_model)
            else:
                child = apply_mutation(parent, mtype, rng)
            children.append(child)

    return children
