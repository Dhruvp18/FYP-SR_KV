"""Single-needle "needle in a haystack" retrieval.

A needle sentence carrying a random magic number is inserted at a given depth
into filler text, the model is asked to recall the number, and the answer is
scored by exact string match on the number. Depth sweeps 0/25/50/75/100% and
context length sweeps 2k/4k/8k/16k.

Two haystack sources:

* `synthetic` (default) - a deterministic shuffle of neutral filler sentences.
  No download, byte-identical on every machine, which matters because Kaggle
  sessions get killed and restarted mid-sweep.
* `pg` - Paul Graham essays via `datasets`, the corpus the original NIAH used.
  More natural text and therefore a harder distractor set; needs network.

Report both if you use `pg`; do not compare a `synthetic` number against a
published `pg` number, they are not the same task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

DEPTHS = (0, 25, 50, 75, 100)
CONTEXT_LENGTHS = (2048, 4096, 8192, 16384)

_FILLER = [
    "The grass is green and the sky is a pale, cloudless blue.",
    "Trains leave the station every twenty minutes on weekdays.",
    "She kept a notebook of small observations about the weather.",
    "The library closes at six, except on Thursdays when it stays open late.",
    "Rain collected in the gutter and ran down toward the street.",
    "He learned to make bread from a recipe written on an index card.",
    "The map on the wall showed roads that no longer existed.",
    "Sunlight moved slowly across the floorboards during the afternoon.",
    "Someone had left a bicycle leaning against the fence.",
    "The kettle whistled and then went quiet again.",
    "Birds gathered on the wire before flying off together.",
    "The river was low that summer and the stones showed through.",
    "A radio played faintly from an open window across the courtyard.",
    "They walked the long way home to keep talking.",
    "The paint on the door had cracked into small islands.",
    "Snow fell for an hour and then turned to rain.",
]

_CITIES = [
    "Mumbai", "Lisbon", "Nairobi", "Osaka", "Bogota", "Helsinki", "Cairo",
    "Toronto", "Jakarta", "Warsaw", "Lima", "Dublin", "Seoul", "Chennai",
]

NEEDLE_TEMPLATE = "The special magic {city} number is: {number}."
QUESTION_TEMPLATE = "What is the special magic {city} number mentioned in the text above? Answer with the number only."


@dataclass
class NIAHSample:
    """One NIAH cell instance."""

    context_len: int
    depth: int
    city: str
    number: str
    sample_idx: int
    context: str = field(repr=False, default="")
    question: str = ""
    answer: str = ""

    @property
    def task_id(self) -> str:
        return f"niah/ctx{self.context_len}/depth{self.depth}/s{self.sample_idx}"


def _filler_text(rng: random.Random, min_chars: int) -> str:
    out: list[str] = []
    total = 0
    while total < min_chars:
        line = rng.choice(_FILLER)
        out.append(line)
        total += len(line) + 1
    return " ".join(out)


def _load_pg_text(min_chars: int) -> str:
    from datasets import load_dataset

    ds = load_dataset("sgoel9/paul_graham_essays", split="train")
    chunks, total = [], 0
    for row in ds:
        text = row.get("text") or row.get("essay") or ""
        chunks.append(text)
        total += len(text)
        if total >= min_chars:
            break
    return "\n\n".join(chunks)


def _haystack_ids(tokenizer, rng, corpus: str, min_tokens: int) -> list[int]:
    """Enough haystack for the longest context, measured in tokens not characters."""
    if corpus == "pg":
        ids = tokenizer(_load_pg_text(min_tokens * 8), add_special_tokens=False)["input_ids"]
        if len(ids) < min_tokens:
            raise ValueError(f"pg corpus gave {len(ids)} tokens, need {min_tokens}")
        return ids

    ids: list[int] = []
    while len(ids) < min_tokens:
        ids.extend(tokenizer(_filler_text(rng, 8000), add_special_tokens=False)["input_ids"])
    return ids


def build_samples(
    tokenizer,
    *,
    context_lengths=CONTEXT_LENGTHS,
    depths=DEPTHS,
    n_samples: int = 3,
    corpus: str = "synthetic",
    seed: int = 1234,
) -> list[NIAHSample]:
    """Materialise every (context_len, depth, sample) cell.

    The needle is placed by *token* depth, not character depth, so "50%" means
    the same thing at every context length.
    """
    rng = random.Random(seed)
    max_ctx = max(context_lengths)
    hay_ids = _haystack_ids(tokenizer, rng, corpus, max_ctx + 64)

    samples: list[NIAHSample] = []
    for context_len in context_lengths:
        if len(hay_ids) < context_len:
            raise ValueError(
                f"haystack is {len(hay_ids)} tokens, need {context_len}; "
                "raise the multiplier in build_samples"
            )
        for depth in depths:
            for sample_idx in range(n_samples):
                city = _CITIES[(depth + sample_idx * 7 + context_len) % len(_CITIES)]
                number = str(rng.randint(100000, 999999))
                needle = NEEDLE_TEMPLATE.format(city=city, number=number)
                needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]

                body = hay_ids[: max(0, context_len - len(needle_ids))]
                cut = int(len(body) * depth / 100)
                full_ids = body[:cut] + needle_ids + body[cut:]
                samples.append(
                    NIAHSample(
                        context_len=context_len,
                        depth=depth,
                        city=city,
                        number=number,
                        sample_idx=sample_idx,
                        context=tokenizer.decode(full_ids),
                        question=QUESTION_TEMPLATE.format(city=city),
                        answer=number,
                    )
                )
    return samples


def score(sample: NIAHSample, generated_text: str) -> float:
    """1.0 if the magic number appears in the generation, else 0.0."""
    return 1.0 if sample.answer in generated_text.replace(",", "") else 0.0
