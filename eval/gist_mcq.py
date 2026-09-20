"""Long-range gist retention under compression: does a *coarse* judgment
survive eviction better when clustering has genuinely redundant material to
average, and the question never needs an exact token back?

Motivation (PREREGISTRATION.md, addendum H3). NIAH needs one exact span back;
LongBench's qa_f1/rouge_l need near-exact n-grams; neither task gives
clustering redundant material to work with (each candidate sentence differs
from its neighbours), and neither metric can credit an averaged vector even
when it helps the model's internal state. This task inverts both properties
on purpose:

* the fact the question needs is restated several times, scattered across the
  document in different phrasing - genuinely similar key vectors for
  cosine-clustering to group, unlike a single needle or a dense report;
* the answer is multiple-choice, scored by which *letter* the model outputs,
  not by string overlap with a reference span - a residual, blurry attention
  target can support the right choice without reproducing any wording.

Two variants probe two distinct claims about what a centroid preserves:

* `attribution` - does an entity/attribute link survive when every individual
  restatement is, on its own, unremarkable to windowed-attention scoring
  (nothing in the observation window cares about it yet, because the question
  hasn't been asked)?
* `aggregation` - does the *relative frequency* of scattered mentions survive,
  not just their existence? A centroid averaged over five mentions of one
  topic should point more strongly toward it than one averaged over a single
  mention of another - hard eviction has no equivalent of "how many
  almost-evicted tokens agreed", since a token either survives whole or is
  gone.

Both variants reuse NIAH's haystack builder (`corpus="pg"` by default here -
see `build_samples`'s docstring for why synthetic filler defeats the task)
but are otherwise independent of niah.py.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

import torch

from eval.memory import peak_memory
from eval.niah import _haystack_ids
from src.attn_patch import attach_cache

LETTERS = "ABCD"
VARIANTS = ("attribution", "aggregation")

ENTITIES = [
    "Priya Osei", "Marcus Whitfield", "Elena Kade", "Tomas Reyes",
    "Ingrid Solberg", "Kwame Boateng", "Yuki Tanaka", "Fatima Haidari",
    "Declan Murphy", "Sana Rahimi",
]
COLORS = ["Red", "Blue", "Green", "Yellow", "Orange", "Purple"]
TOPICS = ["Billing", "Shipping", "Login", "Refund", "Warranty"]

ATTRIBUTION_TEMPLATES = [
    "{entity} was formally assigned to the {color} team at the start of the project.",
    "During the mid-project review, {entity}'s contributions were logged under "
    "the {color} team's workstream.",
    "The {color} team's final report specifically credited {entity} for the "
    "initial design draft.",
    "{entity} later confirmed in an internal memo that the {color} team had "
    "approved the change.",
]
DECOY_TEMPLATE = "{entity} briefly consulted with the {color} team on an unrelated matter."
#: spread wide on purpose - by the time compression triggers near the end of a
#: long prefill, none of these is recent enough for windowed-attention scoring
#: to have any reason to protect it on its own.
_ATTRIBUTION_DEPTHS = (8, 32, 58, 84)
_DECOY_DEPTHS = (20, 68)

AGGREGATION_TEMPLATES = [
    "A customer raised a concern about {topic} during the call.",
    "There was another comment about {topic} issues later in the conversation.",
    "{topic} came up again when the representative followed up.",
]
_AGGREGATION_DEPTHS = (6, 19, 32, 45, 58, 71, 84)
_MAJORITY_COUNT = 5


@dataclass
class GistSample:
    """One gist_mcq cell instance."""

    context_len: int
    variant: str
    sample_idx: int
    options: list[str]
    answer_letter: str
    context: str = field(repr=False, default="")
    question: str = ""
    max_new_tokens: int = 12

    @property
    def task_id(self) -> str:
        return f"gist/{self.variant}/ctx{self.context_len}/s{self.sample_idx}"


def _splice(body_ids: list[int], inserts: list[tuple[int, list[int]]]) -> list[int]:
    """Insert each (cut_index, ids) into body_ids, left to right.

    Cut indices are computed against the *un-spliced* body (see
    `_make_context`), so inserting them in increasing order never shifts a
    later cut point - the same trick `niah.build_samples` uses for its single
    needle, generalised to many.
    """
    ordered = sorted(inserts, key=lambda pair: pair[0])
    out: list[int] = []
    prev = 0
    for cut, ids in ordered:
        cut = max(prev, min(cut, len(body_ids)))
        out.extend(body_ids[prev:cut])
        out.extend(ids)
        prev = cut
    out.extend(body_ids[prev:])
    return out


def _make_context(
    tokenizer, hay_ids: list[int], context_len: int, fragments: list[tuple[int, str]]
) -> str:
    """Encode each (depth_pct, text) fragment and splice it into the haystack."""
    # A byte-level BPE tokenizer bakes a word's leading space into *its own*
    # token, not a trailing space on the previous word. Encoding `text` on its
    # own therefore gives its first token no leading space, and splicing that
    # straight into the middle of the haystack fuses it onto the previous word
    # ("closes atIngrid Solberg was..."). A leading space before encoding
    # fixes it; the surrounding filler already supplies the tail-side space
    # because `body[cut:]`'s first token keeps whatever space it originally
    # carried.
    encoded = [
        (depth, tokenizer(" " + text, add_special_tokens=False)["input_ids"])
        for depth, text in fragments
    ]
    total_insert = sum(len(ids) for _, ids in encoded)
    body = hay_ids[: max(0, context_len - total_insert)]
    inserts = [(int(len(body) * depth / 100), ids) for depth, ids in encoded]
    return tokenizer.decode(_splice(body, inserts))


def _options_block(option_pool: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(option_pool))


def _build_attribution(tokenizer, rng: random.Random, hay_ids, context_len):
    entity = rng.choice(ENTITIES)
    decoys = rng.sample([e for e in ENTITIES if e != entity], 2)

    colors = COLORS[:]
    rng.shuffle(colors)
    target_color, decoy_color_a, decoy_color_b, *rest = colors
    option_pool = [target_color] + rest[:3]
    rng.shuffle(option_pool)
    answer_letter = LETTERS[option_pool.index(target_color)]

    fragments = [
        (depth, tmpl.format(entity=entity, color=target_color))
        for depth, tmpl in zip(_ATTRIBUTION_DEPTHS, ATTRIBUTION_TEMPLATES)
    ]
    fragments += [
        (depth, DECOY_TEMPLATE.format(entity=decoy, color=color))
        for depth, decoy, color in zip(_DECOY_DEPTHS, decoys, (decoy_color_a, decoy_color_b))
    ]

    context = _make_context(tokenizer, hay_ids, context_len, fragments)
    question = (
        f"Based on the passage above, which team was {entity} assigned to? "
        "Answer with a single letter and nothing else.\n"
        f"{_options_block(option_pool)}\nAnswer:"
    )
    return context, question, option_pool, answer_letter


def _build_aggregation(tokenizer, rng: random.Random, hay_ids, context_len):
    topics = TOPICS[:]
    rng.shuffle(topics)
    majority, minor_a, minor_b, never_mentioned, _spare = topics

    mentions = [majority] * _MAJORITY_COUNT + [minor_a, minor_b]
    rng.shuffle(mentions)  # position must not leak which topic is the majority

    fragments = [
        (depth, AGGREGATION_TEMPLATES[i % len(AGGREGATION_TEMPLATES)].format(topic=topic))
        for i, (depth, topic) in enumerate(zip(_AGGREGATION_DEPTHS, mentions))
    ]

    option_pool = [majority, minor_a, minor_b, never_mentioned]
    rng.shuffle(option_pool)
    answer_letter = LETTERS[option_pool.index(majority)]

    context = _make_context(tokenizer, hay_ids, context_len, fragments)
    question = (
        "Based on the conversation above, which issue was raised most often? "
        "Answer with a single letter and nothing else.\n"
        f"{_options_block(option_pool)}\nAnswer:"
    )
    return context, question, option_pool, answer_letter


_BUILDERS = {"attribution": _build_attribution, "aggregation": _build_aggregation}


def build_samples(
    tokenizer,
    *,
    context_lengths,
    variants=VARIANTS,
    n_samples: int = 50,
    corpus: str = "pg",
    seed: int = 1234,
) -> list[GistSample]:
    """Materialise every (variant, context_len, sample) cell.

    `corpus="pg"` (Paul Graham essays, real prose) is the default here, unlike
    `niah.build_samples`'s `"synthetic"` default. A first GPU smoke test with
    synthetic filler saturated at 1.000 for every method, including at
    budget=0.15: the filler is short, repetitive, boilerplate sentences, so
    ANY importance score (attention-based or not) trivially ranks a novel
    fact sentence above it - the eviction policy never has to work for its
    answer, defeating the point of the task. Real prose gives windowed-
    attention scoring many equally "locally interesting" candidates to
    compete with the planted facts for budget, which is what actually
    exercises the difference between hard eviction and clustering.

    One shared rng, advanced in a fixed nested order, so resuming a run (or
    adding a method) reproduces byte-identical samples - the same discipline
    `niah.build_samples` uses.
    """
    for variant in variants:
        if variant not in _BUILDERS:
            raise ValueError(f"unknown gist_mcq variant {variant!r}; known: {sorted(_BUILDERS)}")

    rng = random.Random(seed)
    max_ctx = max(context_lengths)
    hay_ids = _haystack_ids(tokenizer, rng, corpus, max_ctx + 512)

    samples: list[GistSample] = []
    for variant in variants:
        builder = _BUILDERS[variant]
        for context_len in context_lengths:
            for sample_idx in range(n_samples):
                context, question, options, answer_letter = builder(
                    tokenizer, rng, hay_ids, context_len
                )
                samples.append(
                    GistSample(
                        context_len=context_len,
                        variant=variant,
                        sample_idx=sample_idx,
                        options=options,
                        answer_letter=answer_letter,
                        context=context,
                        question=question,
                    )
                )
    return samples


#: Tried in order of specificity, not "first A-D letter anywhere": a bare
#: `\b([ABCD])\b` also matches the English article "a" once the text is
#: upper-cased ("It was a cab" -> a spurious "A"), which would score whatever
#: ordinary prose precedes the model's actual choice. Requiring the letter to
#: look like an MCQ answer - "B)", after the word "answer", or at the very
#: start of the (stripped) generation - avoids that collision entirely.
_LETTER_PAREN_RE = re.compile(r"([ABCD])\)")
_AFTER_ANSWER_RE = re.compile(r"\bANSWER\s*:?\s*([ABCD])\b")
_LEADING_LETTER_RE = re.compile(r"^([ABCD])\b")


def score(sample: GistSample, generated_text: str) -> float:
    """1.0 if the generation's MCQ choice matches; 0.0 otherwise."""
    text = generated_text.strip().upper()
    for pattern in (_LETTER_PAREN_RE, _AFTER_ANSWER_RE, _LEADING_LETTER_RE):
        match = pattern.search(text)
        if match:
            return 1.0 if match.group(1) == sample.answer_letter else 0.0
    return 0.0


def _two_pass_texts(tokenizer, context: str, question: str) -> tuple[str, str]:
    """Split what `eval.memory.build_prompt` would produce into (passage,
    question) halves at the exact same boundary, so feeding them as two
    forward passes reproduces the same prompt a single-pass call would - but
    lets the passage's cache compress before the question exists to compress
    around (see `measure`'s docstring for why that distinction is the whole
    point of this task)."""
    if getattr(tokenizer, "chat_template", None):
        user = f"{context}\n\n{question}"
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        idx = wrapped.index(context) + len(context)
        return wrapped[:idx], wrapped[idx:]
    return context, "\n\n" + question


@torch.no_grad()
def measure(model, tokenizer, sample: GistSample, cache, *,
            max_new_tokens: int | None = None, device=None) -> dict:
    """Two forward passes, the same discipline `perplexity.measure` uses.

    A single combined (passage + question) forward - what `eval.memory.
    generate_and_measure` does for every other generation task - lets the
    compression step's observation window BE the question, since it is
    always the last thing in the prompt. Windowed-attention scoring can then
    "look ahead": it directly attends back to whatever the question names,
    regardless of how many times that content was mentioned or how it was
    evicted elsewhere, and every method (hard eviction included) recovers it
    trivially. This is not hypothetical - it is the same mechanism that
    saturates NIAH, and it was confirmed here on real hardware: two GPU smoke
    tests at full target scale (ctx=8192, budget=0.1, real prose) both came
    back at 1.000 accuracy for every method, including snapkv_unified, with a
    single-pass prompt.

    Splitting into two passes makes the passage's compression irreversible
    before the question exists: whatever a policy evicts (or merges) using
    only passage-internal signal is gone (or blurred) by the time the
    question is asked, and a policy cannot un-evict what it no longer holds.
    That is the property this experiment is actually meant to test.
    """
    device = device or next(model.parameters()).device
    max_new_tokens = max_new_tokens if max_new_tokens is not None else sample.max_new_tokens
    passage_text, question_text = _two_pass_texts(tokenizer, sample.context, sample.question)

    passage_ids = tokenizer(passage_text, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)
    question_ids = tokenizer(question_text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(device)
    prompt_tokens = int(passage_ids.shape[1] + question_ids.shape[1])

    with peak_memory() as mem:
        started = time.perf_counter()
        with attach_cache(model, cache):
            model(passage_ids, past_key_values=cache, use_cache=True,
                  attention_mask=torch.ones_like(passage_ids))
            mask = torch.ones((1, cache.get_seq_length() + question_ids.shape[1]),
                               dtype=torch.long, device=device)
            # `generate()` starting fresh against an already-populated cache -
            # unlike every other generation path in this repo, which starts
            # `generate()` from an empty cache and lets it compress internally
            # over one continuous call - has to be told where the new tokens
            # truly sit. Left to infer it (`modeling_qwen2.py`, Qwen2Model.
            # forward: "if position_ids is None: position_ids = arange(seq_len)
            # + past_key_values.get_seq_length()"), HF derives the new tokens'
            # RoPE position from `get_seq_length()`, which is the compressed
            # SLOT count (e.g. 822 at budget=0.1 on an 8192-token passage), not
            # the true count of tokens actually seen (~8192). That 7000+
            # position error corrupts every subsequent attention computation:
            # confirmed on real hardware, this is what turned every compressed
            # method's output to repetitive garbage ("spam spam spam...")
            # while the uncompressed `full` arm - where slot count and true
            # count are identical - answered correctly. `cache.t_now` is the
            # true position of the last token `update()` actually stored
            # (CLAUDE.md A5 / src/caches/base.py), so the question's first
            # token continues from `t_now + 1`. (Not `cache_position`: that
            # argument was removed from the model's forward signature in this
            # transformers version and `generate()` rejects it outright -
            # `position_ids` is the one HF still derives from `get_seq_length()`
            # and the one whose explicit value generate()'s decode loop
            # continues to increment correctly for every later new token.)
            start_pos = int(cache.t_now) + 1
            position_ids = torch.arange(
                start_pos, start_pos + question_ids.shape[1], device=device
            ).unsqueeze(0)
            output = model.generate(
                question_ids,
                attention_mask=mask,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started

    generated_ids = output[0, question_ids.shape[1]:]
    n_new = int(generated_ids.shape[0])
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = cache.get_stats()
    history = list(getattr(cache, "budget_history", []))
    return {
        "generated_text": text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": n_new,
        "seconds": elapsed,
        "tokens_per_sec": (n_new / elapsed) if elapsed > 0 else 0.0,
        "max_memory_allocated": mem["max_memory_allocated"],
        "memory_allocated_delta": mem["memory_allocated_delta"],
        "cache_stats": stats,
        "budget_used_pct_max": max(history) if history else 100.0,
        "budget_used_pct_final": history[-1] if history else 100.0,
        "conservation_ok": bool(cache.check_conservation()),
        "cache_config": cache.config_dict(),
    }
