"""Evolving conversation state: does SR-KV track fact updates better than baselines?

Motivation (PREREGISTRATION.md, addendum H5). Every prior experiment used tasks where
recency is either neutral (NIAH depth-randomised) or actively harmful (SR-KV below its
own ablations). The evolving-conversation task inverts that: a customer makes an initial
request, then corrects it mid-conversation ("I'm now working from home — ship it there
instead"). The correct answer is the *latest updated state*, not the first-mentioned
state. At the same time, early-turn constraints ("must arrive before Friday") persist
unchanged and must not be lost to a pure recency window.

This combination is what full SR-KV is actually designed for:
- Recency weighting protects the most-recent correction.
- Centroid merging retains older-but-still-relevant context in blurred but usable form.

A recency-only method (`recency_hard_evict`) should handle updates correctly but may
discard early constraints; a merging-only method (`centroid_merge`) may preserve
constraints but lose the correction under its soft-averaging. Full SR-KV is the only
method with both properties.

Four question variants:
- ``current_request``  — what is the customer's *current* request (after all corrections)?
- ``accepted_solution``— which proposed fix was accepted (not the one that was rejected)?
- ``unresolved_issue`` — which issue remains open (not the one that was resolved)?
- ``old_constraint``   — what constraint from an early turn still applies?

The first two are answered by recency; the last two require retaining early context.
This split is the diagnostic: if SR-KV only wins on ``current_request`` it is the
recency component, not merging, driving any advantage.

Two-pass measure: same discipline as ``eval/gist_mcq.py``. The conversation is
compressed during a passage-only forward pass before the question exists, so the
question cannot act as an observation-window hint.
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
VARIANTS = ("current_request", "accepted_solution", "unresolved_issue", "old_constraint")

# ---------------------------------------------------------------------------
# Conversation building blocks
# ---------------------------------------------------------------------------

# Products whose lifecycle gives a natural reason for a delivery-address update.
PRODUCTS = [
    "a replacement laptop", "a warranty repair unit", "a loaner device",
    "a refund cheque", "a spare battery pack", "a replacement keyboard",
    "an upgraded router", "a new power adapter",
]

LOCATIONS = [
    ("office at 12 Birch Lane", "flat at 7 Maple Court"),
    ("desk at the downtown branch", "home address in the suburbs"),
    ("company HQ reception", "the satellite office on Park Road"),
    ("registered business address", "personal address on file"),
    ("colleague's desk", "own home address"),
]

SOLUTIONS = [
    ("send a replacement unit", "issue a partial refund"),
    ("upgrade to the next tier", "apply a discount code"),
    ("prioritise express shipping", "schedule a callback"),
    ("escalate to a senior engineer", "provide a self-repair kit"),
    ("offer a same-day swap", "extend the warranty period"),
]

ISSUES = [
    ("wrong item delivered", "billing charge not reversed"),
    ("device powers off randomly", "account login not working"),
    ("shipment delay", "incorrect invoice total"),
    ("product arrived damaged", "missing accessory in box"),
    ("data not transferred", "old device not collected"),
]

CONSTRAINTS = [
    "The replacement must arrive before Friday.",
    "Delivery must go to a ground-floor address only.",
    "The package requires a signature on receipt.",
    "The item must not be left with a neighbour.",
    "A morning delivery window (before noon) is required.",
    "The customer has prepaid for express shipping.",
    "The item must be delivered in its original packaging.",
    "The customer has a limited availability window this week.",
]

# Templates for the four narrative stages of a conversation --------------

_INITIAL_REQUEST_TMPL = (
    "The customer called to request that {product} be shipped to their {loc_a}. "
    "The agent confirmed the address and raised a dispatch ticket."
)

_CORRECTION_TMPL = (
    "The customer called back to update their delivery address. "
    "They explained they are now working from home and asked for the shipment "
    "to be redirected to their {loc_b} instead. The agent updated the ticket accordingly."
)

_REJECTED_SOLUTION_TMPL = (
    "The agent proposed to {solution_b} as an interim measure, "
    "but the customer declined and asked for a different resolution."
)

_ACCEPTED_SOLUTION_TMPL = (
    "After further discussion, the agent offered to {solution_a}. "
    "The customer agreed and confirmed this was acceptable."
)

_RESOLVED_ISSUE_TMPL = (
    "Earlier in the conversation the customer had also raised a concern about "
    "{issue_b}. The agent confirmed this had already been resolved and closed the ticket."
)

_OPEN_ISSUE_TMPL = (
    "The customer mentioned that {issue_a} was still outstanding and had not been "
    "addressed. The agent noted this and promised to follow up."
)

# Distractors — plausible but irrelevant chatter ---------------------------
_DISTRACTORS = [
    "The agent placed the customer on a brief hold to check the account notes.",
    "There was a short delay while the system updated the customer's profile.",
    "The agent confirmed the customer's account number for verification purposes.",
    "The customer was transferred to the appropriate department to continue.",
    "A standard satisfaction survey was offered at the end of the interaction.",
    "The agent apologised for the wait and thanked the customer for their patience.",
    "The customer was reminded of the company's returns policy for reference.",
    "A reference number was generated and shared with the customer by email.",
    "The agent noted that processing times may vary during peak periods.",
    "The customer confirmed their preferred contact number for callbacks.",
]

# Depths for each planted event in the conversation (% of context) ----------
# The correction and accepted-solution are placed late (high recency value):
# a pure recency window should protect them.
# The old constraint and unresolved issue are placed early (low recency value):
# a pure recent-window method at a tight budget will likely evict them.
_DEPTHS = {
    "constraint": 5,       # early — must be retained despite later conversation
    "initial_request": 15, # early
    "open_issue": 25,      # early-mid
    "distractor_a": 35,
    "distractor_b": 45,
    "rejected_solution": 55,
    "resolved_issue": 65,
    "distractor_c": 72,
    "accepted_solution": 80,  # late — recency should protect
    "correction": 92,         # very late — this IS the answer to current_request
}


@dataclass
class ConvSample:
    """One conv_state cell instance."""

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
        return f"conv/{self.variant}/ctx{self.context_len}/s{self.sample_idx}"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _splice(body_ids: list[int], inserts: list[tuple[int, list[int]]]) -> list[int]:
    """Insert each (cut_index, ids) into body_ids, left to right.

    Same helper as gist_mcq._splice – cut indices are computed against the
    un-spliced body so inserting in increasing order never shifts a later cut.
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


# ---------------------------------------------------------------------------
# Sample builders, one per variant
# ---------------------------------------------------------------------------

def _build_current_request(tokenizer, rng: random.Random, hay_ids, context_len):
    """Answer: loc_b (the corrected address). Recency should protect the correction."""
    product = rng.choice(PRODUCTS)
    loc_a, loc_b = rng.choice(LOCATIONS)

    # Distractors as noise for the other variants
    constraint = rng.choice(CONSTRAINTS)
    solution_a, solution_b = rng.choice(SOLUTIONS)
    issue_a, issue_b = rng.choice(ISSUES)

    fragments = [
        (_DEPTHS["constraint"],
         constraint),
        (_DEPTHS["initial_request"],
         _INITIAL_REQUEST_TMPL.format(product=product, loc_a=loc_a)),
        (_DEPTHS["open_issue"],
         _OPEN_ISSUE_TMPL.format(issue_a=issue_a)),
        (_DEPTHS["distractor_a"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["distractor_b"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["rejected_solution"],
         _REJECTED_SOLUTION_TMPL.format(solution_b=solution_b)),
        (_DEPTHS["resolved_issue"],
         _RESOLVED_ISSUE_TMPL.format(issue_b=issue_b)),
        (_DEPTHS["distractor_c"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["accepted_solution"],
         _ACCEPTED_SOLUTION_TMPL.format(solution_a=solution_a)),
        (_DEPTHS["correction"],
         _CORRECTION_TMPL.format(loc_b=loc_b)),
    ]

    context = _make_context(tokenizer, hay_ids, context_len, fragments)

    # MCQ: loc_b is correct; fill with plausible wrong addresses
    wrong_locs = [a for a, b in LOCATIONS if a != loc_a and b != loc_b][:2]
    wrong_locs.append(loc_a)  # the original (superseded) address is a strong distractor
    option_pool = sorted({loc_b} | set(wrong_locs[:3]))[:4]
    # Ensure exactly 4 distinct options
    all_locs = [l for pair in LOCATIONS for l in pair if l != loc_b and l != loc_a]
    while len(option_pool) < 4:
        candidate = all_locs.pop(0)
        if candidate not in option_pool:
            option_pool.append(candidate)
    option_pool = option_pool[:4]
    rng.shuffle(option_pool)
    answer_letter = LETTERS[option_pool.index(loc_b)]

    question = (
        f"Based on the conversation above, where should {product} be delivered now? "
        "Answer with a single letter and nothing else.\n"
        f"{_options_block(option_pool)}\nAnswer:"
    )
    return context, question, option_pool, answer_letter


def _build_accepted_solution(tokenizer, rng: random.Random, hay_ids, context_len):
    """Answer: solution_a (the accepted one). Late-placed; recency should help."""
    product = rng.choice(PRODUCTS)
    loc_a, loc_b = rng.choice(LOCATIONS)
    constraint = rng.choice(CONSTRAINTS)
    solution_a, solution_b = rng.choice(SOLUTIONS)
    issue_a, issue_b = rng.choice(ISSUES)

    fragments = [
        (_DEPTHS["constraint"], constraint),
        (_DEPTHS["initial_request"],
         _INITIAL_REQUEST_TMPL.format(product=product, loc_a=loc_a)),
        (_DEPTHS["open_issue"],
         _OPEN_ISSUE_TMPL.format(issue_a=issue_a)),
        (_DEPTHS["distractor_a"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["distractor_b"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["rejected_solution"],
         _REJECTED_SOLUTION_TMPL.format(solution_b=solution_b)),
        (_DEPTHS["resolved_issue"],
         _RESOLVED_ISSUE_TMPL.format(issue_b=issue_b)),
        (_DEPTHS["distractor_c"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["accepted_solution"],
         _ACCEPTED_SOLUTION_TMPL.format(solution_a=solution_a)),
        (_DEPTHS["correction"],
         _CORRECTION_TMPL.format(loc_b=loc_b)),
    ]

    context = _make_context(tokenizer, hay_ids, context_len, fragments)

    # solution_b is the rejected one — strong distractor
    other_solutions = [s for s, _ in SOLUTIONS if s != solution_a and s != solution_b][:2]
    option_pool = [solution_a, solution_b] + other_solutions[:2]
    rng.shuffle(option_pool)
    answer_letter = LETTERS[option_pool.index(solution_a)]

    question = (
        "Based on the conversation above, which resolution did the customer agree to? "
        "Answer with a single letter and nothing else.\n"
        f"{_options_block(option_pool)}\nAnswer:"
    )
    return context, question, option_pool, answer_letter


def _build_unresolved_issue(tokenizer, rng: random.Random, hay_ids, context_len):
    """Answer: issue_a (still open). Early-placed; a tight recency window may evict it."""
    product = rng.choice(PRODUCTS)
    loc_a, loc_b = rng.choice(LOCATIONS)
    constraint = rng.choice(CONSTRAINTS)
    solution_a, solution_b = rng.choice(SOLUTIONS)
    issue_a, issue_b = rng.choice(ISSUES)

    fragments = [
        (_DEPTHS["constraint"], constraint),
        (_DEPTHS["initial_request"],
         _INITIAL_REQUEST_TMPL.format(product=product, loc_a=loc_a)),
        (_DEPTHS["open_issue"],
         _OPEN_ISSUE_TMPL.format(issue_a=issue_a)),
        (_DEPTHS["distractor_a"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["distractor_b"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["rejected_solution"],
         _REJECTED_SOLUTION_TMPL.format(solution_b=solution_b)),
        (_DEPTHS["resolved_issue"],
         _RESOLVED_ISSUE_TMPL.format(issue_b=issue_b)),
        (_DEPTHS["distractor_c"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["accepted_solution"],
         _ACCEPTED_SOLUTION_TMPL.format(solution_a=solution_a)),
        (_DEPTHS["correction"],
         _CORRECTION_TMPL.format(loc_b=loc_b)),
    ]

    context = _make_context(tokenizer, hay_ids, context_len, fragments)

    # issue_b is the resolved one — strong distractor
    other_issues = [i for i, _ in ISSUES if i != issue_a and i != issue_b][:2]
    option_pool = [issue_a, issue_b] + other_issues[:2]
    rng.shuffle(option_pool)
    answer_letter = LETTERS[option_pool.index(issue_a)]

    question = (
        "Based on the conversation above, which customer concern is still unresolved? "
        "Answer with a single letter and nothing else.\n"
        f"{_options_block(option_pool)}\nAnswer:"
    )
    return context, question, option_pool, answer_letter


def _build_old_constraint(tokenizer, rng: random.Random, hay_ids, context_len):
    """Answer: the constraint planted at depth 5. Must be retained despite later chatter."""
    product = rng.choice(PRODUCTS)
    loc_a, loc_b = rng.choice(LOCATIONS)
    constraint = rng.choice(CONSTRAINTS)
    solution_a, solution_b = rng.choice(SOLUTIONS)
    issue_a, issue_b = rng.choice(ISSUES)

    fragments = [
        (_DEPTHS["constraint"], constraint),
        (_DEPTHS["initial_request"],
         _INITIAL_REQUEST_TMPL.format(product=product, loc_a=loc_a)),
        (_DEPTHS["open_issue"],
         _OPEN_ISSUE_TMPL.format(issue_a=issue_a)),
        (_DEPTHS["distractor_a"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["distractor_b"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["rejected_solution"],
         _REJECTED_SOLUTION_TMPL.format(solution_b=solution_b)),
        (_DEPTHS["resolved_issue"],
         _RESOLVED_ISSUE_TMPL.format(issue_b=issue_b)),
        (_DEPTHS["distractor_c"], rng.choice(_DISTRACTORS)),
        (_DEPTHS["accepted_solution"],
         _ACCEPTED_SOLUTION_TMPL.format(solution_a=solution_a)),
        (_DEPTHS["correction"],
         _CORRECTION_TMPL.format(loc_b=loc_b)),
    ]

    context = _make_context(tokenizer, hay_ids, context_len, fragments)

    # Other constraints are the distractors
    wrong_constraints = [c for c in CONSTRAINTS if c != constraint]
    rng.shuffle(wrong_constraints)
    option_pool = [constraint] + wrong_constraints[:3]
    rng.shuffle(option_pool)
    answer_letter = LETTERS[option_pool.index(constraint)]

    question = (
        "Based on the conversation above, which delivery constraint did the customer "
        "specify at the start? Answer with a single letter and nothing else.\n"
        f"{_options_block(option_pool)}\nAnswer:"
    )
    return context, question, option_pool, answer_letter


_BUILDERS = {
    "current_request": _build_current_request,
    "accepted_solution": _build_accepted_solution,
    "unresolved_issue": _build_unresolved_issue,
    "old_constraint": _build_old_constraint,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_samples(
    tokenizer,
    *,
    context_lengths,
    variants=VARIANTS,
    n_samples: int = 50,
    corpus: str = "pg",
    seed: int = 1234,
) -> list[ConvSample]:
    """Materialise every (variant, context_len, sample) cell.

    ``corpus="pg"`` (Paul Graham essays, real prose) is the default: same
    reasoning as ``gist_mcq.build_samples`` — synthetic filler is too uniform
    for importance scoring to have to work, and the experiment measures
    eviction policy, not whether the model can read.

    One shared rng advanced in a fixed nested order ensures that resuming a run
    (or adding a method) sees byte-identical samples — same discipline as NIAH
    and gist_mcq.
    """
    for variant in variants:
        if variant not in _BUILDERS:
            raise ValueError(
                f"unknown conv_state variant {variant!r}; known: {sorted(_BUILDERS)}"
            )

    rng = random.Random(seed)
    max_ctx = max(context_lengths)
    hay_ids = _haystack_ids(tokenizer, rng, corpus, max_ctx + 512)

    samples: list[ConvSample] = []
    for variant in variants:
        builder = _BUILDERS[variant]
        for context_len in context_lengths:
            for sample_idx in range(n_samples):
                context, question, options, answer_letter = builder(
                    tokenizer, rng, hay_ids, context_len
                )
                samples.append(
                    ConvSample(
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_LETTER_PAREN_RE = re.compile(r"([ABCD])\)")
_AFTER_ANSWER_RE = re.compile(r"\bANSWER\s*:?\s*([ABCD])\b")
_LEADING_LETTER_RE = re.compile(r"^([ABCD])\b")


def score(sample: ConvSample, generated_text: str) -> float:
    """1.0 if the generation's MCQ choice matches; 0.0 otherwise."""
    text = generated_text.strip().upper()
    for pattern in (_LETTER_PAREN_RE, _AFTER_ANSWER_RE, _LEADING_LETTER_RE):
        match = pattern.search(text)
        if match:
            return 1.0 if match.group(1) == sample.answer_letter else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Two-pass measure
# ---------------------------------------------------------------------------

def _two_pass_texts(tokenizer, context: str, question: str) -> tuple[str, str]:
    """Split (passage + question) at the exact boundary, same as gist_mcq."""
    if getattr(tokenizer, "chat_template", None):
        user = f"{context}\n\n{question}"
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        idx = wrapped.index(context) + len(context)
        return wrapped[:idx], wrapped[idx:]
    return context, "\n\n" + question


@torch.no_grad()
def measure(model, tokenizer, sample: ConvSample, cache, *,
            max_new_tokens: int | None = None, device=None) -> dict:
    """Two forward passes: compress conversation, then generate answer.

    Identical in structure to ``gist_mcq.measure``.  The conversation is
    compressed during the first (passage-only) forward pass before the question
    exists, so the question cannot act as an observation-window hint for the
    compression policy.  This is the same two-pass discipline introduced to fix
    the single-pass smoke test that saturated at 1.000 for every method.
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
