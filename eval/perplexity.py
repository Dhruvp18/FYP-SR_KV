"""Perplexity of a held-out continuation, conditioned on a compressed cache.

Phase 8 / hypothesis H1 (see PREREGISTRATION.md). Every metric used before
this - NIAH exact match, LongBench qa_f1 and rouge_l - scores n-gram overlap,
and a centroid is an attention-weighted average that cannot reproduce an exact
token. Those metrics therefore cannot credit whatever a centroid does retain,
and on ROUGE they actively penalise it.

Perplexity is the standard KV-compression metric (StreamingLLM reports it) and
scores *distributional* fidelity instead, which is the thing centroids could
plausibly preserve. If clustering carries any information at all, this is
where it shows; if it does not show here either, the null is much stronger.

The measurement is deliberately the one that tests the cache rather than the
model: prefill a long prefix so the cache compresses to budget, then score a
held-out continuation conditioned on that compressed state. A method that
throws away something the continuation needed pays for it here.

No eviction logic lives here - this module builds samples and computes a loss;
the cache does the compressing (CLAUDE.md "Forbidden").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from src.attn_patch import attach_cache

#: tokens of held-out continuation scored after the prefix is compressed.
#: Long enough for a stable estimate, short enough that decode-time
#: recompression is not the dominant effect being measured.
CONTINUATION_TOKENS = 256


@dataclass
class PerplexitySample:
    context_len: int
    sample_idx: int
    prefix_ids: list[int] = field(repr=False)
    continuation_ids: list[int] = field(repr=False)

    @property
    def task_id(self) -> str:
        return f"ppl/ctx{self.context_len}/s{self.sample_idx}"


def build_samples(
    tokenizer,
    *,
    context_lengths,
    n_samples: int = 50,
    continuation_tokens: int = CONTINUATION_TOKENS,
    seed: int = 1234,
) -> list[PerplexitySample]:
    """Non-overlapping (prefix, continuation) windows from real English text.

    Real text, not the NIAH filler: filler is synthetic and highly repetitive,
    so its perplexity is dominated by memorised boilerplate and would flatter
    every method equally. The Paul Graham essays are the corpus the NIAH path
    already knows how to fetch.

    Windows never overlap, so samples are independent and the paired bootstrap
    in the analysis is valid.
    """
    from eval.niah import _load_pg_text

    longest = max(context_lengths)
    need_tokens = n_samples * (longest + continuation_tokens) + 64
    text = _load_pg_text(need_tokens * 8)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    samples: list[PerplexitySample] = []
    for context_len in sorted(context_lengths):
        stride = context_len + continuation_tokens
        available = len(ids) // stride
        if available < n_samples:
            raise ValueError(
                f"corpus gives {available} non-overlapping windows at "
                f"ctx={context_len}, need {n_samples}; raise the multiplier in "
                "build_samples"
            )
        for i in range(n_samples):
            start = i * stride
            samples.append(
                PerplexitySample(
                    context_len=context_len,
                    sample_idx=i,
                    prefix_ids=ids[start:start + context_len],
                    continuation_ids=ids[start + context_len:start + stride],
                )
            )
    return samples


@torch.no_grad()
def measure(model, sample: PerplexitySample, cache, *, device=None) -> dict:
    """Perplexity of the continuation given a cache compressed on the prefix.

    Two forward passes on purpose. The first prefills the prefix and lets the
    cache compress to budget; the second scores the continuation against that
    compressed state. Scoring in one pass would let the continuation's own
    tokens sit uncompressed in the cache and hide exactly the loss being
    measured.
    """
    device = device or next(model.parameters()).device
    prefix = torch.tensor([sample.prefix_ids], device=device)
    continuation = torch.tensor([sample.continuation_ids], device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # attach_cache is what routes attention through the SR-KV implementation
    # that calls cache.post_attention(), and post_attention is where all
    # compression happens (CLAUDE.md A2). Calling model(...) outside this
    # block runs plain SDPA, the cache never compresses, and every method
    # returns byte-identical perplexity - which is exactly what the first
    # smoke test showed (evicted=0, centroids=0, nll equal to 15 decimals).
    # The masks span the cache as well as the new tokens. They are belt and
    # braces - src.attn_patch registers a mask function for the "srkv"
    # implementation, which is what actually makes the causal mask correct
    # here - but passing them explicitly costs nothing and documents that this
    # is the one place in the project that runs a multi-token forward against
    # an already-populated cache.
    with attach_cache(model, cache):
        out = model(prefix, past_key_values=cache, use_cache=True,
                    attention_mask=torch.ones_like(prefix))
        # last prefix logit predicts the first continuation token, so it is
        # part of the score and must not be dropped.
        first_logits = out.logits[:, -1:, :]

        mask = torch.ones(
            (1, cache.get_seq_length() + continuation.shape[1]),
            dtype=torch.long, device=device,
        )
        out = model(continuation, past_key_values=cache, use_cache=True,
                    attention_mask=mask)
        logits = torch.cat([first_logits, out.logits[:, :-1, :]], dim=1).float()

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        continuation.reshape(-1),
        reduction="mean",
    )
    nll = float(loss)
    stats = cache.get_stats()
    return {
        "nll": nll,
        "cache_stats": stats,
        "budget_used_pct_max": stats["budget_used_pct"],
        "prompt_tokens": int(prefix.numel()),
        "perplexity": float(math.exp(min(nll, 100.0))),
        "scored_tokens": int(continuation.numel()),
        "max_memory_allocated": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }


def score(sample: PerplexitySample, result: dict) -> float:
    """Lower perplexity is better, so report negative NLL as the 'accuracy'.

    The records schema fixes `accuracy` as the comparable field, and every
    gate and figure reads it. Storing raw perplexity there would silently
    invert the direction of every comparison ("higher is better") and mix a
    1-to-infinity scale into plots whose axes assume 0-1. Negative NLL keeps
    "larger is better" true, and `perplexity` is recorded alongside as the
    quantity to actually report.
    """
    return -result["nll"]
