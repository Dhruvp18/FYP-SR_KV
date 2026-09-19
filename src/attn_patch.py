"""Attaching an SR-KV cache to a HuggingFace model.

Why a custom attention implementation at all: `Cache.update()` runs *before*
attention and therefore never sees `query_states`, but SnapKV-style scoring is
defined in terms of what the most recent queries attend to. Registering an
attention implementation gives us a hook that runs *after* the layer's
attention output has been computed, with the queries in hand (CLAUDE.md A2).

The registered function does not implement attention itself - it delegates to
HuggingFace's own SDPA path and then calls `cache.post_attention(...)`. So the
model's numerics are stock SDPA; the only thing we add is the compression step.
There is no full-sequence eager attention here (CLAUDE.md A3).
"""

from __future__ import annotations

import contextlib

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

SRKV_ATTN_NAME = "srkv"

#: the cache currently attached to a model. Generation is single-threaded and
#: batch-1 (CLAUDE.md A6), so a module-level slot is sufficient and keeps the
#: attention function free of per-model state.
_ACTIVE_CACHE = None


def srkv_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
):
    """Stock SDPA attention, plus the post-attention compression hook."""
    attn_output, attn_weights = sdpa_attention_forward(
        module, query, key, value, attention_mask, **kwargs
    )

    cache = _ACTIVE_CACHE
    if cache is not None and hasattr(cache, "post_attention"):
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None:
            with torch.no_grad():
                cache.post_attention(layer_idx, query.detach(), module=module)

    return attn_output, attn_weights


def register_srkv_attention() -> None:
    """Make `attn_implementation="srkv"` a valid choice. Idempotent.

    Registering the attention function is not sufficient. Transformers builds
    the causal mask through a *separate* registry keyed by the same
    implementation name, and it ships entries only for eager/sdpa/flash/flex.
    An unregistered name yields **no mask at all**, and
    `sdpa_attention_forward` then falls back to `is_causal = q_len > 1`.

    For a prefill that is harmless, because q_len == kv_len and torch's
    top-left and bottom-right causal alignments coincide. For a multi-token
    forward against an already-populated cache it is silently catastrophic:
    torch aligns a non-square causal mask to the TOP LEFT, so new token 0
    attends to cache slot 0, token 1 to slots 0-1, and the entire cached
    prefix becomes invisible. Measured on Qwen2.5-0.5B over Paul Graham prose
    with an *uncompressed* cache: nll 2.75 (ppl 15.6) correct, nll 6.47
    (ppl 646) with no mask registered - and every method degrades by the same
    amount, so the comparison still looks orderly and means nothing.

    Borrowing sdpa's mask function is exactly right, because
    `srkv_attention_forward` delegates to sdpa: the mask it needs is the mask
    sdpa would have got.
    """
    if SRKV_ATTN_NAME not in ALL_ATTENTION_FUNCTIONS:
        AttentionInterface.register(SRKV_ATTN_NAME, srkv_attention_forward)
    if SRKV_ATTN_NAME not in ALL_MASK_ATTENTION_FUNCTIONS:
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            SRKV_ATTN_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
        )


def _iter_configs(model):
    seen = set()
    for cfg in [model.config, getattr(model.config, "text_config", None)]:
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            yield cfg
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            yield cfg


@contextlib.contextmanager
def attach_cache(model, cache):
    """Route `model`'s attention through SR-KV for the duration of the block.

    Restores the model's original `_attn_implementation` on exit, so a single
    loaded model can be reused across methods inside one process (which is how
    the sweep in Phase 6 keeps model-loading cost down).
    """
    global _ACTIVE_CACHE
    register_srkv_attention()

    configs = list(_iter_configs(model))
    previous = [(cfg, getattr(cfg, "_attn_implementation", None)) for cfg in configs]
    prev_cache, _ACTIVE_CACHE = _ACTIVE_CACHE, cache
    try:
        for cfg in configs:
            cfg._attn_implementation = SRKV_ATTN_NAME
        yield cache
    finally:
        _ACTIVE_CACHE = prev_cache
        for cfg, impl in previous:
            if impl is not None:
                cfg._attn_implementation = impl


def active_cache():
    """The cache currently attached (used by tests)."""
    return _ACTIVE_CACHE
