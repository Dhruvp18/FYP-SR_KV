"""StreamingLLM baseline: attention sinks + a sliding recent window.

Adapted from the reference implementation (mit-han-lab/streaming-llm,
`StartRecentKVCache`) onto the SR-KV cache interface. The policy is purely
structural - no scores, no queries - which is why it stays a separate class
instead of being folded into `SRKVCache`'s flags (see CLAUDE.md).

One deliberate difference from the reference: StreamingLLM re-positions the
kept tokens to occupy contiguous RoPE positions 0..n-1. We keep each token's
*original* position, because HuggingFace derives `position_ids` for newly
generated tokens from the attention mask (i.e. absolute), and rewriting cached
positions to be contiguous while new tokens keep absolute positions would put
the two on different scales. Keeping original positions is the consistent
choice for every method in this repo, so the comparison stays apples-to-apples.
"""

from __future__ import annotations

import torch

from .base import SRKVCacheBase


class StreamingLLMCache(SRKVCacheBase):
    """Keep the first `n_sink` tokens and the most recent `budget - n_sink`."""

    method_name = "streaming_llm"
    needs_queries = False

    def __init__(self, *, n_sink: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_sink = int(n_sink)

    def config_dict(self) -> dict:
        cfg = super().config_dict()
        cfg["n_sink"] = self.n_sink
        return cfg

    def _compress(self, layer_idx: int, query_states, module) -> None:
        keys, values, positions, weights, centroid_flags = self._slot_state(layer_idx)
        b, h, s, _ = keys.shape
        budget = self.budget_tokens
        if s <= budget:
            return

        n_sink = min(self.n_sink, budget)
        n_recent = budget - n_sink
        # slots are always position-sorted, so "first" and "last" are literal
        keep = torch.cat(
            [
                torch.arange(n_sink, device=keys.device),
                torch.arange(s - n_recent, s, device=keys.device),
            ]
        )
        idx = keep.view(1, 1, -1).expand(b, h, budget)

        dropped_individuals = int((~centroid_flags[0, 0]).sum().item()) - int(
            (~self._gather(centroid_flags, idx)[0, 0]).sum().item()
        )
        self._write_slots(
            layer_idx,
            self._gather(keys, idx),
            self._gather(values, idx),
            self._gather(positions, idx),
            self._gather(weights, idx),
            self._gather(centroid_flags, idx),
            n_individuals_removed=dropped_individuals,
        )
