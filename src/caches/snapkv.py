"""SnapKV baseline: observation-window scoring, then hard eviction.

Adapted from the reference implementation (FasterDecoding/SnapKV,
`snapkv_utils.py::SnapKVCluster.update_kv`) onto the SR-KV cache interface.
The selection rule below is deliberately a transcription of that routine
rather than a call into `src/scoring.py`:

    attn  = softmax(q[..., -W:, :] @ k^T / sqrt(d))    # causal within the window
    votes = attn[..., :-W].sum(over the W queries)     # score the *past* only
    votes = pool1d(votes, kernel)                      # keep spans, not lone tokens
    keep  = topk(votes, capacity - W)  U  {the last W tokens}

Keeping it independent is the point: `tests/test_cache_correctness.py` asserts
that `SRKVCache(use_clustering=False)` reproduces *these* decisions exactly, so
if the unified class ever drifts away from SnapKV's hard-eviction behaviour the
test fails against an implementation that did not drift with it.

Two documented deviations from the reference:

* GQA. The reference was written for MHA (Llama-2); with grouped queries we
  average the vote of the query heads that share a kv head, then select per kv
  head. `src/scoring.py` uses the same convention, so the comparison is fair.
* Decode-time recompression. The reference compresses the prompt once and then
  lets the cache grow. We re-apply the identical rule whenever the cache
  exceeds its budget during decoding, so that "budget" means the same thing for
  every method in the comparison (and `n_tokens_cached <= budget` holds
  throughout generation, which Phase 2 tests).

No H2O. Reproducing H2O needs cumulative attention over the *full* sequence,
i.e. an eager `[seq_len, seq_len]` attention matrix at prefill: at 8k on a 16GB
T4 that is ~1GB per layer per head-group in fp16 before any activations, which
is precisely the constraint SnapKV's observation window exists to dodge. We
cite published H2O numbers instead of reimplementing it (see README).
"""

from __future__ import annotations

import torch

from ..compat import repeat_kv
from ..scoring import pool_scores
from .base import SRKVCacheBase


class SnapKVCache(SRKVCacheBase):
    """Hard eviction driven by observation-window attention votes."""

    method_name = "snapkv"
    needs_queries = True

    def __init__(
        self,
        *,
        obs_window: int = 32,
        pool_kernel: int = 7,
        pool_type: str = "maxpool",
        n_sink: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obs_window = int(obs_window)
        self.pool_kernel = int(pool_kernel)
        self.pool_type = pool_type
        self.n_sink = int(n_sink)

    def config_dict(self) -> dict:
        cfg = super().config_dict()
        cfg.update(
            obs_window=self.obs_window,
            pool_kernel=self.pool_kernel,
            pool_type=self.pool_type,
            n_sink=self.n_sink,
        )
        return cfg

    # ------------------------------------------------------------------
    def post_attention(self, layer_idx: int, query_states=None, module=None):
        if query_states is not None:
            self._record_queries(layer_idx, query_states, self.obs_window)
        super().post_attention(layer_idx, query_states, module)

    def _compress(self, layer_idx: int, query_states, module) -> None:
        keys, values, positions, weights, centroid_flags = self._slot_state(layer_idx)
        b, h, s, _ = keys.shape
        budget = self.budget_tokens
        if s <= budget or layer_idx not in self.query_buf:
            return

        keep_idx = self.select_indices(
            queries=self.query_buf[layer_idx],
            query_positions=self.query_pos_buf[layer_idx],
            keys=keys,
            key_positions=positions,
            capacity=budget,
            obs_window=self.obs_window,
            n_sink=self.n_sink,
            pool_kernel=self.pool_kernel,
            pool_type=self.pool_type,
            scaling=getattr(module, "scaling", None),
        )
        n_removed = s - keep_idx.shape[-1]
        self._write_slots(
            layer_idx,
            self._gather(keys, keep_idx),
            self._gather(values, keep_idx),
            self._gather(positions, keep_idx),
            self._gather(weights, keep_idx),
            self._gather(centroid_flags, keep_idx),
            n_individuals_removed=n_removed,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def select_indices(
        *,
        queries: torch.Tensor,
        query_positions: torch.Tensor,
        keys: torch.Tensor,
        key_positions: torch.Tensor,
        capacity: int,
        obs_window: int,
        n_sink: int = 0,
        pool_kernel: int = 7,
        pool_type: str = "maxpool",
        scaling: float | None = None,
    ) -> torch.Tensor:
        """SnapKV's selection. Returns kept slot indices, [b, kv_heads, capacity].

        Slots are position-sorted, so the observation window is the final
        `obs_window` slots and the sinks are the first `n_sink`.
        """
        b, n_heads, w, d = queries.shape
        kv_heads, s = keys.shape[1], keys.shape[2]
        n_rep = n_heads // kv_heads
        scaling = scaling if scaling is not None else d**-0.5

        # --- votes from the observation window (the ONLY eager attention) ---
        keys_rep = repeat_kv(keys, n_rep)
        attn = torch.matmul(queries.float(), keys_rep.float().transpose(2, 3)) * scaling
        causal = key_positions.unsqueeze(2) > query_positions.view(1, 1, w, 1)
        if n_rep > 1:
            causal = causal.repeat_interleave(n_rep, dim=1)
        attn = torch.softmax(attn.masked_fill(causal, float("-inf")), dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        votes = attn.sum(dim=2).view(b, kv_heads, n_rep, s).mean(dim=2)  # [b, kv, S]

        # --- candidate region: everything except sinks and the recent window ---
        lo, hi = n_sink, s - obs_window
        n_pick = capacity - obs_window - n_sink
        if n_pick <= 0 or hi <= lo:
            # budget smaller than the protected regions: keep the most recent slots
            tail = torch.arange(s - capacity, s, device=keys.device)
            return tail.view(1, 1, -1).expand(b, kv_heads, capacity)
        n_pick = min(n_pick, hi - lo)

        cand = pool_scores(votes[:, :, lo:hi], pool_kernel, pool_type)
        picked = cand.topk(n_pick, dim=-1).indices + lo

        parts = [picked]
        if n_sink:
            parts.insert(0, torch.arange(lo, device=keys.device).view(1, 1, -1).expand(b, kv_heads, n_sink))
        parts.append(
            torch.arange(hi, s, device=keys.device).view(1, 1, -1).expand(b, kv_heads, obs_window)
        )
        return torch.cat(parts, dim=-1)
