"""SRKVCacheBase - shared bookkeeping for every SR-KV cache variant.

The base class on its own is a **pure passthrough**: it behaves exactly like
`DynamicCache` and evicts nothing. That is the Phase 1 no-op cache and the
uncompressed baseline for every experiment.

What it adds on top of `DynamicCache` is the bookkeeping every compressed
variant needs:

* per-slot **positions** (the true RoPE position each cached slot stands at),
  taken from `cache_kwargs["cache_position"]` so they stay correct even after
  the cache has been compressed and is shorter than the real sequence;
* per-slot **weights** (how many original tokens a slot represents: 1 for a
  real token, >= 2 for a centroid) and a per-slot **is_centroid** flag, which
  together give the `get_stats()` accounting described in CLAUDE.md A4;
* the `post_attention()` hook (CLAUDE.md A2), where subclasses compress.

Stats are read off `(batch 0, kv-head 0)`. Every head holds the same *number*
of slots - that uniformity is required by CLAUDE.md A1 - but which tokens each
head keeps differs, so a single head is the only well-defined place to count.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from ..compat import get_kv, layer_initialized, set_kv


class SRKVCacheBase(DynamicCache):
    """Passthrough cache + slot bookkeeping. Subclasses override `_compress`."""

    method_name = "full"
    #: subclasses that need query states set this so attn_patch knows to pass them
    needs_queries = False

    def __init__(
        self,
        *,
        budget: float = 1.0,
        max_capacity: int | None = None,
        min_budget_tokens: int = 32,
        recompress_slack: int = 0,
        **unused,
    ):
        super().__init__()
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        self.budget = float(budget)
        self.max_capacity = max_capacity
        self.min_budget_tokens = int(min_budget_tokens)
        self.recompress_slack = int(recompress_slack)

        self.budget_tokens: int | None = None
        self._budget_initialized = False
        self.prompt_len: int | None = None
        self.n_tokens_seen = 0
        self.n_tokens_evicted = 0

        self.positions: dict[int, torch.Tensor] = {}
        self.slot_weights: dict[int, torch.Tensor] = {}
        self.is_centroid: dict[int, torch.Tensor] = {}

        self.t_now: float = 0.0
        self._pos_counter = 0
        #: budget_used_pct sampled at every forward pass (Phase 3 checks the
        #: whole trajectory, not just the final value)
        self.budget_history: list[float] = []
        #: rolling observation window of recent query vectors, per layer
        self.query_buf: dict[int, torch.Tensor] = {}
        self.query_pos_buf: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # DynamicCache interface
    # ------------------------------------------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        n_new = key_states.shape[-2]
        b, kv_heads = key_states.shape[0], key_states.shape[1]
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(
                self._pos_counter, self._pos_counter + n_new, device=key_states.device
            )
        new_pos = cache_position.detach().float().view(1, 1, n_new).expand(b, kv_heads, n_new)
        new_w = torch.ones(b, kv_heads, n_new, device=key_states.device, dtype=torch.float32)
        new_c = torch.zeros(b, kv_heads, n_new, device=key_states.device, dtype=torch.bool)

        if layer_idx in self.positions:
            self.positions[layer_idx] = torch.cat([self.positions[layer_idx], new_pos], dim=-1)
            self.slot_weights[layer_idx] = torch.cat([self.slot_weights[layer_idx], new_w], dim=-1)
            self.is_centroid[layer_idx] = torch.cat([self.is_centroid[layer_idx], new_c], dim=-1)
        else:
            self.positions[layer_idx] = new_pos.clone()
            self.slot_weights[layer_idx] = new_w
            self.is_centroid[layer_idx] = new_c

        if layer_idx == 0:
            self.n_tokens_seen += n_new
            self._pos_counter = int(cache_position[-1].item()) + 1
            self.t_now = float(cache_position[-1].item())
            if not self._budget_initialized:
                self._init_budget(n_new)

        return keys, values

    def reset(self):
        super().reset()
        self.positions.clear()
        self.slot_weights.clear()
        self.is_centroid.clear()
        self.budget_tokens = None
        self._budget_initialized = False
        self.prompt_len = None
        self.n_tokens_seen = 0
        self.n_tokens_evicted = 0
        self.t_now = 0.0
        self._pos_counter = 0
        self.budget_history.clear()
        self.query_buf.clear()
        self.query_pos_buf.clear()

    # ------------------------------------------------------------------
    # SR-KV hooks
    # ------------------------------------------------------------------
    def post_attention(self, layer_idx: int, query_states: torch.Tensor | None = None, module=None):
        """Called by `src.attn_patch` right after layer `layer_idx` attends.

        This is where compression happens (CLAUDE.md A2). The base class is a
        passthrough and does nothing.
        """
        if self.should_compress(layer_idx):
            self._compress(layer_idx, query_states, module)
        if layer_idx == 0:
            # sampled *after* compression, so the trajectory reflects what the
            # cache actually holds rather than its pre-eviction high-water mark
            self.budget_history.append(self.get_stats()["budget_used_pct"])

    def should_compress(self, layer_idx: int) -> bool:
        if self.budget_tokens is None or not layer_initialized(self, layer_idx):
            return False
        if self.budget >= 1.0 and self.max_capacity is None:
            return False
        return get_kv(self, layer_idx)[0].shape[-2] > self.budget_tokens + self.recompress_slack

    def _compress(self, layer_idx: int, query_states, module) -> None:
        """No-op in the base class; subclasses evict/merge here."""
        return

    # ------------------------------------------------------------------
    # stats (contract: exactly four keys)
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        cached = self.get_seq_length() if len(self.layers) else 0
        n_centroids = 0
        if 0 in self.is_centroid and self.is_centroid[0].numel():
            n_centroids = int(self.is_centroid[0][0, 0].sum().item())
        denom = self.budget_tokens if self.budget_tokens else max(cached, 1)
        return {
            "n_tokens_cached": int(cached),
            "n_tokens_evicted": int(self.n_tokens_evicted),
            "n_centroids": int(n_centroids),
            "budget_used_pct": float(100.0 * cached / denom),
        }

    def check_conservation(self) -> bool:
        """`n_tokens_seen == n_tokens_evicted + (n_tokens_cached - n_centroids)`."""
        s = self.get_stats()
        return self.n_tokens_seen == s["n_tokens_evicted"] + (s["n_tokens_cached"] - s["n_centroids"])

    def config_dict(self) -> dict:
        """Serialisable description of the policy, recorded in every result file."""
        return {
            "method": self.method_name,
            "budget": self.budget,
            "budget_tokens": self.budget_tokens,
            "max_capacity": self.max_capacity,
        }

    # ------------------------------------------------------------------
    # helpers for subclasses
    # ------------------------------------------------------------------
    def _init_budget(self, prompt_len: int) -> None:
        """Fix the token budget from the prompt length, once, at prefill.

        `budget_tokens` stays None for the uncompressed baseline, so that
        `budget_used_pct` reads 100% rather than a meaningless ratio against a
        budget that does not exist.
        """
        self.prompt_len = prompt_len
        self._budget_initialized = True
        if self.budget >= 1.0 and self.max_capacity is None:
            self.budget_tokens = None
            return
        if self.max_capacity is not None:
            self.budget_tokens = int(self.max_capacity)
        else:
            self.budget_tokens = int(round(self.budget * prompt_len))
        self.budget_tokens = max(self.budget_tokens, self.min_budget_tokens)
        self.budget_tokens = min(self.budget_tokens, prompt_len) if self.budget < 1.0 else self.budget_tokens

    def _record_queries(self, layer_idx: int, query_states: torch.Tensor, window: int) -> None:
        """Keep the last `window` query vectors and their absolute positions.

        During prefill the window is the tail of the prompt; during decode it
        accumulates across steps, which is what lets a compressed cache still
        be re-scored later. Positions are derived from `t_now`, set by the
        layer-0 `update()` of the same forward pass.
        """
        q_len = query_states.shape[2]
        end = self.t_now + 1.0
        pos = torch.arange(end - q_len, end, device=query_states.device, dtype=torch.float32)

        q_new = query_states[:, :, -window:, :].detach()
        pos_new = pos[-window:]
        if layer_idx in self.query_buf and q_new.shape[2] < window:
            q_new = torch.cat([self.query_buf[layer_idx], q_new], dim=2)[:, :, -window:, :]
            pos_new = torch.cat([self.query_pos_buf[layer_idx], pos_new])[-window:]
        self.query_buf[layer_idx] = q_new
        self.query_pos_buf[layer_idx] = pos_new

    def _slot_state(self, layer_idx: int):
        keys, values = get_kv(self, layer_idx)
        return (
            keys,
            values,
            self.positions[layer_idx],
            self.slot_weights[layer_idx],
            self.is_centroid[layer_idx],
        )

    @staticmethod
    def _gather(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather along the sequence dim. tensor [b,h,S(,d)], idx [b,h,n]."""
        if tensor.dim() == 4:
            return tensor.gather(2, idx.unsqueeze(-1).expand(*idx.shape, tensor.shape[-1]))
        return tensor.gather(2, idx)

    def _write_slots(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        weights: torch.Tensor,
        centroid_flags: torch.Tensor,
        n_individuals_removed: int,
    ) -> None:
        """Install a new slot set for `layer_idx`, sorted by RoPE position."""
        order = positions.argsort(dim=-1)
        keys = self._gather(keys, order)
        values = self._gather(values, order)
        positions = positions.gather(2, order)
        weights = weights.gather(2, order)
        centroid_flags = centroid_flags.gather(2, order)

        set_kv(self, layer_idx, keys.contiguous(), values.contiguous())
        self.positions[layer_idx] = positions.contiguous()
        self.slot_weights[layer_idx] = weights.contiguous()
        self.is_centroid[layer_idx] = centroid_flags.contiguous()

        # accounting is done once, on the representative layer
        if layer_idx == 0:
            self.n_tokens_evicted += int(n_individuals_removed)
