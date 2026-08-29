"""SRKVCache - the one class behind three of the four experimental conditions.

    use_recency  use_clustering   condition
    -----------  --------------   ---------------------------------------
    False        False            SnapKV-style hard eviction
    False        True             centroid-merge baseline (no recency term)
    True         True             SR-KV (full)

There is deliberately no separate `CentroidMergeCache`. The recency ablation
is only trustworthy if flipping `use_recency` is the *only* thing that differs
between two runs; two independently written classes could differ in a dozen
incidental ways and the difference in accuracy would no longer isolate the
recency term. This is the design constraint the whole project rests on, so if
a future change starts to grow a second implementation of any of these three
conditions, that change is wrong (see CLAUDE.md).

Per compression step, on the candidate region (everything except the `n_sink`
oldest slots and the `obs_window` most recent ones):

    importance = alpha * windowed_attention + beta * exp(-lam * age)
    keep the top `n_pick` outright
    the remainder is either dropped (use_clustering=False) or clustered into
    exactly `n_centroids` attention-weighted centroids, each re-rotated to a
    RoPE position chosen by `rope_position_mode`

The compressed length is `n_sink + n_pick + n_centroids + obs_window`, which is
exactly `budget_tokens` and identical for every layer and head - the uniformity
CLAUDE.md A1 requires.

Note that decoding needs no special case. Once the cache sits at its budget,
each new token makes the candidate remainder one slot larger than the centroid
target, so the same code path merges that one extra slot into the summary
space, incrementally.
"""

from __future__ import annotations

import torch

from ..clustering import cluster_and_merge
from ..rope_positions import POSITION_MODES, RopeHelper
from ..scoring import (
    ScoringConfig,
    combined_importance,
    compute_attention_scores,
    normalize_scores,
    pool_scores,
    recency_decay,
)
from .base import SRKVCacheBase

#: names used in results files, keyed by (use_recency, use_clustering)
CONDITION_NAMES = {
    (False, False): "snapkv_unified",
    (False, True): "centroid_merge",
    (True, False): "recency_hard_evict",
    (True, True): "sr_kv",
}


class SRKVCache(SRKVCacheBase):
    """Scored eviction with optional recency term and optional centroid merge."""

    needs_queries = True

    def __init__(
        self,
        *,
        use_recency: bool = True,
        use_clustering: bool = True,
        rope_position_mode: str = "attn_weighted",
        scoring: ScoringConfig | None = None,
        rope_helper: RopeHelper | None = None,
        n_centroids: int | None = None,
        centroid_frac: float = 0.125,
        cluster_mode: str = "kmeans",
        recluster_centroids: bool = True,
        kmeans_iters: int = 4,
        n_sink: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if rope_position_mode not in POSITION_MODES:
            raise ValueError(f"rope_position_mode must be one of {POSITION_MODES}")
        self.use_recency = bool(use_recency)
        self.use_clustering = bool(use_clustering)
        self.rope_position_mode = rope_position_mode
        self.scoring = scoring or ScoringConfig()
        self.rope_helper = rope_helper
        self.n_centroids_cfg = n_centroids
        self.centroid_frac = float(centroid_frac)
        self.cluster_mode = cluster_mode
        self.recluster_centroids = bool(recluster_centroids)
        self.kmeans_iters = int(kmeans_iters)
        self.n_sink = int(n_sink)

        if self.use_clustering and self.rope_helper is None:
            raise ValueError(
                "use_clustering=True needs a RopeHelper so centroids can be un-rotated "
                "and re-rotated. Build the cache with make_cache(...), which passes "
                "RopeHelper.from_model(model) for you."
            )

    @property
    def method_name(self) -> str:  # type: ignore[override]
        return CONDITION_NAMES[(self.use_recency, self.use_clustering)]

    def config_dict(self) -> dict:
        cfg = super().config_dict()
        cfg.update(
            use_recency=self.use_recency,
            use_clustering=self.use_clustering,
            rope_position_mode=self.rope_position_mode,
            cluster_mode=self.cluster_mode,
            recluster_centroids=self.recluster_centroids,
            n_centroids_cfg=self.n_centroids_cfg,
            centroid_frac=self.centroid_frac,
            n_sink=self.n_sink,
            **{f"scoring_{k}": v for k, v in self.scoring.to_dict().items()},
        )
        return cfg

    # ------------------------------------------------------------------
    def post_attention(self, layer_idx: int, query_states=None, module=None):
        if query_states is not None:
            self._record_queries(layer_idx, query_states, self.scoring.obs_window)
        super().post_attention(layer_idx, query_states, module)

    # ------------------------------------------------------------------
    def _region_sizes(self, n_slots: int) -> tuple[int, int, int, int]:
        """(n_sink, obs_window, n_pick, n_centroids) for a cache of `n_slots`.

        Depends only on `n_slots` and the configuration, never on the data, so
        every layer and head lands on the same compressed length.
        """
        budget = self.budget_tokens
        window = min(self.scoring.obs_window, max(1, budget // 2))
        sink = min(self.n_sink, max(0, budget // 8))

        n_centroids = 0
        if self.use_clustering:
            n_centroids = (
                self.n_centroids_cfg
                if self.n_centroids_cfg is not None
                else max(1, int(round(self.centroid_frac * budget)))
            )
            n_centroids = min(n_centroids, max(0, budget - window - sink - 1))
            # the remainder to cluster is (n_slots - budget) + n_centroids slots,
            # which is always >= n_centroids, so k <= N always holds
            n_centroids = max(n_centroids, 0)

        n_pick = budget - window - sink - n_centroids
        n_pick = max(n_pick, 0)
        n_pick = min(n_pick, max(0, n_slots - window - sink))
        return sink, window, n_pick, n_centroids

    def _compress(self, layer_idx: int, query_states, module) -> None:
        keys, values, positions, weights, flags = self._slot_state(layer_idx)
        b, h, n_slots, d = keys.shape
        budget = self.budget_tokens
        if n_slots <= budget or layer_idx not in self.query_buf:
            return

        sink, window, n_pick, n_centroids = self._region_sizes(n_slots)
        lo, hi = sink, n_slots - window
        if hi <= lo or n_pick <= 0:
            # budget is smaller than the protected regions: keep the newest slots
            tail = torch.arange(n_slots - budget, n_slots, device=keys.device)
            idx = tail.view(1, 1, -1).expand(b, h, budget)
            self._write_slots(
                layer_idx,
                self._gather(keys, idx),
                self._gather(values, idx),
                self._gather(positions, idx),
                self._gather(weights, idx),
                self._gather(flags, idx),
                n_individuals_removed=int((~flags[0, 0]).sum() - (~self._gather(flags, idx)[0, 0]).sum()),
            )
            return

        # -- score the candidate region ---------------------------------
        attn_raw = compute_attention_scores(
            self.query_buf[layer_idx],
            keys,
            self.query_pos_buf[layer_idx],
            positions,
            scaling=getattr(module, "scaling", None),
        )
        cand_attn = pool_scores(attn_raw[:, :, lo:hi], self.scoring.pool_kernel, self.scoring.pool_type)
        if self.scoring.normalize_attn:
            cand_attn = normalize_scores(cand_attn)
        decay = recency_decay(
            positions[:, :, lo:hi], self.t_now, self.scoring.lam, use_recency=self.use_recency
        )
        importance = combined_importance(cand_attn, decay, self.scoring.alpha, self.scoring.beta)

        if self.recluster_centroids and n_centroids > 0:
            # Existing centroids always re-enter the merge pool instead of
            # competing for an individual slot. Without this the summary space
            # ratchets upwards during decoding - a centroid that scores well is
            # kept whole while a fresh set of centroids is formed beside it -
            # until most of the budget is lossy summaries. Forcing them back in
            # keeps the split at exactly `n_centroids` summary slots.
            importance = importance.masked_fill(flags[:, :, lo:hi], float("-inf"))

        n_pick = min(n_pick, importance.shape[-1])
        picked_local = importance.topk(n_pick, dim=-1).indices
        n_rest = importance.shape[-1] - n_pick

        keep_parts = [picked_local + lo]
        if sink:
            keep_parts.insert(0, torch.arange(sink, device=keys.device).view(1, 1, -1).expand(b, h, sink))
        keep_parts.append(
            torch.arange(hi, n_slots, device=keys.device).view(1, 1, -1).expand(b, h, window)
        )
        keep_idx = torch.cat(keep_parts, dim=-1)

        new_keys = self._gather(keys, keep_idx)
        new_values = self._gather(values, keep_idx)
        new_pos = self._gather(positions, keep_idx)
        new_w = self._gather(weights, keep_idx)
        new_flags = self._gather(flags, keep_idx)

        # -- the remainder: dropped, or merged into centroids ------------
        removed_individuals = 0
        if n_rest > 0:
            keep_mask = torch.ones(b, h, importance.shape[-1], dtype=torch.bool, device=keys.device)
            keep_mask.scatter_(-1, picked_local, False)
            # counts are equal across heads, so a stable argsort gives a batched
            # "indices of the True entries" without ragged tensors
            rest_local = keep_mask.float().argsort(dim=-1, descending=True, stable=True)[..., :n_rest]
            rest_idx = rest_local + lo

            rest_flags = self._gather(flags, rest_idx)
            removed_individuals = int((~rest_flags[0, 0]).sum().item())

            if self.use_clustering and n_centroids > 0:
                ck, cv, cpos, cw = self._merge(
                    keys=self._gather(keys, rest_idx),
                    values=self._gather(values, rest_idx),
                    positions=self._gather(positions, rest_idx),
                    slot_weights=self._gather(weights, rest_idx),
                    merge_weights=cand_attn.gather(2, rest_local),
                    n_centroids=n_centroids,
                )
                new_keys = torch.cat([new_keys, ck], dim=2)
                new_values = torch.cat([new_values, cv], dim=2)
                new_pos = torch.cat([new_pos, cpos], dim=2)
                new_w = torch.cat([new_w, cw], dim=2)
                new_flags = torch.cat(
                    [new_flags, torch.ones(b, h, n_centroids, dtype=torch.bool, device=keys.device)],
                    dim=2,
                )

        self._write_slots(
            layer_idx,
            new_keys,
            new_values,
            new_pos,
            new_w,
            new_flags,
            n_individuals_removed=removed_individuals,
        )

    # ------------------------------------------------------------------
    def _merge(self, *, keys, values, positions, slot_weights, merge_weights, n_centroids):
        """Cluster the remainder and build centroids in un-rotated key space."""
        b, h, n, d = keys.shape
        keys_unrot = self.rope_helper.unrotate_to_zero(keys, positions)

        flat = lambda t, *shape: t.reshape(b * h, *shape)  # noqa: E731
        ck, cv, cpos, cw = cluster_and_merge(
            flat(keys_unrot, n, d),
            flat(values, n, d),
            flat(positions, n),
            flat(merge_weights, n),
            flat(slot_weights, n),
            n_centroids,
            mode=self.cluster_mode,
            n_iter=self.kmeans_iters,
            rope_position_mode=self.rope_position_mode,
        )
        ck = ck.reshape(b, h, n_centroids, d)
        cv = cv.reshape(b, h, n_centroids, d)
        cpos = cpos.reshape(b, h, n_centroids)
        cw = cw.reshape(b, h, n_centroids)
        # rotate the position-free centroid to the position the mode picked
        ck = self.rope_helper.rotate_to(ck, cpos).to(keys.dtype)
        return ck, cv.to(values.dtype), cpos, cw
