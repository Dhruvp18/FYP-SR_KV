"""Clustering of low-importance tokens and centroid construction.

All functions are batched over "groups", where one group is one
``(batch_item, kv_head)`` pair - SnapKV-style selection is per-head, so
clustering is per-head too.

Fixed-k clustering only (see CLAUDE.md A1): HuggingFace builds one causal
mask per forward pass and shares it across layers, so every layer must end up
with exactly the same cache length. A cosine-*threshold* rule produces a
data-dependent number of clusters and would break that invariant; it is kept
here as `cosine_threshold_labels` for offline analysis only and is not wired
into any cache.

Centroids are built in *un-rotated* key space (RoPE undone), so cluster
membership and the merged vector are position-free; the caller rotates the
result to the position chosen by `rope_position_mode`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rope_positions import assign_cluster_positions

CLUSTER_MODES = ("kmeans", "temporal_chunk")


def _temporal_chunk_labels(n: int, k: int, device) -> torch.Tensor:
    """Split `n` position-sorted candidates into `k` contiguous chunks."""
    return (torch.arange(n, device=device) * k) // max(n, 1)


def _one_hot(labels: torch.Tensor, k: int) -> torch.Tensor:
    """[G, N] int labels -> [G, N, k] float one-hot."""
    return F.one_hot(labels.long(), num_classes=k).to(torch.float32)


def _kmeans_cosine(keys_n: torch.Tensor, k: int, n_iter: int) -> torch.Tensor:
    """Spherical k-means on unit-normalised keys. Returns labels [G, N].

    Initialised from contiguous temporal chunks: deterministic (no RNG in the
    experiment path) and it starts from a partition that already respects
    sequence locality, which matters because a cluster's members later get
    collapsed onto a single RoPE position.
    """
    g, n, _ = keys_n.shape
    labels = _temporal_chunk_labels(n, k, keys_n.device).expand(g, n).contiguous()

    for _ in range(n_iter):
        onehot = _one_hot(labels, k)                                   # [G, N, k]
        counts = onehot.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)       # [G, k, 1]
        centroids = torch.einsum("gnk,gnd->gkd", onehot, keys_n) / counts
        centroids = F.normalize(centroids, dim=-1)
        sim = torch.einsum("gnd,gkd->gnk", keys_n, centroids)
        labels = sim.argmax(dim=-1)

    return _fix_empty_clusters(keys_n, labels, k)


def _fix_empty_clusters(keys_n: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Guarantee every one of the k slots has at least one member.

    An empty cluster would give a 0/0 centroid, and the slot has to exist
    anyway because the compressed length is fixed. Worst-fitting tokens
    (lowest similarity to their own centroid) are moved into the empty slots;
    any group still left with an empty slot falls back to temporal chunking.
    """
    g, n, _ = keys_n.shape
    onehot = _one_hot(labels, k)
    counts = onehot.sum(dim=1)                                          # [G, k]
    if bool((counts > 0).all()):
        return labels
    if k > n:
        raise ValueError(f"cannot form {k} non-empty clusters from {n} candidates")

    centroids = F.normalize(
        torch.einsum("gnk,gnd->gkd", onehot, keys_n) / counts.clamp_min(1e-6).unsqueeze(-1),
        dim=-1,
    )
    own_sim = torch.einsum("gnd,gkd->gnk", keys_n, centroids)
    own_sim = own_sim.gather(-1, labels.unsqueeze(-1)).squeeze(-1)      # [G, N]

    labels = labels.clone()
    for gi in range(g):
        empty = (counts[gi] == 0).nonzero(as_tuple=True)[0]
        if empty.numel() == 0:
            continue
        worst = own_sim[gi].argsort()[: empty.numel()]
        labels[gi, worst] = empty
        still_empty = _one_hot(labels[gi : gi + 1], k).sum(dim=1) == 0
        if bool(still_empty.any()):
            labels[gi] = _temporal_chunk_labels(n, k, keys_n.device)
    return labels


def cluster_and_merge(
    keys_unrot: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    slot_weights: torch.Tensor,
    k: int,
    *,
    mode: str = "kmeans",
    n_iter: int = 4,
    rope_position_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse N candidate slots into exactly k centroid slots.

    keys_unrot   : [G, N, D] position-free keys (RoPE already undone)
    values       : [G, N, D]
    positions    : [G, N] float, original RoPE positions
    weights      : [G, N] attention weight of each candidate (>= 0)
    slot_weights : [G, N] how many original tokens each candidate stands for

    Returns (centroid_keys_unrot [G,k,D], centroid_values [G,k,D],
    centroid_positions [G,k], centroid_slot_weights [G,k]). The caller rotates
    the keys to `centroid_positions`.
    """
    if mode not in CLUSTER_MODES:
        raise ValueError(f"cluster mode must be one of {CLUSTER_MODES}, got {mode!r}")
    g, n, d = keys_unrot.shape
    if k > n:
        raise ValueError(f"asked for {k} centroids from only {n} candidates")

    # process candidates in temporal order, so chunk-based clustering and
    # chunk-based init mean what they say regardless of the caller's gather order
    order = positions.argsort(dim=-1)
    gather_d = order.unsqueeze(-1).expand(g, n, d)
    keys_unrot = keys_unrot.gather(1, gather_d)
    values = values.gather(1, gather_d)
    positions = positions.gather(1, order)
    weights = weights.gather(1, order)
    slot_weights = slot_weights.gather(1, order)

    if mode == "temporal_chunk":
        labels = _temporal_chunk_labels(n, k, keys_unrot.device).expand(g, n).contiguous()
    else:
        labels = _kmeans_cosine(F.normalize(keys_unrot.float(), dim=-1), k, n_iter)

    onehot = _one_hot(labels, k)                                        # [G, N, k]

    # attention-weighted average inside each cluster; a cluster whose members
    # all score zero degrades to a plain mean rather than 0/0
    w = weights.float().clamp_min(0.0).unsqueeze(-1) * onehot           # [G, N, k]
    w_sum = w.sum(dim=1, keepdim=True)                                  # [G, 1, k]
    w = torch.where(w_sum > 1e-9, w, onehot)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)

    centroid_k = torch.einsum("gnk,gnd->gkd", w, keys_unrot.float()).to(keys_unrot.dtype)
    centroid_v = torch.einsum("gnk,gnd->gkd", w, values.float()).to(values.dtype)
    centroid_sw = torch.einsum("gnk,gn->gk", onehot, slot_weights.float())

    centroid_pos = assign_cluster_positions(
        rope_position_mode,
        member_positions=positions.unsqueeze(1).expand(g, k, n),
        member_weights=w.transpose(1, 2),
        member_mask=onehot.transpose(1, 2).bool(),
    )
    return centroid_k, centroid_v, centroid_pos, centroid_sw


def cosine_threshold_labels(keys_unrot: torch.Tensor, threshold: float) -> torch.Tensor:
    """OFFLINE ANALYSIS ONLY - greedy cosine-threshold clustering.

    Produces a data-dependent number of clusters, which violates the uniform
    cache-length invariant (CLAUDE.md A1). Deliberately not used by any cache;
    it exists so the report can quantify how many clusters a threshold rule
    would have produced at a given similarity cutoff.
    """
    keys_n = F.normalize(keys_unrot.float(), dim=-1)
    g, n, _ = keys_n.shape
    labels = torch.full((g, n), -1, dtype=torch.long, device=keys_n.device)
    for gi in range(g):
        centers: list[torch.Tensor] = []
        for i in range(n):
            v = keys_n[gi, i]
            if centers:
                sims = torch.stack([c @ v for c in centers])
                best = int(sims.argmax())
                if float(sims[best]) >= threshold:
                    labels[gi, i] = best
                    continue
            centers.append(v)
            labels[gi, i] = len(centers) - 1
    return labels
