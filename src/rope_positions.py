"""RoPE position assignment for merged centroids.

A cached key has already been rotated to its own position. Moving a key from
position ``p`` to ``p'`` is therefore a rotation by ``p' - p``: RoPE is a
rotation whose angle is linear in position, so ``R(p') = R(p' - p) R(p)``.
This holds for every rope type whose ``inv_freq`` is a constant of the model
(``default``, ``linear``, ``llama3``) and breaks for length-dependent types
(``dynamic``/NTK-by-length), which we reject explicitly.

Merging therefore works in three steps:
    1. un-rotate every cluster member to position 0   (delta = -p_i)
    2. take the attention-weighted mean               (position-free space)
    3. rotate the mean to the target position         (delta = +p_cluster)

``rope_position_mode`` decides step 3's target and is the thing Phase 4
ablates. Nothing else about the merge changes between modes.
"""

from __future__ import annotations

import torch

POSITION_MODES = ("latest", "earliest", "attn_weighted")

# rope types for which inv_freq does not depend on the sequence length seen
# so far, which is what makes the rotate-by-delta identity exact.
_SUPPORTED_ROPE_TYPES = {"default", "linear", "llama3"}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF convention: split the head dim in half and rotate the halves."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RopeHelper:
    """Applies RoPE rotations by an arbitrary (possibly fractional) delta.

    Parameters
    ----------
    inv_freq:
        1-D tensor of shape ``[head_dim // 2]``, taken from the model's own
        rotary embedding module so any rope scaling the model applies is
        inherited rather than re-derived.
    """

    def __init__(self, inv_freq: torch.Tensor):
        if inv_freq.ndim != 1:
            raise ValueError(f"inv_freq must be 1-D, got shape {tuple(inv_freq.shape)}")
        self.inv_freq = inv_freq.detach().float()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_model(cls, model) -> "RopeHelper":
        """Pull inv_freq off a loaded HF decoder model, validating rope type."""
        base = getattr(model, "model", model)
        rotary = getattr(base, "rotary_emb", None)
        if rotary is None:  # some wrappers nest one level deeper
            inner = getattr(base, "model", None)
            rotary = getattr(inner, "rotary_emb", None) if inner is not None else None
        if rotary is None or not hasattr(rotary, "inv_freq"):
            raise RuntimeError(
                "Could not find `rotary_emb.inv_freq` on the model. SR-KV needs the "
                "model's own inv_freq to re-position merged centroids."
            )

        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        scaling = getattr(cfg, "rope_scaling", None) or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type not in _SUPPORTED_ROPE_TYPES:
            raise NotImplementedError(
                f"rope_type={rope_type!r} is not supported by SR-KV centroid re-positioning. "
                f"Supported: {sorted(_SUPPORTED_ROPE_TYPES)}. Length-dependent rope scaling "
                "makes the rotate-by-delta identity invalid (see CLAUDE.md A5)."
            )
        return cls(rotary.inv_freq)

    @classmethod
    def from_config(cls, head_dim: int, rope_theta: float = 10000.0) -> "RopeHelper":
        """Build the default (unscaled) rope for tests and synthetic models."""
        idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
        return cls(1.0 / (rope_theta ** (idx / head_dim)))

    # -- core op ---------------------------------------------------------
    def rotate(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` by ``delta`` positions.

        x:     [..., seq, head_dim]
        delta: [..., seq]  (broadcastable to x's leading dims; float ok)
        """
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = delta.float().unsqueeze(-1) * inv_freq  # [..., seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        return x * cos + rotate_half(x) * sin

    def unrotate_to_zero(self, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Undo each key's own rotation, giving position-free key vectors."""
        return self.rotate(keys, -positions)

    def rotate_to(self, keys_at_zero: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate position-free key vectors to ``positions``."""
        return self.rotate(keys_at_zero, positions)


def assign_cluster_positions(
    mode: str,
    member_positions: torch.Tensor,
    member_weights: torch.Tensor,
    member_mask: torch.Tensor,
) -> torch.Tensor:
    """Pick one RoPE position per cluster.

    member_positions: [..., n_clusters, n_members]  original positions
    member_weights:   [..., n_clusters, n_members]  attention weights (>= 0)
    member_mask:      [..., n_clusters, n_members]  bool, True = real member

    Returns [..., n_clusters] float positions.
    """
    if mode not in POSITION_MODES:
        raise ValueError(f"rope_position_mode must be one of {POSITION_MODES}, got {mode!r}")

    pos = member_positions.float()
    mask = member_mask.bool()

    if mode == "latest":
        return pos.masked_fill(~mask, float("-inf")).amax(dim=-1)
    if mode == "earliest":
        return pos.masked_fill(~mask, float("inf")).amin(dim=-1)

    # attn_weighted
    w = member_weights.float().clamp_min(0.0).masked_fill(~mask, 0.0)
    denom = w.sum(dim=-1)
    weighted = (w * pos).sum(dim=-1)
    # a cluster whose weights are all zero falls back to the plain mean of its
    # members rather than producing 0/0.
    fallback_denom = mask.float().sum(dim=-1).clamp_min(1.0)
    fallback = (pos * mask.float()).sum(dim=-1) / fallback_denom
    return torch.where(denom > 0, weighted / denom.clamp_min(1e-9), fallback)
