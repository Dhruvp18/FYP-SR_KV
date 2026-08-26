"""Token importance scoring: windowed attention + recency decay.

    importance_i = alpha * attn_score_i + beta * recency_decay_i

`attn_score_i` is SnapKV-style: the attention mass that the last `obs_window`
query positions pay to cached slot `i`. This is the ONLY place in the repo
where an attention matrix is materialised, and its shape is
`[batch, kv_heads, obs_window, n_slots]` - `obs_window` is 8-32, never
`seq_len`. That bound is exactly why this is affordable on a T4 where H2O's
full `[seq_len, seq_len]` eager attention is not (CLAUDE.md A3).

`recency_decay_i = exp(-lambda * (t_now - t_i))`, and is identically 1.0 when
`use_recency=False`, so the beta term degenerates to a constant offset that
cannot reorder tokens. That is what makes the Phase 5 recency ablation a
genuine one-flag change rather than two code paths.

Causality is enforced from *positions*, not from tensor layout: after the
cache has been compressed, slot order no longer implies position order, and a
query may only see slots whose position precedes it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .compat import repeat_kv


@dataclass
class ScoringConfig:
    """Scoring hyperparameters (alpha/beta/lam are swept in Phase 6)."""

    alpha: float = 1.0
    beta: float = 0.3
    lam: float = 1e-3
    obs_window: int = 32
    pool_kernel: int = 7
    pool_type: str = "maxpool"  # "maxpool" | "avgpool" | "none"
    normalize_attn: bool = True

    def __post_init__(self):
        if self.pool_type not in ("maxpool", "avgpool", "none"):
            raise ValueError(f"pool_type must be maxpool|avgpool|none, got {self.pool_type!r}")
        if self.obs_window < 1:
            raise ValueError("obs_window must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


def compute_attention_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    scaling: float | None = None,
) -> torch.Tensor:
    """Attention mass each cached slot receives from the observation window.

    queries        : [b, n_heads, W, d]   last W query vectors (RoPE applied)
    keys           : [b, kv_heads, S, d]  all cached keys (RoPE applied)
    query_positions: [W]                  absolute position of each window query
    key_positions  : [b, kv_heads, S]     absolute position of each cached slot

    Returns [b, kv_heads, S], non-negative, unpooled and unnormalised.
    """
    b, n_heads, w, d = queries.shape
    kv_heads, s = keys.shape[1], keys.shape[2]
    if n_heads % kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} not divisible by kv_heads={kv_heads}")
    n_rep = n_heads // kv_heads
    scaling = scaling if scaling is not None else d**-0.5

    keys_rep = repeat_kv(keys, n_rep)
    # [b, n_heads, W, S] - the only materialised attention matrix in the repo
    logits = torch.matmul(queries.float(), keys_rep.float().transpose(2, 3)) * scaling

    # a query may only attend to slots that precede it in true position space
    causal = key_positions.unsqueeze(2) > query_positions.view(1, 1, w, 1)  # [b,kv,W,S]
    if n_rep > 1:
        causal = causal.repeat_interleave(n_rep, dim=1)
    logits = logits.masked_fill(causal, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)  # a query that sees nothing scores nothing
    scores = probs.sum(dim=2)  # [b, n_heads, S] summed over the window queries

    # collapse query heads onto their kv head (GQA): mean within the group
    return scores.view(b, kv_heads, n_rep, s).mean(dim=2)


def pool_scores(scores: torch.Tensor, pool_kernel: int, pool_type: str) -> torch.Tensor:
    """SnapKV's smoothing step: keeps whole informative spans, not lone tokens.

    Applied to the candidate region only - pooling over a wider region and
    slicing afterwards would give different values at the boundary, and the
    Phase 3 equivalence test against the SnapKV baseline would not hold.
    """
    if pool_type == "none" or pool_kernel <= 1 or scores.shape[-1] <= pool_kernel:
        return scores
    b, h, s = scores.shape
    pad = pool_kernel // 2
    flat = scores.reshape(b * h, 1, s)
    if pool_type == "maxpool":
        flat = F.max_pool1d(flat, kernel_size=pool_kernel, stride=1, padding=pad)
    else:
        flat = F.avg_pool1d(flat, kernel_size=pool_kernel, stride=1, padding=pad)
    return flat[..., :s].reshape(b, h, s)


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Max-normalise per head so alpha and beta live on comparable scales.

    Monotone per head, so it cannot change which tokens a top-k selects.
    """
    return scores / scores.amax(dim=-1, keepdim=True).clamp_min(1e-9)


def recency_decay(
    positions: torch.Tensor,
    t_now: float | torch.Tensor,
    lam: float,
    use_recency: bool = True,
) -> torch.Tensor:
    """exp(-lam * age), or exactly 1.0 everywhere when `use_recency=False`.

    Ones rather than zeros: the beta term then contributes a constant that
    cannot re-rank tokens, so flipping `use_recency` changes exactly one thing
    about the policy and nothing else.
    """
    if not use_recency:
        return torch.ones_like(positions, dtype=torch.float32)
    now = torch.as_tensor(t_now, device=positions.device, dtype=torch.float32)
    age = (now - positions.float()).clamp_min(0.0)
    return torch.exp(-lam * age)


def combined_importance(
    attn_scores: torch.Tensor,
    decay: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """importance = alpha * attn + beta * recency."""
    return alpha * attn_scores.float() + beta * decay.float()


def importance_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    t_now: float,
    cfg: ScoringConfig,
    *,
    use_recency: bool = True,
    scaling: float | None = None,
) -> torch.Tensor:
    """Full importance for every cached slot: [b, kv_heads, S].

    Convenience wrapper; the caches call the pieces separately so that pooling
    can be restricted to the candidate region.
    """
    attn = compute_attention_scores(
        queries, keys, query_positions, key_positions, scaling=scaling
    )
    attn = pool_scores(attn, cfg.pool_kernel, cfg.pool_type)
    if cfg.normalize_attn:
        attn = normalize_scores(attn)
    decay = recency_decay(key_positions, t_now, cfg.lam, use_recency=use_recency)
    return combined_importance(attn, decay, cfg.alpha, cfg.beta)
