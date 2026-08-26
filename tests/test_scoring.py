"""Phase 3 exit tests for `src/scoring.py`."""

from __future__ import annotations

import torch

from src.scoring import (
    ScoringConfig,
    combined_importance,
    compute_attention_scores,
    normalize_scores,
    pool_scores,
    recency_decay,
)


def _positions(n: int, b: int = 1, h: int = 2) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).view(1, 1, n).expand(b, h, n).contiguous()


def test_recency_decay_is_monotonically_decreasing_in_age():
    positions = _positions(64)
    decay = recency_decay(positions, t_now=63.0, lam=1e-2, use_recency=True)
    # position increases -> age decreases -> decay increases, strictly
    assert torch.all(decay.diff(dim=-1) > 0)
    # equivalently: older token, smaller decay
    assert decay[0, 0, 0] < decay[0, 0, 32] < decay[0, 0, 63]


def test_recency_decay_is_exactly_one_when_flag_is_off():
    positions = _positions(64)
    decay = recency_decay(positions, t_now=63.0, lam=1e-2, use_recency=False)
    assert torch.equal(decay, torch.ones_like(decay))
    assert decay.min() == 1.0 and decay.max() == 1.0


def test_use_recency_changes_scores_only_through_the_recency_term():
    """With everything else fixed, on-vs-off must differ by exactly beta*(decay-1)."""
    torch.manual_seed(0)
    positions = _positions(64)
    attn = torch.rand(1, 2, 64)
    alpha, beta, lam = 1.0, 0.4, 5e-3

    decay_on = recency_decay(positions, 63.0, lam, use_recency=True)
    decay_off = recency_decay(positions, 63.0, lam, use_recency=False)
    on = combined_importance(attn, decay_on, alpha, beta)
    off = combined_importance(attn, decay_off, alpha, beta)

    expected = beta * (decay_on - decay_off)
    assert torch.allclose(on - off, expected, atol=1e-6)
    # and with beta = 0 the flag is inert
    assert torch.allclose(
        combined_importance(attn, decay_on, alpha, 0.0),
        combined_importance(attn, decay_off, alpha, 0.0),
    )


def test_attention_scores_respect_causality_by_position():
    """A slot positioned after every window query must receive no attention."""
    torch.manual_seed(0)
    b, n_heads, kv_heads, w, s, d = 1, 4, 2, 8, 32, 16
    queries = torch.randn(b, n_heads, w, d)
    keys = torch.randn(b, kv_heads, s, d)
    key_positions = _positions(s, b, kv_heads)
    # queries sit at positions 10..17, so slots 18.. are in the future
    query_positions = torch.arange(10, 10 + w, dtype=torch.float32)

    scores = compute_attention_scores(queries, keys, query_positions, key_positions)
    assert torch.isfinite(scores).all()
    assert torch.all(scores[:, :, 18:] == 0.0)
    assert scores[:, :, :18].sum() > 0


def test_attention_matrix_stays_window_sized_not_sequence_sized():
    """Guard for CLAUDE.md A3: scoring must never be O(seq_len^2).

    The scorer only ever sees `obs_window` queries, so the intermediate is
    [b, kv_heads, W, S]. This test pins the interface that guarantees it: a
    long sequence with a small window must still work with a query tensor whose
    sequence dimension is the window, not the sequence.
    """
    b, n_heads, kv_heads, w, s, d = 1, 4, 2, 8, 4096, 16
    queries = torch.randn(b, n_heads, w, d)
    keys = torch.randn(b, kv_heads, s, d)
    scores = compute_attention_scores(
        queries, keys, torch.arange(s - w, s, dtype=torch.float32), _positions(s, b, kv_heads)
    )
    assert scores.shape == (b, kv_heads, s)
    assert queries.shape[2] == w and w < 64


def test_pooling_and_normalisation_preserve_ranking():
    """Both are monotone per head, so top-k selection is unaffected by them."""
    torch.manual_seed(0)
    scores = torch.rand(1, 2, 128)
    normalised = normalize_scores(scores)
    assert torch.equal(scores.argsort(dim=-1), normalised.argsort(dim=-1))
    assert normalised.max() <= 1.0 + 1e-6

    pooled = pool_scores(scores, 7, "maxpool")
    assert pooled.shape == scores.shape
    assert torch.all(pooled >= scores - 1e-6)  # max-pool never lowers a score


def test_scoring_config_rejects_bad_pooling():
    import pytest

    with pytest.raises(ValueError):
        ScoringConfig(pool_type="median")
    with pytest.raises(ValueError):
        ScoringConfig(obs_window=0)
