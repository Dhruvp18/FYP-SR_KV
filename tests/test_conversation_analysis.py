import pytest

from scripts.analyse_conversation_pilot import analyse, paired_interval


def test_paired_interval_resamples_conversations():
    # Two question scores are aggregated before reaching this function.
    result = paired_interval({0: 1.0, 1: 0.5}, {0: 0.5, 1: 0.0})
    assert result == dict(n=2, difference=0.5, ci_low=0.5, ci_high=0.5)


def test_unmatched_pairs_are_rejected():
    with pytest.raises(ValueError):
        paired_interval({0: 1.0}, {1: 1.0})


def test_different_contexts_cannot_be_paired():
    with pytest.raises(ValueError, match="different conversations"):
        analyse([
            dict(sample_idx=0, method="sr_kv", budget=0.1, context_sha256="one"),
            dict(sample_idx=0, method="recency_hard_evict", budget=0.1, context_sha256="two"),
        ])
