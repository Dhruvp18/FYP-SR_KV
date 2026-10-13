from collections import Counter

import pytest

from eval.consensus_update import N_REINFORCEMENTS, VARIANTS, build_samples, score
from src.models import build_tiny_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return build_tiny_tokenizer()


def test_every_variant_builds_a_valid_mcq(tokenizer):
    samples = build_samples(tokenizer, context_len=1536, n_samples=6)
    assert len(samples) == 6 * len(VARIANTS)
    for sample in samples:
        assert len(set(sample.options)) == 4
        assert sample.new_destination in sample.options
        assert sample.old_destination in sample.options
        assert sample.options["ABCD".index(sample.answer_letter)] == sample.new_destination
        assert sample.n_reinforcements == N_REINFORCEMENTS[sample.variant]
        assert len(sample.reinforcement_depths) == sample.n_reinforcements
        assert score(sample, sample.answer_letter) == 1.0
        assert score(sample, "Z") == 0.0


def test_same_conversation_shares_everything_except_reinforcement_count(tokenizer):
    samples = build_samples(tokenizer, context_len=1536, n_samples=4)
    by_idx = {}
    for s in samples:
        by_idx.setdefault(s.sample_idx, []).append(s)
    for idx, group in by_idx.items():
        assert len(group) == len(VARIANTS)
        assert len({s.old_destination for s in group}) == 1
        assert len({s.new_destination for s in group}) == 1
        assert len({s.decoy_destination for s in group}) == 1
        # options are independently shuffled per variant (the rng has
        # advanced by a different amount by the time each variant samples
        # its reinforcement templates), so answer_letter can legitimately
        # differ - what must hold is that it always points at new_destination
        for s in group:
            assert s.options["ABCD".index(s.answer_letter)] == s.new_destination
        # the passages themselves differ, because the reinforcement count differs
        assert len({s.context_sha256 for s in group}) == len(VARIANTS)


def test_decoy_and_reinforcements_are_both_present_and_late(tokenizer):
    samples = build_samples(tokenizer, context_len=2048, n_samples=3,
                             variants=("consensus_n3",))
    for sample in samples:
        ctx = sample.context
        assert "before it was corrected" in ctx
        assert ctx.index("I need to correct") > ctx.index("Please send it")
        assert ctx.index("before it was corrected") > ctx.index("I need to correct")
        # three distinct reinforcement sentences, each mentioning new_destination
        reinforcement_mentions = ctx.count(sample.new_destination)
        # new_destination appears once in the correction line itself plus
        # once per reinforcement
        assert reinforcement_mentions >= sample.n_reinforcements


def test_decoy_never_claims_the_old_address_is_still_current(tokenizer):
    """Regression test for the 2026-10-13 sanity failure: an earlier decoy
    wording ("disregard the update... after all") made the task genuinely
    ambiguous even for the uncompressed `full` reference - every one of its
    wrong answers picked the decoy, not a random option, because the decoy
    explicitly claimed to override the real correction. The decoy must only
    ever describe the old address as a superseded, past fact.
    """
    samples = build_samples(tokenizer, context_len=2048, n_samples=5)
    for sample in samples:
        ctx = sample.context.lower()
        for phrase in ("after all", "disregard the update", "actually",
                       "take priority", "override"):
            assert phrase not in ctx, (
                f"decoy wording reintroduced an override claim: {phrase!r}"
            )


def test_dataset_is_stable_when_sample_count_changes(tokenizer):
    short = build_samples(tokenizer, context_len=1536, n_samples=2, seed=99)
    long = build_samples(tokenizer, context_len=1536, n_samples=4, seed=99)
    assert [s.to_dict() for s in short] == [s.to_dict() for s in long[:len(short)]]
    other = build_samples(tokenizer, context_len=1536, n_samples=2, seed=100)
    assert short[0].context_sha256 != other[0].context_sha256


def test_n1_has_nothing_to_cluster_n5_has_the_most(tokenizer):
    samples = build_samples(tokenizer, context_len=2048, n_samples=5)
    counts = Counter(s.n_reinforcements for s in samples if s.variant == "consensus_n1")
    assert set(counts) == {1}
    counts5 = Counter(s.n_reinforcements for s in samples if s.variant == "consensus_n5")
    assert set(counts5) == {5}


def test_rejects_unknown_variant(tokenizer):
    with pytest.raises(ValueError):
        build_samples(tokenizer, context_len=1536, n_samples=1, variants=("nope",))


def test_rejects_context_too_short_for_every_reinforcement(tokenizer):
    with pytest.raises(ValueError):
        build_samples(tokenizer, context_len=256, n_samples=1)
