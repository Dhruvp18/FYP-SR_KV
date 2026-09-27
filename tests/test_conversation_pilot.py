from collections import defaultdict

import pytest

from eval.conversation_pilot import build_samples, score
from src.models import build_tiny_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return build_tiny_tokenizer()


def test_questions_share_context_and_answer_different_facts(tokenizer):
    samples = build_samples(tokenizer, context_len=1024, n_samples=9)
    for updated, old in zip(samples[::2], samples[1::2]):
        assert updated.context == old.context
        assert updated.context_sha256 == old.context_sha256
        assert updated.current_destination != updated.old_destination
        assert updated.old_destination in updated.options
        assert updated.options["ABCD".index(updated.answer_letter)] == updated.current_destination
        assert old.options["ABCD".index(old.answer_letter)] == old.persistent_constraint
        assert updated.persistent_constraint in updated.context
        assert updated.context.index("I need to correct") > updated.context.index("Please send it")
        assert abs(len(tokenizer(updated.context)["input_ids"]) - 1024) < 50
        for sample in (updated, old):
            assert len(set(sample.options)) == 4
            assert score(sample, sample.answer_letter) == 1.0


def test_dataset_is_stable_when_sample_count_changes(tokenizer):
    short = build_samples(tokenizer, context_len=1024, n_samples=2, seed=17)
    long = build_samples(tokenizer, context_len=1024, n_samples=4, seed=17)
    assert [s.to_dict() for s in short] == [s.to_dict() for s in long[:4]]
    other = build_samples(tokenizer, context_len=1024, n_samples=2, seed=18)
    assert short[0].context_sha256 != other[0].context_sha256


def test_placements_and_answer_letters_vary(tokenizer):
    samples = build_samples(tokenizer, context_len=1024, n_samples=18)
    assert {s.correction_depth for s in samples} == {60, 75, 90}
    assert {s.constraint_depth for s in samples} == {5, 15, 25}
    for variant in ("current_request", "old_constraint"):
        assert len({s.answer_letter for s in samples if s.variant == variant}) >= 3
