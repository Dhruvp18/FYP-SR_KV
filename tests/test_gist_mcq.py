"""Phase 8 / H3: gist recall under redundant, MCQ-scored conditions.

See PREREGISTRATION.md addendum. These pin the properties the experiment
design depends on: the MCQ has exactly one right answer among distinct
options, samples are reproducible, and the scorer reads the model's choice
robustly rather than requiring an exact-format match (the whole point of this
task is to avoid the exact-string penalty H1 identified).
"""

from __future__ import annotations

import pytest

from eval import gist_mcq as G
from src.models import build_tiny_tokenizer


@pytest.fixture(scope="module")
def tok():
    return build_tiny_tokenizer()


def _build(tok, **kwargs):
    """`corpus="synthetic"` for every test here: these check structure
    (splicing, MCQ shape, determinism), not task difficulty, so they should
    stay offline and fast. `corpus="pg"` is the real-run default (see
    `build_samples`'s docstring for why) and is exercised by the harness
    integration test instead."""
    return G.build_samples(tok, corpus="synthetic", **kwargs)


# ---------------------------------------------------------------------------
# sample construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", G.VARIANTS)
def test_each_sample_has_one_valid_answer_among_distinct_options(tok, variant):
    samples = _build(tok, context_lengths=[512], variants=[variant], n_samples=8)
    assert len(samples) == 8
    for s in samples:
        assert len(s.options) == 4
        assert len(set(s.options)) == 4, "duplicate MCQ options make the question ambiguous"
        assert s.answer_letter in G.LETTERS
        idx = G.LETTERS.index(s.answer_letter)
        assert s.options[idx]  # the lettered option the answer points to exists
        assert s.question.count(")") == 4
        assert s.variant == variant


def test_samples_are_reproducible_given_the_same_seed(tok):
    """Kaggle sessions get killed and restarted mid-sweep - the corpus and every
    fact placement must be byte-identical on a re-run, same as NIAH."""
    a = _build(tok, context_lengths=[512], n_samples=4, seed=7)
    b = _build(tok, context_lengths=[512], n_samples=4, seed=7)
    assert [s.context for s in a] == [s.context for s in b]
    assert [s.answer_letter for s in a] == [s.answer_letter for s in b]


def test_different_seeds_do_not_collapse_to_the_same_answer_key(tok):
    """A fixed answer key (e.g. always 'A') would let a model guess for free."""
    samples = _build(tok, context_lengths=[512], variants=["attribution"], n_samples=20)
    letters = {s.answer_letter for s in samples}
    assert len(letters) > 1


def test_unknown_variant_is_rejected(tok):
    with pytest.raises(ValueError):
        _build(tok, context_lengths=[512], variants=["not_a_real_variant"])


def test_task_ids_are_unique_and_carry_the_variant(tok):
    samples = _build(tok, context_lengths=[512], variants=list(G.VARIANTS), n_samples=5)
    ids = [s.task_id for s in samples]
    assert len(set(ids)) == len(ids)
    assert all("/attribution/" in i or "/aggregation/" in i for i in ids)


def test_context_length_is_controlled_by_token_count_not_variant(tok):
    """Both variants insert a different number/length of fragments; the target
    context length must still be respected rather than drifting with them."""
    for variant in G.VARIANTS:
        samples = _build(tok, context_lengths=[1024], variants=[variant], n_samples=3)
        for s in samples:
            n_tokens = len(tok(s.context, add_special_tokens=False)["input_ids"])
            # decode->re-encode can drift slightly at splice boundaries (same
            # caveat as niah.build_samples); it must stay in the right ballpark
            assert abs(n_tokens - 1024) < 50, f"{variant} context drifted to {n_tokens} tokens"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("generated,answer,expected", [
    ("B", "B", 1.0),
    ("b", "B", 1.0),
    ("Answer: B", "B", 1.0),
    ("The answer is B) Blue.", "B", 1.0),
    ("A", "B", 0.0),
    ("I'm not sure.", "B", 0.0),
    ("", "B", 0.0),
])
def test_score_reads_the_first_standalone_letter(generated, answer, expected):
    sample = G.GistSample(context_len=512, variant="attribution", sample_idx=0,
                           options=["Red", "Blue", "Green", "Yellow"], answer_letter=answer)
    assert G.score(sample, generated) == expected


def test_score_does_not_match_a_letter_embedded_in_a_word():
    """'Cab' contains a 'B' but must not score as a choice of B."""
    sample = G.GistSample(context_len=512, variant="attribution", sample_idx=0,
                           options=["Red", "Blue", "Green", "Yellow"], answer_letter="B")
    assert G.score(sample, "It was a Cab, definitely.") == 0.0
