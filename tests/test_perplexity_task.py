"""Phase 8 / H1: perplexity on a cache compressed over the prefix.

See PREREGISTRATION.md. These guard the two ways this measurement can look
fine and mean nothing.
"""

from __future__ import annotations

import pytest
import torch

from eval import perplexity as P
from src.caches import make_cache
from src.models import build_tiny_model, build_tiny_tokenizer


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model(), build_tiny_tokenizer()


def test_measure_actually_compresses_the_cache(tiny):
    """Without attach_cache, model(...) runs plain SDPA and nothing compresses.

    This is not hypothetical: the first working version of measure() called
    model(...) directly, every method returned nll equal to 15 decimal places,
    and the caches reported evicted=0, centroids=0. A perplexity comparison
    over uncompressed caches would have produced a confident "no difference"
    from an experiment that never tested anything.
    """
    model, tok = tiny
    sample = P.build_samples(tok, context_lengths=[384], n_samples=1)[0]

    compressed = make_cache("snapkv_unified", model=model, budget=0.3)
    P.measure(model, sample, compressed)
    stats = compressed.get_stats()
    assert stats["n_tokens_evicted"] > 0, "cache never compressed - is attach_cache missing?"
    assert stats["n_tokens_cached"] < len(sample.prefix_ids)

    merging = make_cache("centroid_merge", model=model, budget=0.3)
    P.measure(model, sample, merging)
    assert merging.get_stats()["n_centroids"] > 0, "clustering produced no centroids"


def test_uncompressed_and_compressed_differ(tiny):
    """If full and a compressed method agree exactly, compression is a no-op."""
    model, tok = tiny
    sample = P.build_samples(tok, context_lengths=[384], n_samples=1)[0]
    full = P.measure(model, sample, make_cache("full", model=model, budget=1.0))
    comp = P.measure(model, sample, make_cache("snapkv_unified", model=model, budget=0.3))
    assert full["nll"] != comp["nll"], "compressed cache gave identical loss to uncompressed"


def test_windows_do_not_overlap(tiny):
    """Overlapping windows would make the paired bootstrap invalid."""
    _, tok = tiny
    samples = P.build_samples(tok, context_lengths=[128], n_samples=4, continuation_tokens=32)
    spans = [(s.prefix_ids + s.continuation_ids) for s in samples]
    assert all(len(s) == 160 for s in spans)
    for i in range(len(spans) - 1):
        assert spans[i] != spans[i + 1], "windows repeat; samples are not independent"


def test_score_keeps_larger_is_better(tiny):
    """`accuracy` is read by every gate and figure as 'higher is better'.

    Putting raw perplexity there would invert every comparison and mix an
    unbounded scale into 0-1 axes.
    """
    result = {"nll": 2.0, "perplexity": 7.389}
    assert P.score(None, result) == -2.0
    assert P.score(None, {"nll": 1.0}) > P.score(None, {"nll": 3.0})


def test_perplexity_records_are_not_counted_as_niah():
    """They carry a context_len, so anything inferring task from that is wrong."""
    from scripts.check_results import check_completeness
    from scripts.make_plots import _task_of

    ppl = {"model": "m", "method": "sr_kv", "budget": 0.3, "context_len": 2048,
           "sample_idx": 0, "accuracy": -2.0, "task": "perplexity"}
    assert _task_of(ppl) == "perplexity"
    # no KeyError on the missing 'depth', and it contributes no NIAH cells
    missing = check_completeness(
        [ppl], model="m", budgets=[0.3], methods=["sr_kv"],
        contexts=[2048], depths=[0], n_samples=1, lb_tasks=[],
    )
    assert len(missing) == 1 and "sr_kv" in missing[0]
