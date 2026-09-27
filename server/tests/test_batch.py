"""Pytest for `server/batch.py` - the small live demo of the real Phase 8
gist_mcq benchmark table. Runs against the tiny random-weights model (no
network, no downloaded weights), with `n_per_cell` overridden small so the
test stays fast - `run_gist_batch` doesn't touch eviction/scoring logic
itself (that's `decode_loop.run_gist`, already covered by
`test_decode_loop.py`), so this only needs to check the batch orchestration:
every cell gets exactly `n_per_cell` observations, the table only ever grows,
and cancellation leaves a consistent partial table.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src  # noqa: E402
from src.models import build_tiny_model, build_tiny_tokenizer  # noqa: E402

from server.batch import BATCH_VARIANTS, run_gist_batch  # noqa: E402
from server.lanes import LANE_METHODS  # noqa: E402


def _run_tiny_batch(n_per_cell: int, should_cancel=lambda: False):
    model = build_tiny_model()
    tokenizer = build_tiny_tokenizer()
    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)

    result = run_gist_batch(
        model, tokenizer, context_len=150, emit=emit, should_cancel=should_cancel, n_per_cell=n_per_cell
    )
    return events, result


def test_every_cell_gets_exactly_n_per_cell_observations(monkeypatch):
    # run_gist_batch always requests corpus="pg" (the live-demo default);
    # force synthetic here so the test needs no network/download, exactly
    # like decode_loop_cli.py's default path.
    import server.batch as batch_module

    real_build = batch_module.build_demo_sample

    def synthetic_build(tokenizer, *, variant, context_len, corpus):
        return real_build(tokenizer, variant=variant, context_len=context_len, corpus="synthetic")

    monkeypatch.setattr(batch_module, "build_demo_sample", synthetic_build)

    events, result = _run_tiny_batch(n_per_cell=2)
    table = result["table"]
    assert result["cancelled"] is False

    for variant in BATCH_VARIANTS:
        for method in LANE_METHODS:
            cell = table[variant][method]
            assert cell["total"] == 2, f"{variant}/{method}: {cell}"
            assert 0 <= cell["correct"] <= cell["total"]

    sample_completes = [e for e in events if e["type"] == "batch_sample_complete"]
    assert len(sample_completes) == len(BATCH_VARIANTS) * 2

    progresses = [e for e in events if e["type"] == "batch_progress"]
    assert [p["done"] for p in progresses] == list(range(1, len(sample_completes) + 1))

    completes = [e for e in events if e["type"] == "batch_complete"]
    assert len(completes) == 1
    assert completes[0]["table"] == table


def test_cancel_mid_batch_leaves_a_consistent_partial_table(monkeypatch):
    import server.batch as batch_module

    real_build = batch_module.build_demo_sample

    def synthetic_build(tokenizer, *, variant, context_len, corpus):
        return real_build(tokenizer, variant=variant, context_len=context_len, corpus="synthetic")

    monkeypatch.setattr(batch_module, "build_demo_sample", synthetic_build)

    calls = {"n": 0}

    def cancel_after_two_samples():
        return calls["n"] >= 2

    # count samples via a side channel: wrap emit to bump `calls` on every
    # batch_sample_complete, so should_cancel can see how far we've gotten.
    events: list[dict] = []
    model = build_tiny_model()
    tokenizer = build_tiny_tokenizer()

    def emit(event: dict) -> None:
        if event["type"] == "batch_sample_complete":
            calls["n"] += 1
        events.append(event)

    result = run_gist_batch(
        model, tokenizer, context_len=150, emit=emit, should_cancel=cancel_after_two_samples, n_per_cell=5
    )

    assert result["cancelled"] is True
    total_observed = sum(
        cell["total"]
        for variant in BATCH_VARIANTS
        for method in LANE_METHODS
        for cell in [result["table"][variant][method]]
    )
    # every lane observes every sample together (interleaved), so total
    # observations = (samples completed) * (lanes)
    assert total_observed % len(LANE_METHODS) == 0
    assert total_observed > 0

    completes = [e for e in events if e["type"] == "batch_complete"]
    assert len(completes) == 1
    assert completes[0]["cancelled"] is True
