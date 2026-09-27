"""Pytest wrapper around the checks in `server/tools/decode_loop_cli.py`.

Runs against the repo's tiny random-weights model (no network, no downloaded
weights) via `build_demo_sample(..., corpus="synthetic")` - same philosophy
as `eval/run.py --tiny --allow_cpu`. Covers the things that would silently
break the live demo: the conservation identity, the `get_stats()` contract,
and fidelity to the real eval's own two-pass protocol
(`eval.gist_mcq.measure()`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src  # noqa: E402
from eval.gist_mcq import measure  # noqa: E402
from src.caches import make_cache  # noqa: E402
from src.models import build_tiny_model, build_tiny_tokenizer  # noqa: E402

from server.decode_loop import run_gist  # noqa: E402
from server.gist_source import build_demo_sample  # noqa: E402
from server.lanes import DEFAULT_OVERRIDES, LANE_METHODS  # noqa: E402

STATS_KEYS = {"n_tokens_cached", "n_tokens_evicted", "n_centroids", "budget_used_pct"}


def _run_tiny(budget: float, *, variant: str | None = None, context_len: int = 150):
    model = build_tiny_model()
    tokenizer = build_tiny_tokenizer()
    sample = build_demo_sample(tokenizer, variant=variant, context_len=context_len, corpus="synthetic")
    events: list[dict] = []
    result = run_gist(model, tokenizer, sample, budget=budget, overrides=None, emit=events.append)
    return model, tokenizer, sample, events, result


def test_all_five_lanes_present():
    _, _, _, _, result = _run_tiny(budget=0.3)
    assert set(result["lanes"]) == set(LANE_METHODS)


def test_conservation_holds_every_step_every_lane():
    _, _, _, events, _ = _run_tiny(budget=0.15, context_len=200)
    step_events = [e for e in events if e["type"] == "step"]
    assert step_events, "expected at least one step event"
    broken = [e for e in step_events if not e["conservation_ok"]]
    assert not broken, f"conservation broke at: {[(e['lane'], e['step']) for e in broken]}"


def test_stats_contract_keys_never_drift():
    _, _, _, events, _ = _run_tiny(budget=0.3)
    for e in events:
        if e["type"] == "step":
            assert set(e["stats"]) == STATS_KEYS


def test_evicted_cumulative_matches_n_tokens_evicted_for_hard_eviction_lanes():
    """Regression test for the bug where a hard-eviction lane's brand-new
    replacement token was mistaken for the evicted slot "repositioning"
    (same weight, so it looked like a no-op instead of a real eviction)."""
    _, _, _, events, _ = _run_tiny(budget=0.2, context_len=200)
    for lane in ("streaming_llm", "snapkv_unified"):
        lane_steps = [e for e in events if e["type"] == "step" and e["lane"] == lane]
        last = lane_steps[-1]
        assert len(last["diff"]["evicted_positions_cumulative"]) == last["stats"]["n_tokens_evicted"]


def test_every_position_accounted_for_exactly_once_including_passage_eviction():
    """Regression test for the bug where compression that happened *inside*
    a multi-token prefill forward pass (segment longer than remaining budget
    headroom) was invisible to the diff - it compared a "before" that didn't
    yet include that segment's own tokens against an already-compressed
    "after", so evictions/merges within that same call looked like positions
    that were simply never seen rather than ones that got dropped. First
    found in the passage prefill; the two-pass protocol adds a second
    at-risk call (the question prefill), so this checks both by checking the
    whole run. Asserts the same invariant the frontend relies on for every
    lane's word-by-word rendering: every position seen so far is in exactly
    one of {still an individual slot, folded into a centroid, evicted}."""
    _, _, _, events, _ = _run_tiny(budget=0.2, context_len=200)
    started = next(e for e in events if e["type"] == "run_started")
    seen_before_decode = started["passage_tokens"] + len(started["question_token_texts"])
    for lane in LANE_METHODS:
        lane_steps = [e for e in events if e["type"] == "step" and e["lane"] == lane]
        last = lane_steps[-1]
        seen = seen_before_decode + sum(1 for e in lane_steps if e["phase"] == "decode")
        n_alive_plain = last["stats"]["n_tokens_cached"] - last["stats"]["n_centroids"]
        n_folded = len(last["diff"]["folded_positions"])
        n_evicted = len(last["diff"]["evicted_positions_cumulative"])
        assert n_alive_plain + n_folded + n_evicted == seen, (
            f"{lane}: alive={n_alive_plain} folded={n_folded} evicted={n_evicted} != seen={seen}"
        )


def test_full_lane_matches_gist_mcq_measure_oracle():
    """The manual round-robin loop must be faithful to `eval.gist_mcq.
    measure()`'s own two-pass protocol - proof before trusting it on the
    compressed lanes, which have no equivalent independent oracle."""
    model, tokenizer, sample, _, result = _run_tiny(budget=0.3)
    manual_text = tokenizer.decode(result["lanes"]["full"].generated_ids, skip_special_tokens=True)

    oracle_cache = make_cache("full", model=model, budget=0.3, **DEFAULT_OVERRIDES)
    oracle = measure(model, tokenizer, sample, oracle_cache)

    assert manual_text == oracle["generated_text"]


def test_chosen_letter_and_correct_match_gist_mcq_score():
    from eval.gist_mcq import score

    _, _, sample, _, result = _run_tiny(budget=0.3)
    for row in result["summary"]:
        expected_correct = bool(score(sample, row["generated_text"]))
        assert row["correct"] == expected_correct, row["lane"]
        if row["chosen_letter"] is not None:
            assert row["chosen_letter"] in "ABCD"
