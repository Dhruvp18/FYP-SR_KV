"""Pytest wrapper around the checks in `server/tools/decode_loop_cli.py`.

Runs against the repo's tiny random-weights model (no network, no downloaded
weights) - same philosophy as `eval/run.py --tiny --allow_cpu`. Covers the
three things that would silently break the live demo: the conservation
identity, the `get_stats()` contract, and fidelity to `model.generate()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src  # noqa: E402
from eval.memory import generate_and_measure  # noqa: E402
from src.caches import make_cache  # noqa: E402
from src.models import build_tiny_model, build_tiny_tokenizer  # noqa: E402

from server.decode_loop import run  # noqa: E402
from server.lanes import DEFAULT_OVERRIDES, LANE_METHODS  # noqa: E402

STATS_KEYS = {"n_tokens_cached", "n_tokens_evicted", "n_centroids", "budget_used_pct"}
PROMPT = "hello world this is a short prompt used only for CPU plumbing"


def _run_tiny(budget: float, max_new_tokens: int):
    model = build_tiny_model()
    tokenizer = build_tiny_tokenizer()
    events: list[dict] = []
    result = run(
        model,
        tokenizer,
        PROMPT,
        budget=budget,
        max_new_tokens=max_new_tokens,
        overrides=None,
        emit=events.append,
    )
    return model, tokenizer, events, result


def test_all_five_lanes_present():
    _, _, _, result = _run_tiny(budget=0.3, max_new_tokens=6)
    assert set(result["lanes"]) == set(LANE_METHODS)


def test_conservation_holds_every_step_every_lane():
    _, _, events, _ = _run_tiny(budget=0.15, max_new_tokens=15)
    step_events = [e for e in events if e["type"] == "step"]
    assert step_events, "expected at least one step event"
    broken = [e for e in step_events if not e["conservation_ok"]]
    assert not broken, f"conservation broke at: {[(e['lane'], e['step']) for e in broken]}"


def test_stats_contract_keys_never_drift():
    _, _, events, _ = _run_tiny(budget=0.3, max_new_tokens=8)
    for e in events:
        if e["type"] == "step":
            assert set(e["stats"]) == STATS_KEYS


def test_evicted_cumulative_matches_n_tokens_evicted_for_hard_eviction_lanes():
    """Regression test for the bug where a hard-eviction lane's brand-new
    replacement token was mistaken for the evicted slot "repositioning"
    (same weight, so it looked like a no-op instead of a real eviction)."""
    _, _, events, _ = _run_tiny(budget=0.2, max_new_tokens=12)
    for lane in ("streaming_llm", "snapkv_unified"):
        lane_steps = [e for e in events if e["type"] == "step" and e["lane"] == lane]
        last = lane_steps[-1]
        assert len(last["diff"]["evicted_positions_cumulative"]) == last["stats"]["n_tokens_evicted"]


def test_every_position_accounted_for_exactly_once_including_prefill_eviction():
    """Regression test for the bug where compression that happened *inside*
    the prefill forward pass (prompt_len > budget_tokens already, before any
    decode step) was invisible to the diff - it compared an empty "before"
    against an already-compressed "after", so prefill evictions/merges looked
    like positions that were simply never seen rather than ones that got
    dropped. Asserts the same invariant the frontend relies on for every
    lane's word-by-word rendering: every position seen so far is in exactly
    one of {still an individual slot, folded into a centroid, evicted}."""
    _, _, events, _ = _run_tiny(budget=0.2, max_new_tokens=10)
    prompt_tokens = next(e["prompt_tokens"] for e in events if e["type"] == "run_started")
    for lane in LANE_METHODS:
        lane_steps = [e for e in events if e["type"] == "step" and e["lane"] == lane]
        last = lane_steps[-1]
        seen = prompt_tokens + sum(1 for e in lane_steps if e["phase"] == "decode")
        n_alive_plain = last["stats"]["n_tokens_cached"] - last["stats"]["n_centroids"]
        n_folded = len(last["diff"]["folded_positions"])
        n_evicted = len(last["diff"]["evicted_positions_cumulative"])
        assert n_alive_plain + n_folded + n_evicted == seen, (
            f"{lane}: alive={n_alive_plain} folded={n_folded} evicted={n_evicted} " f"!= seen={seen}"
        )


def test_full_lane_matches_generate_and_measure_oracle():
    """The manual round-robin loop must be faithful to `model.generate()` -
    proof before trusting it on the compressed lanes, which have no
    equivalent independent oracle."""
    model, tokenizer, _, result = _run_tiny(budget=0.3, max_new_tokens=10)
    manual_text = tokenizer.decode(result["lanes"]["full"].generated_ids, skip_special_tokens=True)

    oracle_cache = make_cache("full", model=model, budget=0.3, **DEFAULT_OVERRIDES)
    oracle = generate_and_measure(model, tokenizer, PROMPT, oracle_cache, max_new_tokens=10)

    assert manual_text == oracle["generated_text"]
