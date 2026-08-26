"""Phase 1-3 exit tests for the cache classes.

These run against a real (randomly initialised) `Qwen2ForCausalLM` on CPU, so
they exercise the genuine HuggingFace cache plumbing, attention dispatch and
mask construction - not a mock. Random weights mean the *outputs* are
meaningless, but every property tested here is about bookkeeping and selection,
which is exactly what can silently break.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from src.attn_patch import attach_cache
from src.caches import FACTORIAL_CONDITIONS, make_cache
from src.caches.snapkv import SnapKVCache
from src.caches.sr_kv import SRKVCache
from src.rope_positions import POSITION_MODES, RopeHelper
from src.scoring import ScoringConfig

PROMPT_LEN = 256
NEW_TOKENS = 12
BUDGET = 0.25
COMPRESSED_METHODS = ["streaming_llm", "snapkv", "snapkv_unified", "centroid_merge", "sr_kv"]


@pytest.fixture(scope="module")
def prompt_ids():
    torch.manual_seed(7)
    return torch.randint(0, 1000, (1, PROMPT_LEN))


def _generate(model, cache, ids, max_new_tokens=NEW_TOKENS):
    with attach_cache(model, cache):
        return model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )


def _cache(model, method, **kw):
    kw.setdefault("obs_window", 16)
    kw.setdefault("n_centroids", 8)
    return make_cache(method, model=model, budget=BUDGET, **kw)


# ---------------------------------------------------------------------------
# Phase 1: the no-op cache
# ---------------------------------------------------------------------------
def test_noop_cache_is_indistinguishable_from_dynamic_cache(tiny_model, prompt_ids):
    cache = make_cache("full", model=tiny_model)
    ours = _generate(tiny_model, cache, prompt_ids)
    reference = tiny_model.generate(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        past_key_values=DynamicCache(),
        use_cache=True,
        max_new_tokens=NEW_TOKENS,
        do_sample=False,
    )
    assert torch.equal(ours, reference), "the passthrough cache changed the model's output"


def test_noop_cache_evicts_nothing(tiny_model, prompt_ids):
    cache = make_cache("full", model=tiny_model)
    _generate(tiny_model, cache, prompt_ids)
    stats = cache.get_stats()
    assert stats["n_tokens_evicted"] == 0
    assert stats["n_centroids"] == 0
    assert stats["n_tokens_cached"] == PROMPT_LEN + NEW_TOKENS - 1
    assert cache.check_conservation()


def test_get_stats_returns_exactly_the_contract_keys(tiny_model, prompt_ids):
    for method in ["full", *COMPRESSED_METHODS]:
        cache = _cache(tiny_model, method)
        _generate(tiny_model, cache, prompt_ids)
        stats = cache.get_stats()
        assert set(stats) == {
            "n_tokens_cached",
            "n_tokens_evicted",
            "n_centroids",
            "budget_used_pct",
        }, f"{method} returned {sorted(stats)}"
        assert isinstance(stats["n_tokens_cached"], int)
        assert isinstance(stats["n_tokens_evicted"], int)
        assert isinstance(stats["n_centroids"], int)
        assert isinstance(stats["budget_used_pct"], float)


# ---------------------------------------------------------------------------
# Phase 2: budget is respected and no token is lost or double-counted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", COMPRESSED_METHODS)
def test_cache_never_exceeds_its_budget(tiny_model, prompt_ids, method):
    cache = _cache(tiny_model, method)
    _generate(tiny_model, cache, prompt_ids)
    budget = cache.budget_tokens
    assert budget is not None and budget < PROMPT_LEN
    assert cache.get_stats()["n_tokens_cached"] <= budget
    # and throughout generation, not just at the end
    assert max(cache.budget_history) <= 100.0 + 1e-6
    assert min(cache.budget_history[1:]) >= 99.0, "cache dipped below its budget mid-run"


@pytest.mark.parametrize("method", COMPRESSED_METHODS)
def test_token_accounting_is_conserved(tiny_model, prompt_ids, method):
    """seen == evicted + (cached - centroids): nothing vanishes, nothing doubles."""
    cache = _cache(tiny_model, method)
    _generate(tiny_model, cache, prompt_ids)
    stats = cache.get_stats()
    assert cache.n_tokens_seen == PROMPT_LEN + NEW_TOKENS - 1
    assert cache.n_tokens_seen == stats["n_tokens_evicted"] + (
        stats["n_tokens_cached"] - stats["n_centroids"]
    )
    if method in ("streaming_llm", "snapkv", "snapkv_unified"):
        assert stats["n_centroids"] == 0
        assert stats["n_tokens_evicted"] + stats["n_tokens_cached"] == cache.n_tokens_seen


@pytest.mark.parametrize("method", COMPRESSED_METHODS)
def test_all_layers_keep_identical_cache_lengths(tiny_model_gqa, prompt_ids, method):
    """CLAUDE.md A1: one shared causal mask means one shared length."""
    cache = _cache(tiny_model_gqa, method)
    _generate(tiny_model_gqa, cache, prompt_ids)
    lengths = {layer.keys.shape[-2] for layer in cache.layers}
    assert len(lengths) == 1, f"{method} left layers at differing lengths: {lengths}"


# ---------------------------------------------------------------------------
# Phase 3: the unified class degrades to SnapKV exactly
# ---------------------------------------------------------------------------
def test_unified_class_reproduces_snapkv_eviction_decisions(tiny_model_gqa, prompt_ids):
    """`SRKVCache(use_clustering=False)` must drop the *same tokens* as SnapKV.

    Not "similar accuracy" - the same set of retained positions, per layer and
    per head. The two implementations are written independently (see
    `src/caches/snapkv.py`), so this is a genuine cross-check that the shared
    class has not drifted away from the baseline it is supposed to subsume.
    """
    model = tiny_model_gqa
    shared = dict(budget=BUDGET, obs_window=16, pool_kernel=7, pool_type="maxpool")

    snap = SnapKVCache(n_sink=0, **shared)
    unified = SRKVCache(
        use_recency=False,
        use_clustering=False,
        n_sink=0,
        budget=BUDGET,
        scoring=ScoringConfig(
            alpha=1.0, beta=0.3, obs_window=16, pool_kernel=7, pool_type="maxpool"
        ),
    )

    for cache in (snap, unified):
        with attach_cache(model, cache):
            model(prompt_ids, past_key_values=cache, use_cache=True)

    assert snap.budget_tokens == unified.budget_tokens
    for layer_idx in snap.positions:
        a = snap.positions[layer_idx].sort(dim=-1).values
        b = unified.positions[layer_idx].sort(dim=-1).values
        assert torch.equal(a, b), f"layer {layer_idx}: kept different tokens"


def test_recency_flag_changes_which_tokens_survive(tiny_model_gqa, prompt_ids):
    """The ablation switch has to actually do something, or it proves nothing."""
    model = tiny_model_gqa
    common = dict(
        budget=BUDGET,
        use_clustering=False,
        n_sink=0,
        scoring=ScoringConfig(alpha=1.0, beta=1.0, lam=1e-2, obs_window=16),
    )
    off = SRKVCache(use_recency=False, **common)
    on = SRKVCache(use_recency=True, **common)
    for cache in (off, on):
        with attach_cache(model, cache):
            model(prompt_ids, past_key_values=cache, use_cache=True)

    differs = any(
        not torch.equal(
            off.positions[i].sort(dim=-1).values, on.positions[i].sort(dim=-1).values
        )
        for i in off.positions
    )
    assert differs, "use_recency=True kept exactly the same tokens as False"
    assert on.method_name == "recency_hard_evict"
    assert off.method_name == "snapkv_unified"


def test_condition_names_map_to_one_class(tiny_model_gqa):
    """Phase 5's four conditions must not spawn a second implementation."""
    from src.caches import METHODS

    scored = {"snapkv_unified", "centroid_merge", "sr_kv", "recency_hard_evict"}
    classes = {METHODS[name]["cls"] for name in scored}
    assert classes == {SRKVCache}, "a scored condition escaped the shared class"
    assert set(FACTORIAL_CONDITIONS) == {
        "streaming_llm",
        "snapkv_unified",
        "centroid_merge",
        "sr_kv",
    }


@pytest.mark.parametrize("mode", POSITION_MODES)
def test_every_rope_mode_produces_finite_non_degenerate_output(tiny_model_gqa, prompt_ids, mode):
    """Phase 4 gate: no NaNs, no crashes, from any position convention."""
    model = tiny_model_gqa
    cache = _cache(model, "sr_kv", rope_position_mode=mode)
    out = _generate(model, cache, prompt_ids)
    logits = model(out).logits
    assert torch.isfinite(logits).all(), f"{mode} produced non-finite logits"
    assert cache.get_stats()["n_centroids"] > 0
    assert cache.check_conservation()

    # centroid positions must be real positions within the sequence
    for layer_idx, positions in cache.positions.items():
        assert torch.isfinite(positions).all()
        assert positions.min() >= 0
        assert positions.max() <= cache.n_tokens_seen


def test_centroid_allocation_does_not_ratchet_during_decoding(tiny_model_gqa, prompt_ids):
    """Summary slots must stay at the configured allocation, not creep upward.

    Without re-clustering existing centroids, a centroid that scores well is
    kept whole while a fresh set is formed beside it, and after enough decode
    steps most of the budget is lossy summaries.
    """
    cache = _cache(tiny_model_gqa, "sr_kv", n_centroids=8)
    _generate(tiny_model_gqa, cache, prompt_ids, max_new_tokens=40)
    assert cache.get_stats()["n_centroids"] == 8


def test_rope_helper_is_required_for_merging(tiny_model):
    with pytest.raises(ValueError, match="RopeHelper"):
        SRKVCache(use_clustering=True, budget=BUDGET)
    # and the factory supplies one
    cache = make_cache("sr_kv", model=tiny_model, budget=BUDGET)
    assert isinstance(cache.rope_helper, RopeHelper)
