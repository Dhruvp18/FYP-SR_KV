"""Cache registry: one place that maps a method name to a configured cache.

Every experiment names its policy by a string, and this is the only function
allowed to turn that string into an object. In particular the three scored
conditions all come out of the *same* `SRKVCache` class with different flags -
that is enforced here rather than trusted to callers (see CLAUDE.md).
"""

from __future__ import annotations

from ..rope_positions import RopeHelper
from ..scoring import ScoringConfig
from .base import SRKVCacheBase
from .snapkv import SnapKVCache
from .sr_kv import CONDITION_NAMES, SRKVCache
from .streaming_llm import StreamingLLMCache

__all__ = [
    "SRKVCacheBase",
    "SnapKVCache",
    "SRKVCache",
    "StreamingLLMCache",
    "make_cache",
    "METHODS",
]

#: method name -> (class, fixed flags). The four factorial conditions of Phase 5
#: are "streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv".
METHODS: dict[str, dict] = {
    # uncompressed reference
    "full": {"cls": SRKVCacheBase, "flags": {}},
    "none": {"cls": SRKVCacheBase, "flags": {}},
    # structural baseline
    "streaming_llm": {"cls": StreamingLLMCache, "flags": {}},
    # reference-adapted SnapKV, kept independent as a cross-check
    "snapkv": {"cls": SnapKVCache, "flags": {}},
    # the three conditions that share one class
    "snapkv_unified": {"cls": SRKVCache, "flags": {"use_recency": False, "use_clustering": False}},
    "centroid_merge": {"cls": SRKVCache, "flags": {"use_recency": False, "use_clustering": True}},
    "recency_hard_evict": {"cls": SRKVCache, "flags": {"use_recency": True, "use_clustering": False}},
    "sr_kv": {"cls": SRKVCache, "flags": {"use_recency": True, "use_clustering": True}},
}

#: the Phase 5 factorial matrix, in reporting order
FACTORIAL_CONDITIONS = ["streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"]


def make_cache(method: str, *, model=None, budget: float = 1.0, **overrides) -> SRKVCacheBase:
    """Build a cache by name.

    `model` is only needed by methods that merge (they need the model's own
    `inv_freq` to re-position centroids); it is accepted for every method so
    callers do not have to special-case.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    spec = METHODS[method]
    cls, flags = spec["cls"], dict(spec["flags"])
    if method in ("full", "none"):
        # the uncompressed reference ignores any budget it is handed, so a
        # sweep can pass one uniformly without accidentally half-compressing
        budget = 1.0

    kwargs: dict = {"budget": budget, **flags}

    scoring_keys = set(ScoringConfig().to_dict())
    scoring_over = {k: overrides.pop(k) for k in list(overrides) if k in scoring_keys}
    kwargs.update(overrides)

    if cls is SRKVCache:
        base_scoring = kwargs.pop("scoring", None) or ScoringConfig()
        if scoring_over:
            base_scoring = ScoringConfig(**{**base_scoring.to_dict(), **scoring_over})
        kwargs["scoring"] = base_scoring
        if kwargs.get("use_clustering") and kwargs.get("rope_helper") is None:
            if model is None:
                raise ValueError(f"method {method!r} merges centroids and needs `model=` to read its RoPE")
            kwargs["rope_helper"] = RopeHelper.from_model(model)
    elif cls is SnapKVCache:
        # SnapKV takes the same window/pooling knobs, spelled without the dataclass
        for key in ("obs_window", "pool_kernel", "pool_type"):
            if key in scoring_over:
                kwargs[key] = scoring_over[key]
    return cls(**kwargs)
