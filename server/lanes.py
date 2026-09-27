"""The 5 lane configs shown side by side in the live demo.

Every lane is built through `src.caches.make_cache` - the one factory the
research code allows (see `src/caches/__init__.py`). Nothing here decides
eviction/merge behaviour; it only picks which method name + budget to ask for.
"""

from __future__ import annotations

from src.caches import METHODS, make_cache

#: matches the 5 method families under `configs/` (excludes the standalone
#: "snapkv" cross-check and the unnamed "recency_hard_evict" corner - neither
#: is part of the demo's comparison set).
LANE_METHODS = ["full", "streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"]

#: whether each lane's cache ever forms a centroid, read from the real flags
#: `make_cache` builds it with (`src/caches/__init__.py`) rather than
#: guessed here - `server/telemetry.py` needs this to know whether "not
#: alive anymore" means "folded" or "hard-evicted" for that lane.
USES_CLUSTERING: dict[str, bool] = {name: bool(METHODS[name]["flags"].get("use_clustering")) for name in LANE_METHODS}

#: the real experiment defaults from `configs/defaults.yaml`, applied
#: identically to every lane. Each cache class absorbs whatever it doesn't
#: need via its own `**kwargs`/`**unused` catch-all, so no per-lane
#: special-casing is required here.
DEFAULT_OVERRIDES: dict = dict(
    rope_position_mode="attn_weighted",
    alpha=2.0,
    beta=0.3,
    lam=0.001,
    obs_window=32,
    pool_kernel=7,
    n_sink=4,
    centroid_frac=0.125,
    cluster_mode="kmeans",
    # SRKVCacheBase's own default is 32 (a real research choice, protecting
    # short sequences from being over-compressed). For demo prompts in the
    # 40-150 token range at 10-50% budget, budget*prompt_len is routinely
    # below 32, so that floor silently overrides the budget slider entirely -
    # every compressed lane converges on exactly 32 regardless of the
    # requested budget, which reads as "all four methods behave identically"
    # when they were never actually tested at the requested compression. This
    # override is demo-only (not touching `configs/defaults.yaml`, which real
    # experiments read) so the budget slider stays meaningful at realistic
    # demo prompt lengths.
    min_budget_tokens=8,
)


def build_lanes(model, budget: float, overrides: dict | None = None) -> dict:
    """One cache instance per lane, all sharing `budget` and `overrides`.

    `make_cache` already forces `full`/`none` to `budget=1.0` regardless of
    what it's handed (`src/caches/__init__.py`), so passing the same `budget`
    to every lane is safe and keeps the three budgeted methods (snapkv_unified,
    centroid_merge, sr_kv) on an apples-to-apples footing.
    """
    ov = dict(DEFAULT_OVERRIDES)
    if overrides:
        ov.update(overrides)
    return {name: make_cache(name, model=model, budget=budget, **ov) for name in LANE_METHODS}
