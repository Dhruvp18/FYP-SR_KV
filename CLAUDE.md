# CLAUDE.md — SR-KV Cache Interface Contract

## Base class
All cache variants subclass `SRKVCacheBase(transformers.cache_utils.DynamicCache)`.

## Required interface
- `update(key_states, value_states, layer_idx, cache_kwargs) -> (k, v)`
  Standard DynamicCache signature; eviction/merge logic hooks in here.
- `get_stats() -> dict`
  Must return exactly: `{"n_tokens_cached": int, "n_tokens_evicted": int,
  "n_centroids": int, "budget_used_pct": float}`. Key names are fixed — the
  eval harness polls this and does not tolerate renamed keys.
- `attn_implementation="eager"` is permitted ONLY inside the local
  observation window computed in `scoring.py`. If eager attention is applied
  to the full sequence anywhere in the codebase, that is a bug — this is the
  exact mistake that makes H2O impractical on a T4.

## Method config flags (shared across all scored methods)
- `budget: float` — target cache size as fraction of full sequence
- `use_recency: bool` — if False, recency_decay is zeroed (ablation switch)
- `use_clustering: bool` — if False, low-score tokens are hard-evicted
  instead of merged (ablation switch)
- `rope_position_mode: "latest" | "earliest" | "attn_weighted"`

`SRKVCache` with `use_recency=False, use_clustering=True` IS the
centroid-merge baseline. Do not write a separate class for it. This is what
makes the recency-term ablation trustworthy — it isolates one config flag,
not two independent implementations that could differ in unrelated ways.

`SnapKV` with hard eviction is `use_recency=False, use_clustering=False` and
CAN reuse the same class if convenient, but StreamingLLM is architecturally
different (sink + sliding window, not a scored method) and stays a separate
class — do not force it into the same flags.

## Forbidden
- Reimplementing H2O's full eager-attention scoring anywhere.
- Any caching/eviction logic inside `eval/*.py` — eval code only calls
  `model.generate()` with an attached cache instance and reads `get_stats()`.
- Silent CPU fallback on CUDA OOM — raise the error so checkpoint/resume
  logic in `scripts/checkpoint_utils.py` can catch it and resume cleanly.

---

# Appendix A — implementation invariants derived from the contract

Everything above is the contract as specified. Everything below is a
consequence of it that surfaced while implementing against
`transformers>=5.0`, and is binding on all future code in this repo.

## A1. Uniform cache length across layers is mandatory
HuggingFace builds the causal attention mask **once per forward pass**, using
`cache.get_mask_sizes()`, and reuses that single mask for every decoder
layer. Therefore every layer's cache must have **identical length** at the
start of each forward pass, or attention raises a shape error (or worse,
silently mis-masks).

Consequences:
- Compression always compresses to an **exact, deterministic token count**
  (`budget_tokens`), identical for every layer. Never a data-dependent count.
- Threshold-based cosine clustering (variable number of clusters) is
  therefore **not usable** as the production path: it yields a different
  number of centroids per layer/head. `clustering.py` implements fixed-k
  clustering instead (`kmeans`, `temporal_chunk`); the cosine-threshold
  variant exists only as `cluster_mode="cosine_threshold"` for offline
  analysis and is deliberately not wired into the cache. Engineering
  constraint, documented rather than hidden.

## A2. Compression happens *after* attention, not inside `update()`
`update()` is called before the attention interface runs, so it has no access
to `query_states` — and query states are exactly what SnapKV-style scoring
needs. The cache therefore exposes a second hook:

- `post_attention(layer_idx, query_states)` — called by the patched attention
  function in `src/attn_patch.py` immediately *after* that layer's attention
  output has been computed. All eviction/merging happens there.

`update()` remains where cache growth and accounting hook in, per the
contract. This also means peak prefill KV memory is
`(L-1) * budget_tokens + full_seq_len` rather than `L * full_seq_len`, since
layer *i* is compressed before layer *i+1* runs.

## A3. Where eager attention is allowed
The model always runs with the `srkv` attention implementation, which
delegates to SDPA for the actual attention output. The **only** materialised
attention matrix in this codebase is in
`scoring.compute_attention_scores()`, of shape
`[batch, kv_heads, obs_window, seq_len]` — `obs_window` is 8–32, never
`seq_len`. Any code materialising `[*, seq_len, seq_len]` is a bug.

## A4. `get_stats()` accounting semantics
The four contract keys mean exactly:
- `n_tokens_cached` — KV slots currently in the cache, per layer (centroid
  slots included). Layers are uniform, so this is one number.
- `n_tokens_evicted` — cumulative count of *original* tokens no longer
  individually represented: hard-dropped tokens **and** tokens folded into a
  centroid.
- `n_centroids` — slots currently holding a centroid.
- `budget_used_pct` — `100 * n_tokens_cached / budget_tokens`.

Every slot carries a `weight` = how many original tokens it represents (1 for
a real token, >=2 for a centroid). Dropping a slot adds its weight to
`n_tokens_evicted`. The conservation law checked in
`tests/test_cache_correctness.py` is:

    n_tokens_seen == n_tokens_evicted + (n_tokens_cached - n_centroids)

For hard-eviction methods (`StreamingLLM`, `use_clustering=False`)
`n_centroids == 0` and this reduces to `evicted + cached == seen`.
`n_tokens_seen` is exposed as an attribute, not a `get_stats()` key, because
the contract fixes `get_stats()` to exactly four keys.

## A5. RoPE re-positioning is a rotation by a delta
Cached keys are already RoPE-rotated. Moving a key from position `p` to `p'`
is a rotation by `p' - p`, so centroids are built by: un-rotating each member
to position 0, taking the attention-weighted mean, then rotating to the
target position chosen by `rope_position_mode`. This is exact for RoPE types
whose `inv_freq` does not depend on sequence length — `default`, `linear`,
`llama3`. `src/rope_positions.py` asserts the model's rope type is one of
these and raises otherwise (`dynamic`/NTK-by-length would invalidate the
delta trick).

## A6. Batch size 1
All eval runs use batch size 1 with no padding. With a compressed cache a 2D
padding mask of full sequence length no longer aligns with the cache's
`kv_length`; batch-1 unpadded generation makes that mask all-ones and
therefore harmless. `eval/run.py` enforces this.
