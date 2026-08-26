# SR-KV — summarise-and-retain KV cache eviction

A training-free KV cache compression policy for decoder-only LLMs. Instead of
permanently discarding low-importance tokens, SR-KV **clusters them into
centroid summary vectors**, so pruned context keeps a partial semantic
presence in the cache.

The contribution is the *combination*, and the codebase is arranged so that the
combination can be taken apart:

- SnapKV-style **observation-window attention scoring** (cheap: never
  materialises a full `[seq_len, seq_len]` attention matrix, which is what
  makes H2O impractical on a 16GB T4),
- an explicit **recency decay** term,
- **centroid clustering** of the remainder rather than hard eviction,
- under a single token budget, with a **controlled study of how to assign RoPE
  positions to merged centroids** — an aspect prior merge-based work leaves
  largely unaddressed.

```
importance_i = alpha * attn_score_i + beta * exp(-lambda * (t_now - t_i))
```

## The design constraint everything else follows from

The three scored conditions — SnapKV-style hard eviction, centroid-merge
without recency, and full SR-KV — are **one class with two boolean flags**
(`src/caches/sr_kv.py`), not three implementations:

| Condition | `use_recency` | `use_clustering` |
|---|---|---|
| SnapKV-style hard evict | False | False |
| Centroid merge (no recency) | False | True |
| **SR-KV (full)** | True | True |

If they were separate classes, a difference in accuracy between them would
measure implementation drift as much as it measures the recency term. Keeping
them unified is what makes the ablation mean anything, and
`tests/test_cache_correctness.py` enforces it — including a test that
`SRKVCache(use_clustering=False)` retains the *exact same token set* as the
independently written SnapKV baseline, per layer and per head.

StreamingLLM stays a separate class: it is structural (sinks + sliding window),
not scored, so forcing it into the same flags would misrepresent it.

## Layout

```
src/
  caches/base.py          SRKVCacheBase — passthrough + slot bookkeeping
  caches/streaming_llm.py sinks + sliding window (reference-adapted)
  caches/snapkv.py        observation-window scoring + hard eviction (reference-adapted)
  caches/sr_kv.py         ONE class: hard-evict / centroid-merge / full SR-KV
  scoring.py              windowed attention + recency decay
  clustering.py           fixed-k spherical k-means, attention-weighted centroids
  rope_positions.py       three centroid position conventions + rotate-by-delta
  attn_patch.py           the post-attention hook (why: see CLAUDE.md A2)
  models.py               Qwen2.5 / Llama-3.2 loading, 4-bit fallback
eval/
  run.py                  the single CLI entrypoint for every experiment
  niah.py, longbench.py   task builders and metrics
  memory.py               the only place model.generate() is called
scripts/
  checkpoint_utils.py     incremental save/resume + shard partitioning
  make_plots.py           every report figure, from results/
  check_results.py        completeness + the SR-KV-vs-ablations sanity assertion
  freeze_rope_mode.py     Phase 4: write the measured winner into configs/
```

`CLAUDE.md` is the binding interface contract plus the implementation
invariants (uniform cache length, where eager attention is allowed, the
`get_stats()` accounting law, the RoPE delta identity). Read it before changing
any cache.

## Quickstart

```bash
pip install -r requirements.txt        # needs transformers>=5.0
python -m pytest -q                    # 70 tests, CPU only, no downloads

python eval/run.py --method sr_kv --model qwen2.5-1.5b --task niah \
  --context_len 8192 --budget 0.3 --output results/srkv_8k.json
```

Every run is resumable: results are fsynced to `<output>.jsonl` per task, and
re-running the same command skips completed work. `--shard i --num_shards n`
partitions a sweep across workers.

**No GPU?** `--tiny --allow_cpu` runs the whole harness against a
randomly-initialised 2-layer Qwen2 on CPU. Outputs are meaningless; the plumbing
is real, which is what the tests check.

**Running on Kaggle: see [KAGGLE.md](KAGGLE.md)** for the phase-by-phase
commands, session-limit survival, and troubleshooting.

## Status

| Phase | State |
|---|---|
| 0 — scaffolding, contract | complete |
| 1 — model loading, no-op cache, eval harness | code complete; CPU-verified. GPU sanity run pending |
| 2 — StreamingLLM + SnapKV baselines | code complete; budget/conservation verified. GPU trend check pending |
| 3 — scoring, clustering, RoPE modes, unified `SRKVCache` | code complete; all unit + equivalence tests pass |
| 4 — RoPE position ablation | needs GPU. `make phase4` then `freeze_rope_mode.py --apply` |
| 5 — factorial matrix + full sweep | needs GPU. `make phase5` |
| 6 — hyperparameter sweep + 3B transfer | needs GPU. `make phase6-3b` |
| 7 — figures | code complete; `make report_artifacts` runs against real results |

`configs/defaults.yaml` currently carries a **placeholder** RoPE mode with
`rope_position_mode_frozen: false`. Until Phase 4 has run, every SR-KV number is
provisional, and `freeze_rope_mode.py` refuses to freeze a winner if all three
modes come out at chance.

## Scope, stated plainly

**In scope**: Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct (bf16, or 4-bit
where VRAM demands it), contexts up to 8k–16k, a 3–4 task LongBench subset,
single-needle NIAH with a depth sweep.

**Out of scope**, deliberately: 7B+ models, 32k+ contexts, multi-needle NIAH,
vLLM/production serving integration, and a full H2O reproduction. H2O needs
cumulative attention over the whole sequence, i.e. eager
`[seq_len, seq_len]` attention at prefill, which does not fit the compute this
project is built for — published numbers are cited instead. Not reimplementing
it is a decision, not an oversight.

**Target**: ≥50% KV cache reduction within 1–2% of the uncompressed baseline,
*on the models and context lengths listed above only*. Nothing here licenses a
claim about longer contexts that were never tested.

**Fallback if compute runs short**: Qwen2.5-0.5B and 1.5B at 4k/8k. If that
happens it goes in the limitations section explicitly — an honest constraint
beats a quiet scope reduction.

## References

- StreamingLLM — Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (mit-han-lab/streaming-llm)
- SnapKV — Li et al., *SnapKV: LLM Knows What You are Looking for Before Generation* (FasterDecoding/SnapKV)
- H2O — Zhang et al., *Heavy-Hitter Oracle for Efficient Generative Inference of LLMs* (cited, not reproduced)
- LongBench — Bai et al., *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*
