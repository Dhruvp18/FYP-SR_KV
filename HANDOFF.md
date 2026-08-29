# SR-KV — project state and the remaining pipeline

This is the working document for anyone (human or agent) picking the project
up. It says exactly what is built, what is left, and how to execute the rest on
Kaggle without a browser.

Repo: <https://github.com/Dhruvp18/FYP-SR_KV>

---

## Part 1 — What is done

**Phases 0–3 are complete and verified. 115 tests pass on CPU, with no GPU and
no downloads.** Run `python -m pytest -q` to confirm before touching anything.

### The code

| Area | Files | State |
|---|---|---|
| Cache interface + bookkeeping | `src/caches/base.py` | Passthrough cache, per-slot positions/weights/centroid flags, `get_stats()` accounting |
| StreamingLLM baseline | `src/caches/streaming_llm.py` | Sinks + sliding window, adapted from the reference |
| SnapKV baseline | `src/caches/snapkv.py` | Observation-window voting + hard eviction, written independently on purpose |
| **The unified class** | `src/caches/sr_kv.py` | One class, three conditions, via `use_recency` / `use_clustering` |
| Scoring | `src/scoring.py` | Windowed attention + recency decay, causal by *position* not by layout |
| Clustering | `src/clustering.py` | Fixed-k spherical k-means, attention-weighted centroids |
| RoPE handling | `src/rope_positions.py` | Three position conventions, rotate-by-delta |
| Attention hook | `src/attn_patch.py` | Registers an SDPA-delegating implementation to get a post-attention hook |
| Model loading | `src/models.py` | Qwen2.5/Llama-3.2, bf16 with automatic 4-bit fallback, no silent CPU fallback |
| Eval harness | `eval/run.py`, `eval/memory.py`, `eval/niah.py`, `eval/longbench.py` | One CLI, resumable, shardable |
| Checkpointing | `scripts/checkpoint_utils.py` | Per-task fsync, resume, shard partitioning |
| Results validation | `scripts/check_results.py` | Completeness, ablation sanity, and the seven **phase gates** |
| Figures | `scripts/make_plots.py` | Pareto, NIAH heatmaps, ablation bars, RoPE comparison |
| Kaggle automation | `scripts/kaggle_kernel.py` | generate / push / status / pull / run |

### What the tests actually prove

Not "it runs" — these are the load-bearing claims:

1. **The no-op cache is bit-identical to `DynamicCache`.** `torch.equal` on
   generated token ids. If the passthrough changed the model's output, every
   later comparison would be against a moving baseline.
2. **`SRKVCache(use_clustering=False)` retains the exact same token set as the
   independently-written SnapKV class** — per layer, per head. This is the
   proof that the unified class genuinely subsumes the baseline, checked
   against an implementation that could not have drifted with it.
3. **Token accounting conserves**: `seen == evicted + (cached − centroids)`.
4. **The cache never exceeds its budget at any point during generation**, not
   just at the end.
5. **All layers keep identical cache lengths** — required because HF builds one
   causal mask per forward pass and shares it across layers.
6. **The RoPE rotate-by-delta identity holds** to 1e-5, which is what makes
   centroid re-positioning valid at all.
7. **Resume works**: kill after N of M tasks, re-run, exactly M−N more happen,
   no duplicates, and a half-written final line is survived.
8. **Shards are exhaustive and non-overlapping** for every shard count tested.
9. **`use_recency` actually changes which tokens survive** — an ablation switch
   that did nothing would prove nothing.

### Design decisions that are binding

Read `CLAUDE.md` before changing any cache. The short version:

- **Compression runs in a `post_attention` hook, not inside `update()`.**
  `update()` is called before attention and never sees `query_states`, which is
  exactly what SnapKV-style scoring needs. Side benefit: peak prefill KV memory
  is `(L−1)·budget + seq_len` rather than `L·seq_len`.
- **Fixed-k clustering only.** A cosine-*threshold* rule gives a data-dependent
  cluster count, which breaks the uniform-cache-length invariant. The threshold
  version exists as analysis-only code with a test documenting why.
- **`transformers>=5.0` is required.** The code targets the real 5.x
  `Cache`/`AttentionInterface` API; `src/compat.py` raises a clear error on
  anything older.
- **Centroids are re-clustered rather than kept.** Without this, summary slots
  ratchet upward during decoding until most of the budget is lossy summaries.
- **The three scored conditions must stay one class.** If you ever find
  yourself writing a second implementation of centroid-merge, stop — that
  destroys the ablation.

### What is provisional

`configs/defaults.yaml` carries a **placeholder** RoPE mode with
`rope_position_mode_frozen: false`. Until Phase 4 runs, every SR-KV number is
provisional. `scripts/freeze_rope_mode.py` refuses to freeze a winner if all
three modes come out at chance.

---

## Part 2 — What is remaining

All of it needs a GPU. Nothing below is a code task; they are experiment runs
with pass conditions.

| Phase | Command | Gate | Pass condition |
|---|---|---|---|
| 1 | `make phase1 phase1-4bit` | `make gate1` | Uncompressed NIAH at 512 tokens > 0.9 |
| 2 | `make phase2` | `make gate2` | StreamingLLM underperforms SnapKV at mid-depths |
| 3 | `make phase3` | `make gate3` | Conservation + budget hold across a full 8k run |
| 4 | `make phase4` → `make freeze-rope` | `make gate4` | All three modes ran; best is above chance |
| 5 | `make phase5 phase5-longbench` | `make gate5` | Grid complete; SR-KV not below **both** ablations |
| 6 | `make phase6-sweep phase6-3b` | `make gate6` | 3B completes with no OOM and the same JSON schema |
| 7 | `make phase7` | `make gate7` | Four figure families exist, none an empty canvas |

Every gate exits `0` = proceed, `1` = failed or incomplete, `2` = needs a human
look. **Gate 5 returning 2 is not a bug to hide** — it means SR-KV lost to both
of its ablations, which is either a real defect or a real negative finding, and
either way it belongs in the report.

### Rough cost

Phase 1 is minutes. Phase 2 is under an hour. Phase 4 is 1–2 hours. **Phase 5
is the expensive one** — prefill dominates, so the 16k cells cost more than
everything else combined. Run 2k/4k/8k first and add 16k once the rest is done.
Budget against roughly 30 GPU-hours/week.

---

## Part 3 — Running it on Kaggle from an agent

### Why this works without a browser

Kaggle notebooks pushed through the API run **detached** — the same thing the
UI calls "Save & Run All". So the loop is:

```
generate kernel → push → poll status → pull output → run gate locally → next phase
```

`scripts/kaggle_kernel.py run --phase N` does all five and **exits with the
gate's status**, so the decision to continue is made on evidence rather than on
the kernel merely having finished.

### One-time setup

1. `pip install kaggle`
2. Kaggle → Settings → API → **Create New Token**, save to `~/.kaggle/kaggle.json`,
   `chmod 600`.
3. Phone-verify the Kaggle account (required for GPU and internet).
4. Push this repo to GitHub (already done if you are reading this there).

### The commands

```bash
# generate the kernel directory for a phase (writes kaggle/phaseN/)
python scripts/kaggle_kernel.py generate --phase 1 --user <kaggle-username>

# push, poll to completion, pull results, run the gate — the one an agent calls
python scripts/kaggle_kernel.py run --phase 1 --user <kaggle-username>

# individual steps, if you want them
python scripts/kaggle_kernel.py push   --phase 1
python scripts/kaggle_kernel.py status --phase 1 --user <kaggle-username>
python scripts/kaggle_kernel.py pull   --phase 1 --user <kaggle-username>

# see exactly what would be called, without calling it
python scripts/kaggle_kernel.py run --phase 5 --user <kaggle-username> --dry-run
```

Phase N's kernel automatically mounts phases 1..N−1 as `kernel_sources`, so
their results land under `/kaggle/input` and the run resumes instead of redoing
finished work.

### If you have a Kaggle MCP server connected

A Kaggle MCP server wraps the same Kaggle API. Tool names differ between
servers, so the agent should first call `ListMcpResourcesTool` / inspect the
available tools and map them onto these four operations:

| Operation | Kaggle CLI equivalent | What the MCP tool is usually called |
|---|---|---|
| Push a notebook | `kaggle kernels push -p <dir>` | `push_kernel`, `create_kernel`, `kernels_push` |
| Check run status | `kaggle kernels status <user>/<slug>` | `kernel_status`, `get_kernel_status` |
| Download output | `kaggle kernels output <user>/<slug> -p <dir>` | `kernel_output`, `get_kernel_output` |
| List kernels | `kaggle kernels list -m` | `list_kernels` |

The generated `kaggle/phaseN/` directory is what any of them consume: a
`kernel-metadata.json` plus the `.ipynb` it names. **If the MCP server has no
push capability, fall back to the CLI** — `scripts/kaggle_kernel.py` needs
nothing but `kaggle` on PATH.

Note the MCP server is not required. It is a convenience layer over the same
API; the fallback path is always the CLI.

### What the generated notebook does

Thin by design — it clones the repo, installs `transformers>=5.0`, restores
prior results from `/kaggle/input`, runs the CPU test suite, then calls the
phase's Makefile targets and its gate. **No project logic lives in the
notebook**; there is a test asserting that.

---

## Part 4 — Prompts to give your agent

These are written to be pasted as-is. Each assumes the agent has shell access
to a clone of the repo.

### Run one phase

> Run SR-KV Phase 1 on Kaggle. My Kaggle username is `<username>`.
> Use `python scripts/kaggle_kernel.py run --phase 1 --user <username>`.
> When it finishes, report the gate output verbatim. If the gate exits non-zero,
> do not proceed to the next phase — diagnose it and tell me what you found.

### Run the whole remaining pipeline

> Execute the remaining SR-KV pipeline on Kaggle. Kaggle username `<username>`.
>
> For each phase N in 1,2,3,4,5,6,7:
>   1. `python scripts/kaggle_kernel.py run --phase N --user <username>`
>   2. If the gate exits 0, commit any new files under `results/`,
>      `figures/`, and `configs/` with message "phase N results", push, and
>      continue to phase N+1.
>   3. If the gate exits 1, STOP. Report the gate output and your diagnosis.
>      Do not continue to later phases — every later phase depends on this one.
>   4. If the gate exits 2, STOP and report. Exit 2 means SR-KV lost to both of
>      its ablations, which needs a human decision, not a retry.
>
> After Phase 4 passes, confirm `configs/defaults.yaml` has
> `rope_position_mode_frozen: true` and commit it before starting Phase 5.
>
> Never edit files under `src/` to make a gate pass. If a gate fails, the
> finding is the deliverable — report it.

### Resume after an interrupted run

> The SR-KV Phase 5 Kaggle kernel was killed mid-run. Re-run
> `python scripts/kaggle_kernel.py run --phase 5 --user <username>` — it resumes
> from `results/*.jsonl` rather than starting over. Afterwards run
> `make check-complete` and tell me if any cells are still missing.

### Diagnose a failing gate

> `make gate2` is failing for SR-KV. Read `scripts/check_results.py::gate_phase2`
> to see what it checks, read the relevant records in `results/`, and tell me
> whether this is a harness bug or a real property of the method. Do not change
> any thresholds to make it pass.

### Guardrails worth including in any prompt

- **Never loosen a gate threshold to make a phase pass.** The thresholds encode
  what the project claims; moving them changes the claim.
- **Never edit `src/` to make an experiment come out better.** If a result is
  bad, that is the result.
- **Always run `python -m pytest -q` before pushing a kernel.** Two CPU minutes
  beats discovering a broken commit after an hour of GPU quota.
- **Do not silently reduce scope on OOM.** Switch to `--precision 4bit`, record
  the change, and say so in the report.

---

## Part 5 — Fast reference

```bash
python -m pytest -q                      # 115 tests, CPU, ~2 min
make help                                # every target, with its gate
make test                                # same as pytest
python eval/run.py --tiny --allow_cpu \
  --method sr_kv --task niah --context_len 384 --depths 50 \
  --n_samples 1 --output results/smoke.json    # harness smoke test, no GPU
python scripts/check_results.py gate --phase 5 --model qwen2.5-1.5b
python scripts/kaggle_kernel.py run --phase 5 --user <username>
```

Docs: `README.md` (what and why), `CLAUDE.md` (the binding contract),
`KAGGLE.md` (manual/browser workflow), this file (state + automation).
