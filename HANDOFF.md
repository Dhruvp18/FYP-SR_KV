# SR-KV — project state and the remaining pipeline

This is the working document for anyone (human or agent) picking the project
up. It says exactly what is built, what is left, and how to execute the rest
on Kaggle without a browser.

Repo: <https://github.com/Dhruvp18/FYP-SR_KV>

---

## READ THIS FIRST — all phases complete, 2026-09-19

**Phases 0–7 are done.** Every grid is complete, every gate has been run, and
the figures are generated. The remaining work is writing it up, plus one
optional experiment described at the end.

### Gate status

| phase | gate | result |
|---|---|---|
| 1 | gate1 | PASS |
| 2 | gate2 | PASS |
| 3 | gate3 | PASS |
| 4 | gate4 | exit 2 — no significant difference between RoPE modes (p=1.000) |
| 5 | gate5 | **exit 2 — FLAG: SR-KV below both ablations** |
| 6 | gate6 | PASS |
| 7 | gate7 | PASS |

Exit 2 is "a human must look", not failure. Both 2s are real findings.

### Data on disk, all zero-error

    phase5_niah_qwen2.5-1.5b        180/180   (budget 0.2)
    phase5_niah_full_qwen2.5-1.5b    45/45
    phase5_longbench_qwen2.5-1.5b   500/500
    phase6_sweep_*                  270/270   (alpha/beta, 9 cells)
    phase6_niah_qwen2.5-3b          240/240
    phase6_niah_full_qwen2.5-3b      30/30
    figures/                         14 figures

### The headline result, stated plainly

SR-KV's central claim is that clustering evicted tokens into centroids
preserves more than dropping them. **The data does not support it.**
`centroid_merge` (clustering, no recency) against `snapkv_unified` (hard
eviction), the exact one-flag comparison the whole design exists to make:

    NIAH 1.5B @ budget 0.2     1.000  vs  1.000
    NIAH 3B    (both budgets)  0.983  vs  0.983
    LongBench 1.5B @ 0.3       0.248  vs  0.249

Identical to three decimals on NIAH, within 0.001 on LongBench. Clustering
neither helps nor hurts anywhere measured.

The recency term is worse than neutral. `sr_kv` is the weakest scored method
in all three evaluations, and gate5 flags it below both of its own ablations
at budget 0.2: 0.733 against 1.000 and 1.000. The mechanism is established,
not guessed — accuracy rises monotonically with needle depth:

    depth      0     25     50     75    100
    sr_kv  0.667  0.333  0.667  1.000  1.000

NIAH places the needle uniformly at random, so a recency prior evicts
early-placed needles. `centroid_merge` and `snapkv_unified` have recency off
and are unaffected. Phase 6's beta sweep predicted exactly this gradient, and
the factorial reproduced it independently at a fixed config.

This is a clean negative result, not a broken experiment: 1,000+ records,
zero errors, conservation and budget invariants hold, generations checked for
degeneration. The ablation is one class with two boolean flags, so it measures
the mechanism rather than implementation drift — which is precisely what makes
the null trustworthy.

### What would give the thesis a fair last shot

The one condition never tested: **LongBench at a tighter budget (0.1–0.15).**
Everything measured so far sits where compression is nearly free — LongBench
at 0.3 has hard eviction at 0.249 against uncompressed 0.255, so there is
almost nothing for merging to recover. NIAH cannot test the claim at all by
construction: a centroid is an average and cannot reproduce an exact token.
If centroid-merging has an advantage anywhere, it is where hard eviction
starts losing real information. `make phase5-longbench BUDGET=0.1` is the run.

### Scope reductions, both hardware-forced and both recorded

- **No 16384 on 1.5B.** Measured bf16 peaks: 5.11 GiB at 4096 (n=210), 10.96
  at 8192 (n=603), so 16384 lands near 16.8 against a 14.56 GiB T4.
- **No 8192 on 3B.** bf16 OOM'd (4.05 GiB requested, 3.91 free); 4-bit fits
  but was unreliable.

Compression does not rescue either: the peak is prefill attention over the
full sequence, which every method pays before any eviction runs (CLAUDE.md
A2). So the honest scope is **NIAH factorial at 2k–8k on 1.5B, transfer at
2k–4k on 3B**, and 16k — where KV compression matters most — is untested for
hardware reasons, not measured and found wanting.

### How to run this on Kaggle without losing days

Measured throughput, qwen2.5-1.5b bf16: ~3s/task at 2048, ~7s at 4096, ~20s
at 8192. Flat within a context. A cumulative rate that appears to fall is just
the grid walking up the context ladder — it is not a stall, and twice this
session a perfectly healthy run was cancelled because of that misreading.

- **One kernel at a time.** Two concurrent GPU sessions banked zero tasks each.
- **Few large `eval/run.py` invocations, not many small ones.** Per-chunk
  isolation, added as a safety measure, was itself the cause of four
  consecutive zero-progress cycles; removing it banked 157 tasks in one go.
- **Tee everything to a file under /kaggle/working.** Kaggle truncates the
  kernel log at ~120s, so a stall after that point leaves no evidence at all.
  A persistent log is what finally made throughput visible.
- `SRKV_WATCHDOG_SECONDS=300` turns a hung task into a stack trace.
- Assert that any mounted data source exists. `modelDataSources` is silently
  ignored by the Kaggle MCP server: a probe with four ref spellings reported
  `/kaggle/input: []`, and a notebook that printed "mount exists: False" and
  carried on cost a whole session.

### Bugs fixed this session

Five instances of one pattern — **two experiments sharing a directory get
silently averaged** — plus three others:

16. **gate4 counted Phase 6's sweep as extra RoPE samples.** attn_weighted
    moved 0.770 → 0.700 and p 1.000 → 0.264 on a 100-vs-370 comparison, with
    nothing re-run. `freeze_rope_mode.py` had it too. Fixed with a scoring-knob
    guard and a `run_key` guard sharing one resolver.
17. **gate6 could never have passed** — it required `full` at budget=1.0 while
    runs recorded whatever `--budget` was passed.
18. **`precision` was missing from `run_key`**, so 4-bit records would have
    marked bf16 tasks done and reported a grid never measured.
19. **Two gate6 tests and one gate5 test rotted** when grids narrowed — each
    hardcoded a context the gate no longer inspects. They now read the
    constant.
20. **The figures averaged NIAH with LongBench.** Exact-match 0/1 blended with
    F1/ROUGE: the budget=0.3 snapkv bar mixed 9 NIAH records at 1.000 with 100
    LongBench at 0.249 and plotted 0.311, contradicting gate5. Figures now
    split by task, and the ablation skips budgets missing conditions.
21. **`choose_precision()` under-estimates**, confirmed a fourth time: it sizes
    weights + KV + 2.5 GB and never models the attention activation.

The lesson worth carrying: **a gate passing is not evidence the figure is
right, and a figure rendering is not evidence the number is right.** Every one
of these was found by comparing two views of the same data and noticing they
disagreed.

---

## BACKGROUND — the 2026-09-18 session, still accurate for Phases 1-5

You are picking up an FYP project cold. This section is written to be
complete on its own — you should not need to ask "what happened before this"
to start working. It has been produced entirely by an AI agent (Claude Code)
driving `scripts/kaggle_kernel.py` end to end: pushing Kaggle kernels,
polling them, pulling results, running gates, and finding/fixing real bugs
along the way, under the direction of Chaitra (the project owner). If you're
also working with an agent, hand it this whole file. If you're driving by
hand, every command below is copy-pasteable.

### What this project is, in one paragraph

SR-KV is a training-free KV-cache compression policy for decoder-only LLMs.
Instead of just dropping low-importance tokens from the cache (like
StreamingLLM/SnapKV do), it clusters them into centroid summary vectors, so
pruned context keeps a partial semantic presence. The three conditions that
matter — hard eviction, centroid-merge, and full SR-KV (merge + recency) —
are deliberately **one class with two boolean flags**
(`src/caches/sr_kv.py`), not three separate implementations, so an ablation
comparing them measures the actual mechanism, not implementation drift. Read
`README.md` for the full pitch and `CLAUDE.md` for the binding interface
contract before changing any cache code — both are short and load-bearing.

### Who's who

- **Chaitra** (`chaitrasamant` on Kaggle) — the project owner, directing this
  work.
- **Dhruv** (`dhruvp18` on GitHub) — owns the GitHub repo
  (`https://github.com/Dhruvp18/FYP-SR_KV`), teammate, has his own Kaggle
  account already set up from an earlier handoff.
- **You** — new to this, need your own Kaggle account (setup steps below).
- Kaggle kernel names throughout follow the pattern `<user>/sr-kv-phaseN`.
  The **committed** `kaggle/phaseN/` files in the repo are generated for
  `dhruvp18` by convention; anyone running their own phase regenerates them
  for their own username first (commands below) and should revert that local
  regeneration afterward (`git checkout -- kaggle/phaseN/`) so the committed
  files don't drift to someone else's identity.

### Set up your own Kaggle account — do this first

```bash
# Kaggle -> Settings -> API -> Create New Token (current format is a bearer
# token like KGAT_..., not the old kaggle.json - both work with the current
# CLI, this is just what you'll be issued)
mkdir -p ~/.kaggle
printf '%s' "<YOUR_TOKEN_HERE>" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

pip install kaggle
kaggle --version
# if "command not found": the install went to a user Scripts dir not on
# PATH. Find it and prepend it for your shell session, e.g. on Windows:
#   export PATH="$HOME/AppData/Roaming/Python/Python312/Scripts:$PATH"
# or invoke everywhere below as `python -m kaggle` instead of `kaggle`.

kaggle kernels list -m   # confirms auth; lists your own kernels (empty at first)
```

Also: **phone-verify your Kaggle account** (Settings → Phone Verification).
Without this, GPU + internet-enabled kernels fail outright. Your Kaggle
username goes in every `--user` flag below.

**Hard limit worth knowing up front**: this account tier caps you at **2
concurrent GPU kernel sessions**. Pushing a 3rd while 2 are already running
gets rejected with `Kernel push error: Maximum batch GPU session count of 2
reached.` — not a bug, just capacity. If you and Chaitra/Dhruv both want work
running simultaneously, you need it spread across *your own* accounts, or
you wait for a slot to free up.

### Where things actually stand (real Tesla T4 runs, not aspirational)

**Phase 1 — PASS.** bf16 sanity (qwen2.5-1.5b, ctx=512): 1.000 accuracy, 10
samples. 4-bit fallback (qwen2.5-3b): 1.000 accuracy, 3 samples. Zero errors.

**Phase 2 — PASS.** `full`=1.000, `snapkv`=1.000, `streaming_llm`=0.333 at
mid-depths (ctx=4096, budget=0.3) — the expected shape: StreamingLLM's
4-token sink is far smaller than the needle sentence, so it only survives
when the needle lands in its always-kept recent window. 75 records, zero
errors.

**Phase 3 — PASS.** Conservation and budget invariants hold across a real 8k
run for `sr_kv`/`centroid_merge`/`snapkv_unified`. All three score 1.000 at
every depth; `sr_kv`/`centroid_merge` both form exactly 309 centroids
consistently. 27 records, zero errors.

**Phase 4 — DONE, with a real conclusion.** The RoPE centroid position-mode
ablation (`latest`/`earliest`/`attn_weighted`) ran properly-powered sweeps at
two budgets and found **no significant difference between the three
conventions** — at the discriminating budget (0.2, found via a separate
scan), n=100/mode: latest=76/100, earliest=76/100, attn_weighted=77/100,
Fisher exact p=1.000. `configs/defaults.yaml` has
`rope_position_mode: attn_weighted` with `rope_position_mode_frozen: true`
— frozen as a **principled default** (it weights cluster members by
attention score, consistent with SR-KV's scoring design elsewhere), explicitly
**not** an empirical winner. The evidence field says this directly. Do not
re-open this without a real reason; two properly-powered experiments already
answered it.

**Phase 5 — LongBench half: DONE, verified, committed.** NIAH half: RUN,
but saturated and its records were never committed. Chaitra reports running
the NIAH factorial and getting 1.000 everywhere, which is why the project
pivoted to LongBench. Committed Phase 3 data corroborates it exactly: at
budget=0.3, ctx=8192, `sr_kv`/`centroid_merge`/`snapkv_unified` all score
1.000 at depths 0/50/100 (n=3 each). So `gate5` reporting INCOMPLETE is
accurate about the *files* and misleading about the *work*: the grid was
measured, it just measured a ceiling, and nothing was kept. LongBench needed five separate real-bug fixes to run cleanly (see
"What actually happened" below), and even the first clean 500/500 run turned
up a genuine correctness bug (repetition collapse in `sr_kv`/`centroid_merge`
on long generations, fixed via `recompress_slack=16`). The *re-verified* run
is clean: 500/500 records, zero errors, **zero degenerate generations across
every method and every task** (checked by reading actual generated text, not
just accuracy — see bug #13/#14 below for why that check matters here).
Real comparison (overall mean, all 4 tasks): `full`=0.255,
`snapkv_unified`=0.249, `centroid_merge`=0.248, `sr_kv`=0.244,
`streaming_llm`=0.215 — centroid-merging is essentially tied with hard
eviction, not a clean win, reported honestly rather than spun. `gate5`
correctly reports incomplete (exit 1): that gate checks the NIAH grid, which
hasn't been run — LongBench being done doesn't satisfy it.

**Phase 6 — alpha/beta sweep: DONE, verified, committed** (at `budget=0.2`,
not the Makefile's default 0.3 — see why below). Clean, monotonic result:
`beta=0` (no recency weighting) hits perfect NIAH accuracy at any `alpha`;
accuracy degrades as `beta` increases (down to 0.367 at alpha=0.5/beta=0.6).
Makes sense mechanistically — NIAH's needle depth is randomized, so a
recency bias fights against retrieving an early-placed needle. `phase6-3b`
(transfer to qwen2.5-3b) has **not** been run and has a known, unaddressed
OOM risk — `make phase6-3b-scan` first (see "What's actually left to do").
**Also**: `gate6` had a real false-positive bug, found and fixed while
checking this data (see bug #15 below) — don't trust a `gate6` PASS from
before this fix without rerunning it.

**Phase 7 — not started.** Don't run it until Phase 5's NIAH half and
Phase 6's 3B transfer have real, complete data; it'll just render figures
from whatever's in `results/`, gaps and all.

### Kaggle account note

Both Phase 5 and Phase 6 above ran under `chaitrasamant`'s Kaggle account —
private kernels, so you can't query or pull them with your own credentials.
That's fine, their results are already verified and committed to `main` (see
commit log). This matters for what *you* run next: your own kernels will be
under your own username, and this account's **2-concurrent-GPU-session cap**
(see setup section above) applies per-account, so you're not competing with
`chaitrasamant`'s quota, just your own.

```bash
git pull origin main
git log --oneline -20   # see exactly what's landed
python -m pytest -q     # always, before touching anything
```

### What actually happened this session (why the code looks the way it does)

Fifteen distinct real bugs were found and fixed, in roughly this order.
Skim this before touching `src/`, `eval/`, or `scripts/kaggle_kernel.py` —
several of these look like they'd be easy to "fix" a different, more obvious
way, and that obvious way was already tried and shown to be wrong.

1. **Per-layer position corruption (`src/caches/base.py`)** — the most
   consequential fix. `transformers>=5.0`'s Qwen2/Llama attention calls
   `Cache.update()` with **no `cache_kwargs` at all** (confirmed by reading
   the installed modeling code). The cache's own position-counter fallback
   is therefore not a rare corner case — it's the *only* path that ever
   runs, for every layer, on every model. A single shared `_pos_counter`
   int, bumped only at `layer_idx==0`, let every other layer in the same
   forward pass read an already-advanced counter, corrupting the RoPE delta
   used to reposition merged centroids for every layer past 0. Fixed: a
   per-layer counter dict. This never affected any GPU results, since
   nothing had run on GPU before this was found and fixed.

2. **Kaggle kernel slug bug (`scripts/kaggle_kernel.py`)** — Kaggle derives
   a pushed kernel's *live* slug from `kernel-metadata.json`'s `title`, not
   `id`. A descriptive title silently diverged from the slug every
   `status`/`pull` call assumed. Fixed: title is now literally the slug.

3. **Llama-3.2-3B → Qwen2.5-3B pivot** — Llama-3.2-3B is gated on HF and
   blocked Phase 1 outright. `microsoft/Phi-3.5-mini-instruct` was
   considered and **ruled out after checking its actual HF config**:
   `rope_scaling.type == "longrope"`, exactly the length-dependent scheme
   `rope_positions.py` rejects. Qwen2.5-3B-Instruct verified safe
   (`rope_scaling: None`, 32768 native context) before adopting it as the
   default `MODEL3B` everywhere. Trade-off for the writeup: Phase 6 now
   tests generalization across scale within Qwen, not across architectures.
   `llama3.2-3b` remains a supported value for whenever HF access lands.

4. **No `--force` on Kaggle pulls** — the Kaggle CLI skips any file whose
   local copy "looks newer," so re-pulling the same phase silently re-serves
   the *previous* run's results. Caught because a pulled file's own
   `metadata.argv` still said the old budget. Fixed: pulls always `--force`
   now. Still, **always `rm -rf .kaggle_output/phaseN` before re-pulling the
   same phase** — belt and suspenders.

5. **Gate/freeze had no statistical rigor
   (`scripts/check_results.py`, `scripts/freeze_rope_mode.py`)** — both
   would silently average across different budgets if two sweeps' result
   files existed under overlapping content (this literally happened: the
   gate reported a different "winner" depending on which files were on
   disk), and neither had any concept of "is this difference bigger than
   noise." Both now refuse to aggregate across budgets and run a Fisher
   exact test, refusing to name a winner at p ≥ 0.05. **Read the p-value
   line from any Phase 4-style comparison, not just the printed mean.**

6. **Phase 4-scan / real-sweep task_id collision** — a 3-sample diagnostic
   scan and the real 100-sample sweep measured the identical
   (method, budget, mode, depth, sample_idx) cell, and `load_records`'
   directory-wide dedup let the scan's value silently overwrite the sweep's
   on one cell, based on alphabetical file order. Fixed: diagnostic runs now
   write to `results/diagnostics/` (`load_records` globs non-recursively,
   so this is the whole fix), and `merge_outputs()` (see bug #10) had to be
   fixed too, since it was flattening that structure right back.

7. **LongBench loading: `datasets` library removed loading-script support**
   — `THUDM/LongBench` ships a `LongBench.py` loading script; current
   `datasets` versions hard-refuse those
   (`RuntimeError: Dataset scripts are no longer supported`). Not a
   permissions issue — a removed feature. Fixed by reading that script
   directly (it just downloads `data.zip` and reads `data/<task>.jsonl`)
   and replicating it via `huggingface_hub` + `zipfile` + `json`, bypassing
   `datasets.load_dataset` entirely (`eval/longbench.py`).

8. **LongBench OOM #1: no context truncation** — LongBench's native
   documents run far past this project's 8k-16k scope untruncated
   (`full` tried to allocate **43.89 GiB** on a 14.56 GiB T4 on one real
   narrativeqa document). Fixed: token-based truncation (matching how NIAH
   already does it precisely) via a new `--longbench_max_context_tokens`
   flag, keeping both ends of the document (LongBench's own convention for
   handling overlength context).

9. **LongBench OOM #2: allocator fragmentation** — `expandable_segments`
   (PyTorch's documented fix, named in the OOM message itself) got further
   (280 tasks vs 81) but didn't fully solve it, meaning it wasn't pure
   fragmentation. Set anyway (in the generated Kaggle notebook, before any
   subprocess runs) as a real, if partial, improvement.

10. **`merge_outputs()` flattening bug (`scripts/kaggle_kernel.py`)** —
    pulled `results/**/*.json` files were copied to just their basename,
    which silently undid fix #6 (diagnostic output moved to
    `results/diagnostics/`) on the very next pull. Confirmed happening
    twice. Fixed: now preserves the path relative to the results/figures
    directory component.

11. **LongBench OOM #3, root cause: no memory margin, not a leak** —
    correlating real `max_memory_allocated` against actual token counts
    showed **every** document truncated to the full 8192-token cap used
    11.9-13.5 GiB on a 14.56 GiB T4 — effectively zero margin, so whichever
    document happened to be near-cap first in the queue triggered the OOM.
    Fixes #9's fragmentation theory and #4/#10's earlier "progress" were
    coincidences of task ordering, not real fixes for this. Actual fix:
    lowered `--longbench_max_context_tokens` default 8192 → 4096 (real
    headroom, ~1.5 GiB/1000 tokens measured scaling). Also threaded the
    parameter into `run_key` so a future cap change can't silently blend
    with old data the way bug #6 did.

12. **Per-method process isolation for `phase5-longbench`** — applied
    between finding #9 and #11 as a robustness measure (one `eval/run.py`
    process per method, not one process for all five, so a fresh CUDA
    context can't inherit whatever the previous method's process was
    holding). Kept even after #11 explained the real cause, since it's a
    reasonable belt-and-suspenders and matches the per-item-loop pattern
    `phase4`/`phase6-sweep` already use.

13. **Repetition collapse in `sr_kv`/`centroid_merge` on long generations**
    — the first clean 500/500 LongBench run (after fixes #7-#12) completed
    with zero task errors, but reading the *actual generated text* (not just
    accuracy) showed `sr_kv`/`centroid_merge` degenerating into repetition
    loops ("annually annually annually...", "if if if if...") on long
    outputs. Quantified: 0/25 degenerate at 32-token generations, up to
    10/25 (40%) at 512 tokens — monotonic in generation length, zero for
    every non-clustering method at every length. Root cause: `_compress()`
    fully re-clusters every candidate from scratch (fresh k-means init, not
    continuing the previous step's assignment) every time it triggers, and
    `recompress_slack=0` (the only value ever used, and the only path
    available before this session — see #14) means a tight budget during
    long decode triggers this on nearly every single new token. A 512-token
    generation meant ~500 independent full re-clusterings of the same ~155
    centroids, giving the model a KV history whose semantic identity
    reshuffles every step — not what it was trained against.

14. **`recompress_slack` never exposed on the CLI** — it existed in
    `src/caches/base.py` as a constructor parameter since Phase 0-3, but
    `eval/run.py` never let you set it, so it was always 0 in practice.
    Exposed as `--recompress_slack`, threaded into `run_key`. A targeted
    probe (gov_report only, `centroid_merge`+`sr_kv` only, n=10) confirmed
    `slack=16` and `slack=32` both fully fix bug #13 (0/10 degenerate at
    either value, repetition scores back to the clean ~0.07 baseline).
    Adopted `RECOMPRESS_SLACK=16` as the `phase5-longbench` Makefile
    default, applied **uniformly to all five methods** (not just the two
    that showed the bug) — it also changes eviction timing for
    `streaming_llm`/`snapkv_unified`, and this project's whole ablation
    design rests on changing exactly one thing between conditions.
    **Not** applied to Phase 1-4's NIAH work: NIAH's short generations
    (16-32 tokens) never showed the bug at slack=0, and those results are
    already gated.

15. **`gate6` false-positive (`scripts/check_results.py`)** — found while
    checking whether the Phase 6 sweep data satisfied `gate6`. It printed
    `PASS` using 3 completely unrelated records — Phase 1's qwen2.5-3b 4-bit
    sanity check (`method=full`, `ctx=512`) — because it only filtered by
    `model`, never checking *which* method/context those records were
    actually for. `phase6-3b` itself had never been run; the gate should
    have said so. Same class of bug as #5 (gate rigor). Fixed: `gate6` now
    runs a real completeness check against `phase6-3b`'s actual
    (method, context, depth) grid before counting anything as evidence — the
    same rigor already applied to `gate4`/`gate5`. Confirmed against real
    data: now correctly reports "INCOMPLETE: 75 missing cells."

Also worth knowing: **while Chaitra's session was between windows, a
different tool (GitHub Copilot) independently pushed a `phase5` (full NIAH)
kernel to the same Kaggle account and it failed partway through** (OOM on
the first `context_len=16384` task for `full` — consistent with a risk
already flagged in this doc). That partial, incomplete data was found,
inspected, and discarded — not committed, not used. If you see stray Kaggle
kernel versions or output you don't recognize, check the actual content
before trusting it; this project has now hit real contamination/staleness
bugs (#4, #6, #10) plus this one external-tool surprise, so "check the raw
data, not just whether something ran" is a load-bearing habit here, not
paranoia.

### What's actually left to do

1. ~~Let Phase 5 LongBench and Phase 6 sweep finish~~ — **done**, both
   verified and committed (see above). Nothing to do here.

2. **Before running `phase6-3b`** (transfer to qwen2.5-3b, not yet run): run
   `make phase6-3b-scan` first (2 cheap samples, `full` only, ctx=8192). This
   project has now hit real, hardware-driven OOMs three separate times on
   `full`/long-context combinations (bugs #8, #9, #11) — qwen2.5-3b is a
   bigger model at the same 8192-token ceiling that 1.5b was already
   borderline at, and `choose_precision()`'s "auto" mode only estimates
   KV-cache size, not the activation/logits memory that actually caused the
   earlier OOMs. Don't skip this check to save five minutes and risk losing
   hours the way LongBench did.

3. **Phase 5's NIAH half was run at BUDGET=0.3 and saturated at 1.000; do
   not simply re-run it.** That is a measured ceiling, not a missing
   experiment (Phase 3's committed records show the same thing), and
   repeating it at 0.3 buys nothing. Either re-run at a discriminating
   budget — 0.2 is where Phase 4 measured mid-range accuracy, found via a
   cheap scan — so the grid and the heatmaps say something, or record the
   saturation as the finding and stop treating gate5's NIAH requirement as
   outstanding work. What is NOT acceptable is leaving it looking unrun:
   the result exists, the files just were not kept. The strategic read from earlier in this project: NIAH
   exact-match retrieval structurally can't show centroid-merging beating
   hard-eviction (a centroid is an average, it can't reproduce an exact
   token) — LongBench (especially `gov_report`, summarization) is where the
   actual thesis (does merging preserve more than dropping) can be tested.
   Treat NIAH as "does compression break exact retrieval," not as the
   headline result.

4. **Phase 7 (figures)** only after 5 and 6 have real, complete data.
   `make phase7` / `python scripts/kaggle_kernel.py run --phase 7 --user
   <you>` (no GPU needed — `enable_gpu` is deliberately `false` for phase 7).

5. **Always, every phase**: `python -m pytest -q` before pushing any kernel.
   `rm -rf .kaggle_output/phaseN` before re-pulling the same phase. Never
   edit `src/` to make a gate pass. Never loosen a gate threshold — this
   project's gates are explicitly designed to surface negative/inconclusive
   findings (exit code 2) rather than hide them, and that's a feature, not a
   bug to route around. Check the actual JSON records and (for anything
   generation-based) the actual generated text before trusting a "PASS" —
   this session's bug list above is the concrete argument for why that habit
   matters, not abstract caution.

6. **Commit and push after every phase**, with a commit message that states
   the actual numbers and what they mean — see `git log` for the pattern
   used throughout this session. Whoever reads the log next (Chaitra, Dhruv,
   or the next person after you) should be able to reconstruct what happened
   without re-deriving it from raw JSON.

---

## Part 1 — What is done

**Phases 0–3 are complete and gate-passed on real GPU hardware. Phase 4 is
in progress.** The section above has the real numbers; this section is the
code-level summary.

### The code

| Area | Files | State |
|---|---|---|
| Cache interface + bookkeeping | `src/caches/base.py` | Passthrough cache, per-slot positions/weights/centroid flags, `get_stats()` accounting. Per-layer position counter (fixed this session). |
| StreamingLLM baseline | `src/caches/streaming_llm.py` | Sinks + sliding window, adapted from the reference |
| SnapKV baseline | `src/caches/snapkv.py` | Observation-window voting + hard eviction, written independently on purpose |
| **The unified class** | `src/caches/sr_kv.py` | One class, three conditions, via `use_recency` / `use_clustering` |
| Scoring | `src/scoring.py` | Windowed attention + recency decay, causal by *position* not by layout |
| Clustering | `src/clustering.py` | Fixed-k spherical k-means, attention-weighted centroids |
| RoPE handling | `src/rope_positions.py` | Three position conventions, rotate-by-delta |
| Attention hook | `src/attn_patch.py` | Registers an SDPA-delegating implementation to get a post-attention hook |
| Model loading | `src/models.py` | Qwen2.5 (0.5B/1.5B/3B) + Llama-3.2 aliases, bf16 with automatic 4-bit fallback, no silent CPU fallback |
| Eval harness | `eval/run.py`, `eval/memory.py`, `eval/niah.py`, `eval/longbench.py` | One CLI, resumable, shardable |
| Checkpointing | `scripts/checkpoint_utils.py` | Per-task fsync, resume, shard partitioning |
| Results validation | `scripts/check_results.py` | Completeness, ablation sanity, and the seven **phase gates** — gate 4 now runs a Fisher exact significance test, not just a mean comparison |
| Figures | `scripts/make_plots.py` | Pareto, NIAH heatmaps, ablation bars, RoPE comparison |
| Kaggle automation | `scripts/kaggle_kernel.py` | generate / push / status / pull / run, plus `--target`/`--skip-gate`/`--make-var` for diagnostic runs like `phase4-scan` |

### What the tests actually prove

126 tests pass on CPU (verified in a clean `transformers>=5.0` venv this
session — the machine's global Python has an older transformers that
breaks collection with an unhelpful `ImportError`; use a venv). Load-bearing
claims, updated from the original list:

1. **The no-op cache is bit-identical to `DynamicCache`.**
2. **`SRKVCache(use_clustering=False)` retains the exact same token set as the
   independently-written SnapKV class** — per layer, per head.
3. **Token accounting conserves**: `seen == evicted + (cached − centroids)`.
4. **The cache never exceeds its budget at any point during generation.**
5. **All layers keep identical cache lengths.**
6. **The RoPE rotate-by-delta identity holds** to 1e-5.
7. **Every layer agrees on token positions** (new this session — the
   regression test for bug #1 above: `test_position_bookkeeping_agrees_across_layers`).
8. **Resume works**: kill after N of M tasks, re-run, exactly M−N more happen.
9. **Shards are exhaustive and non-overlapping.**
10. **`use_recency` actually changes which tokens survive.**
11. **A pushed kernel's metadata `title` always resolves to its own `id`**
    (new this session — the regression test for bug #2).
12. **Pull always forces a fresh download** (new this session — bug #3).
13. **Gate 4 refuses to average across budgets, and refuses to name a winner
    at p ≥ 0.05** (new this session — bug #4).

### Design decisions that are binding

Read `CLAUDE.md` before changing any cache. The short version — unchanged
from before, still true after this session's fixes:

- **Compression runs in a `post_attention` hook, not inside `update()`.**
- **Fixed-k clustering only.**
- **`transformers>=5.0` is required** — and as of this session, confirmed to
  mean something specific: it calls `Cache.update()` with **no
  `cache_kwargs`**, so any cache subclass's own position bookkeeping is load
  bearing, not a fallback. See bug #1.
- **Centroids are re-clustered rather than kept.**
- **The three scored conditions must stay one class.**

### What is provisional

`configs/defaults.yaml` still carries the placeholder RoPE mode with
`rope_position_mode_frozen: false` — correctly, given the evidence gathered so
far (two rejected sweeps, a scan, and a full-power run in progress). Do not
freeze a mode without a passing `gate4`/`freeze_rope_mode.py` run at p < 0.05.

---

## Part 2 — What is remaining

| Phase | Command | Gate | Status |
|---|---|---|---|
| 1 | `make phase1 phase1-4bit` | `make gate1` | **PASS** |
| 2 | `make phase2` | `make gate2` | **PASS** |
| 3 | `make phase3` | `make gate3` | **PASS** |
| 4 | `make phase4` → `make freeze-rope` | `make gate4` | **exit 2** — modes not distinguishable, p=1.000; attn_weighted frozen on principled grounds |
| 5 | `make phase5 phase5-longbench` | `make gate5` | **exit 2 — FLAG** — grid complete (180+45 NIAH @ 0.2, 500 LongBench, 0 errors); SR-KV below both ablations |
| 6 | `make phase6-sweep phase6-3b` | `make gate6` | **PASS** — 270/270, 0 errors |
| 7 | `make phase7` | `make gate7` | **PASS** — 14 figures |

Every gate exits `0` = proceed, `1` = failed or incomplete, `2` = needs a
human look. **A gate returning 2 is not a bug to hide** — for gate 5 it means
SR-KV lost to both of its ablations; for gate 4 (as of this session) it can
also mean the RoPE modes weren't statistically distinguishable. Either way,
report it.

### Rough cost

Phase 1 is minutes. Phase 2 is ~15 min in practice. Phase 3 is fast. Phase 4's
full-power run (300 tasks at 8k) is taking roughly 2 hours. **Phase 5 is still
the expensive one** — run the LongBench half first per the strategic note
above; add the full 16k NIAH cells last. Budget against roughly 30
GPU-hours/week across both accounts now that there are two.

---

## Part 3 — Running it on Kaggle from an agent

### Why this works without a browser

Kaggle notebooks pushed through the API run **detached** — the same thing the
UI calls "Save & Run All". So the loop is:

```
generate kernel → push → poll status → pull output (--force!) → run gate locally → next phase
```

`scripts/kaggle_kernel.py run --phase N` does all five and **exits with the
gate's status**. Pull now always passes `--force` (fixed this session — see
bug #3 above); still `rm -rf .kaggle_output/phaseN` before re-pulling the same
phase, belt-and-suspenders.

For a diagnostic run that isn't a full phase (like the budget scan that found
`budget=0.20`), use `--target <makefile-target> --skip-gate`:

```bash
python scripts/kaggle_kernel.py generate --phase 4 --user <you> \
  --target phase4-scan --skip-gate
python scripts/kaggle_kernel.py push --phase 4
# ... poll, then force-pull ...
```

`--make-var "SAMPLES4=20"` (or any `KEY=value`) passes extra Makefile
variables through, for cases like giving Phase 4 more samples than the sweep
default without changing every other phase.

### One-time setup

1. `pip install kaggle`
2. Kaggle → Settings → API → **Create New Token**. The current token format is
   a bearer token (`KGAT_...`); store it at `~/.kaggle/access_token`
   (`chmod 600`) or export it as `KAGGLE_API_TOKEN` — both work with current
   `kaggle` CLI versions. The older `~/.kaggle/kaggle.json` format also still
   works if that's what you're issued.
3. Phone-verify the Kaggle account (required for GPU and internet).
4. Confirm `kaggle kernels list -m` works before pushing anything.

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

Phase N's kernel automatically mounts phases 1..N−1 **from the same Kaggle
account** as `kernel_sources`, so their results land under `/kaggle/input` and
the run resumes instead of redoing finished work. **This does not cross
accounts** — if Dhruv runs Phase 5 under his own username, it won't
automatically see Chaitra's Phase 1-4 kernels. That's fine; Phase 5 needs
their *code fixes* (from `git pull`), not their kernel outputs mounted.

### If you have a Kaggle MCP server connected

A Kaggle MCP server wraps the same Kaggle API. Tool names differ between
servers, so the agent should first call `ListMcpResourcesTool` / inspect the
available tools and map them onto these four operations:

| Operation | Kaggle CLI equivalent | What the MCP tool is usually called |
|---|---|---|
| Push a notebook | `kaggle kernels push -p <dir>` | `push_kernel`, `create_kernel`, `kernels_push` |
| Check run status | `kaggle kernels status <user>/<slug>` | `kernel_status`, `get_kernel_status` |
| Download output | `kaggle kernels output <user>/<slug> -p <dir> --force` | `kernel_output`, `get_kernel_output` |
| List kernels | `kaggle kernels list -m` | `list_kernels` |

The generated `kaggle/phaseN/` directory is what any of them consume: a
`kernel-metadata.json` plus the `.ipynb` it names. **If the MCP server has no
push capability, fall back to the CLI.**

### What the generated notebook does

Thin by design — it clones the repo, installs `transformers>=5.0`,
authenticates to Hugging Face if an `HF_TOKEN` Kaggle secret is present
(needed only for gated models like `llama3.2-3b`), restores prior results
from `/kaggle/input`, runs the CPU test suite, then calls the phase's
Makefile targets and its gate. **No project logic lives in the notebook**;
there is a test asserting that.

---

## Part 4 — Prompts to give your agent

### Run one phase

> Run SR-KV Phase N on Kaggle. My Kaggle username is `<username>`.
> Use `python scripts/kaggle_kernel.py run --phase N --user <username>`.
> Always `rm -rf .kaggle_output/phaseN` before re-pulling the same phase.
> When it finishes, report the gate output verbatim — including any p-value
> lines from gate 4, not just the printed mean. If the gate exits non-zero,
> do not proceed to the next phase — diagnose it and tell me what you found.
> Check the actual raw JSON records, not just the gate's exit code, before
> declaring a phase's result trustworthy — this project has already caught
> one stale-pull bug and one budget-averaging bug that a naive "gate passed"
> read would have missed.

### Resume after an interrupted run

> The SR-KV Phase N Kaggle kernel was killed mid-run. Re-run
> `python scripts/kaggle_kernel.py run --phase N --user <username>` — it
> resumes from `results/*.jsonl` rather than starting over. Afterwards run
> `make check-complete` and tell me if any cells are still missing.

### Diagnose a failing or suspicious gate

> `make gateN` is failing (or passing with a suspicious result) for SR-KV.
> Read `scripts/check_results.py::gate_phaseN` to see what it checks, read
> the actual records in `results/` (not just the gate's summary line), and
> tell me whether this is a harness bug or a real property of the method.
> Do not change any thresholds to make it pass.

### Guardrails worth including in any prompt

- **Never loosen a gate threshold to make a phase pass.**
- **Never edit `src/` to make an experiment come out better.**
- **Always run `python -m pytest -q` before pushing a kernel.**
- **Always `rm -rf .kaggle_output/phaseN` before pulling the same phase
  again** — the Kaggle CLI silently serves stale local copies otherwise.
- **Always check which `budget` a result file's records actually carry**
  before trusting a comparison across sweeps — `check_results.py`'s gates
  now refuse to auto-aggregate across budgets, but scripts you write yourself
  won't have that guard for free.
- **Do not silently reduce scope on OOM.** Switch to `--precision 4bit`,
  record the change, and say so in the report.

---

## Part 5 — Fast reference

```bash
python -m pytest -q                      # 126 tests, CPU, ~2 min (use a transformers>=5.0 venv)
make help                                # every target, with its gate
make test                                # same as pytest
python eval/run.py --tiny --allow_cpu \
  --method sr_kv --task niah --context_len 384 --depths 50 \
  --n_samples 1 --output results/smoke.json    # harness smoke test, no GPU
python scripts/check_results.py gate --phase 4 --model qwen2.5-1.5b --budget 0.2
python scripts/kaggle_kernel.py run --phase 5 --user <username>
python scripts/kaggle_kernel.py generate --phase 4 --user <username> \
  --target phase4-scan --skip-gate       # diagnostic run, no gate attached
```

Docs: `README.md` (what and why), `CLAUDE.md` (the binding contract),
`KAGGLE.md` (manual/browser workflow), this file (state + automation).
