# SR-KV — project state and the remaining pipeline

This is the working document for anyone (human or agent) picking the project
up. It says exactly what is built, what is left, and how to execute the rest
on Kaggle without a browser.

Repo: <https://github.com/Dhruvp18/FYP-SR_KV>

---

## READ THIS FIRST — live handoff, 2026-09-17 evening

Chaitra is stepping away for a few hours and Dhruv is taking over the GPU
pipeline until she's back in the morning. This section is the ground truth as
of the handoff. If anything below conflicts with the rest of this file, trust
this section — the rest is being kept as reference material and general
process docs, not all of it was touched tonight.

**Everything in this section was produced by an AI agent (Claude Code)
driving `scripts/kaggle_kernel.py` end to end** — pushing kernels, polling,
pulling results, running gates, and fixing bugs it found along the way. If
Dhruv is also working with an agent, it can read this whole file for context
and continue the same pattern. If he's driving by hand, every command below
is copy-pasteable.

### Where things actually stand (real Tesla T4 runs, not aspirational)

**Phase 1 — PASS.** bf16 sanity (qwen2.5-1.5b, ctx=512): 1.000 accuracy, 10
samples. 4-bit fallback (qwen2.5-3b): 1.000 accuracy, 3 samples. Zero errors.
Committed: `results/phase1_sanity.json[l]`, `results/phase1_4bit.json[l]`.

**Phase 2 — PASS.** `full`=1.000, `snapkv`=1.000, `streaming_llm`=0.333 at
mid-depths (ctx=4096, budget=0.3) — the expected shape: StreamingLLM's 4-token
sink is far smaller than the needle sentence, so it only survives when the
needle happens to land in its always-kept recent window. 75 records, zero
errors. Committed: `results/phase2_baselines.json[l]`.

**Phase 3 — PASS.** Conservation and budget invariants hold across a real 8k
run for `sr_kv`/`centroid_merge`/`snapkv_unified`. All three score 1.000 at
every depth; `sr_kv` and `centroid_merge` both form exactly 309 centroids
consistently (real-hardware proof the RoPE centroid re-rotation code — see bug
#1 below — retains information correctly through merging, not just that the
accounting arithmetic balances). 27 records, zero errors. Committed:
`results/phase3_8k.json[l]`.

**Phase 4 — IN PROGRESS, not complete.** This is the RoPE position-mode
ablation: the actual novel piece of the project (`latest` / `earliest` /
`attn_weighted` — how to assign a RoPE position to a merged centroid). Two
rounds already ran and were **rejected on critical review**, both kept as
documented negative evidence rather than deleted:

- `budget=0.3`: **ceiling effect.** All three modes score exactly 1.000, every
  single record (75/75). The needle never gets merged into a centroid at this
  budget — it always survives via the "individually top-k-picked" path — so
  the ablation had zero power to show anything.
  Files: `results/phase4_rope_*_b030_ceiling.json[l]`.
- `budget=0.1`: **floor effect**, and this run also caught a real bug (see #3
  below) — the first "result" I evaluated was silently stale data from a
  previous run. The real number, once correctly pulled: **7/25, 6/25, 7/25**
  correct across the three modes. Fisher exact p=**1.000** on every pairwise
  comparison — the "0.04 spread" was one flipped sample, not a result.
  Files: `results/phase4_rope_*_b010_floor.json[l]`.
- A cheap scan (`make phase4-scan`, budgets 0.15/0.20/0.25, n=3, one mode)
  found **budget=0.20 is the discriminating zone**: d25=0.33, d50=0.67
  (genuinely sub-ceiling, not a wall of 0s/1s), vs. 0.15 floors (d25=d50=0.00)
  and 0.25 already saturates (all 1.0). Files: `results/phase4_scan_b0.*.json[l]`.
- **Running right now**: the full 3-mode ablation at `budget=0.20`,
  `SAMPLES4=20` (300 tasks total, vs. 75 before, for real statistical power).
  Pushed as `chaitrasamant/sr-kv-phase4`, kernel version 4. Estimated finish:
  roughly 2 hours from push (started ~23:40 on 2026-09-17).

  **⚠️ Dhruv cannot monitor or pull this specific run.** It's a private kernel
  under Chaitra's Kaggle account (`chaitrasamant`), and Dhruv's own Kaggle
  credentials have no access to it. See "What Dhruv should do," step 2.

**Phases 5, 6, 7 — not started.**

### Five real bugs found and fixed this session (read before touching anything)

**1. Per-layer position corruption — `src/caches/base.py`. The most important
fix; touches every phase from 3 onward.**
`transformers>=5.0`'s Qwen2/Llama attention calls
`past_key_values.update(key_states, value_states, layer_idx)` with **no
`cache_kwargs` at all** (confirmed by reading the installed modeling code
directly, not assumed). That means the cache's position-counter fallback
isn't a rare corner case — it's the *only* code path that ever runs, for
every layer, on every real model. The old code used one shared `_pos_counter`
int, bumped only when `layer_idx==0`; every other layer in the *same* forward
pass then read an already-advanced counter and recorded positions offset by a
full step's worth of tokens, drifting further every subsequent step. This
silently corrupted the RoPE delta used to un-rotate/re-rotate merged
centroids — the project's actual novel contribution — for every layer past 0.
It didn't crash and didn't produce non-finite output (existing tests didn't
catch it); it only surfaced as an explicit `positions.max() <= n_tokens_seen`
bound in one test. **Fixed**: per-layer counter dict instead of one shared
int. Confirmed on real Kaggle GPU hardware too (the exact same 3 tests failed
in the pulled `pytest_cache/lastfailed` from the very first Phase 1 kernel
run, before the fix landed). Nothing had been run on real GPU before this
session, so no historical results needed correcting — just the code.

**2. Kaggle kernel slug bug — `scripts/kaggle_kernel.py`.**
Kaggle derives a pushed kernel's *live* slug from `kernel-metadata.json`'s
`title` field, not from `id`. The old title
(`"SR-KV Phase 1 - harness sanity"`) slugified to
`sr-kv-phase-1-harness-sanity`, silently diverging from the `sr-kv-phase1`
slug every `status`/`pull` call assumed — so those calls 404'd against a
kernel that actually existed at a different URL. **Fixed**: title is now
literally the slug (`sr-kv-phase1`, etc).

**3. No `--force` on pull — `scripts/kaggle_kernel.py`.**
The Kaggle CLI's `kernels output` **skips any file whose local copy looks
newer**. Re-running a phase and pulling into the same `.kaggle_output/phaseN/`
directory silently re-serves the *previous* run's results. This is exactly
what happened on the first `budget=0.1` attempt — the pulled file's own
`metadata.argv` still read `--budget 0.3`. I only caught it because the
numbers looked too similar to the previous run and I checked the metadata.
**Fixed**: pulls now always pass `--force`. Still, as belt-and-suspenders:
**always `rm -rf .kaggle_output/phaseN` before re-pulling the same phase.**

**4. Gate/freeze had no statistical rigor — `scripts/check_results.py`,
`scripts/freeze_rope_mode.py`.**
Two compounding problems: (a) both scripts silently averaged across different
budgets if multiple sweeps' result files existed under the same base
filenames — which is exactly what would have happened if the `budget=0.1`
files hadn't been renamed before the `budget=0.2` run — and this is not
hypothetical, it's what actually happened: with the old `budget=0.3` ceiling
files still present, the gate reported `attn_weighted` as the winner; with
only the `budget=0.1` files, it reported `latest`. Same code, two different
"winners," decided purely by which files happened to be on disk. (b) Neither
had any concept of statistical power: the real spread (0.04, one flipped
record out of 25) was smaller than the tool's own "note: within 2 points"
threshold could ever catch, since 0.02 is finer than the 1/25 measurement
resolution. **Fixed**: both now (a) refuse to aggregate across budgets — name
one explicitly with `--budget`, and (b) run a Fisher exact test on
best-vs-worst mode and refuse to declare a winner (exit 2 / "REFUSING to
freeze") at p ≥ 0.05.
**When Phase 4's real run lands: read the p-value line, not just the printed
mean.** `configs/defaults.yaml` is still the unfrozen placeholder
(`rope_position_mode: attn_weighted`, `rope_position_mode_frozen: false`) —
correctly, given the evidence so far.

**5. Llama-3.2-3B → Qwen2.5-3B pivot — `src/models.py`, `Makefile`,
`scripts/gen_configs.py`, `scripts/kaggle_kernel.py`.**
`meta-llama/Llama-3.2-3B-Instruct` is gated on Hugging Face and blocked Phase
1 outright (confirmed via the actual kernel log: 401 `GatedRepoError`, nobody
had accepted the license or supplied a token). `microsoft/Phi-3.5-mini-instruct`
was considered as an ungated cross-architecture replacement and **ruled out
after checking its actual HF config**: `rope_scaling.type == "longrope"`,
exactly the length-dependent scheme `rope_positions.py` rejects (see
`CLAUDE.md` A5) — `RopeHelper.from_model()` would raise `NotImplementedError`
the instant `sr_kv`/`centroid_merge` tried to use it. Checked
`Qwen2.5-3B-Instruct` instead before committing: `rope_scaling: None`,
`max_position_embeddings: 32768` — safe. `MODEL3B` now defaults to
`qwen2.5-3b` everywhere (Makefile, `kaggle_kernel.py --model3b`,
`gen_configs.py`). **Trade-off for the writeup: Phase 6 now shows the tuned
config transferring across scale within the Qwen family, not across
architectures.** `llama3.2-3b` remains a supported `--model`/`--model3b`
value for whenever HF access lands (an HF-auth cell reading a Kaggle
`HF_TOKEN` secret was also added to every generated notebook, so it'll work
the moment the license is accepted).

### The strategic question this session surfaced — important for the writeup

Phase 3 showed `snapkv_unified` (pure hard eviction, **no** merging)
*also* scoring 1.000, tied with `sr_kv`/`centroid_merge`, at budget=0.3.
That's not a bug — it's a structural property worth stating plainly:
**single-needle NIAH exact-match retrieval cannot show centroid merging
beating hard eviction, at any budget.** A centroid is a weighted average of
~20 token vectors; it cannot reproduce an exact 6-digit number by
construction. On this task, merging can at best *tie* hard eviction (when the
needle survives via attention scoring regardless of method) and can never
beat it. No amount of budget-tuning fixes this — it's about what the metric
can reward, not about calibration.

Discussed with Chaitra; the plan going in:

- Finish calibrating the NIAH RoPE ablation (in progress — see Phase 4 above).
  NIAH stays useful for "does compression break exact retrieval at all," just
  not for proving merging beats dropping.
- Treat `phase5-longbench` (already built — `eval/longbench.py`), especially
  `gov_report` (ROUGE-L, summarization), as the **primary evidence for the
  actual thesis**. Gist-preservation is exactly what a centroid encodes and
  exactly what hard eviction discards, and graded metrics (F1/ROUGE) carry
  far more statistical power per sample than NIAH's binary exact-match.
- Before running Phase 5's NIAH half at the Makefile's default `BUDGET=0.3`,
  expect it may *also* saturate for the same structural reason. Either sweep
  to a discriminating budget there too, or accept a probable ceiling tie on
  NIAH and lean on LongBench for the headline number — either is fine, but
  don't be surprised by a tie and don't quietly bury it (gate 5 is built to
  surface a genuine SR-KV loss as exit code 2, not a hidden pass — do the
  same in spirit for a NIAH ceiling tie: report it, don't hide it).

### What Dhruv should do, concretely

**1. Set up his own Kaggle credentials — his account is separate from
Chaitra's.**

```bash
# Kaggle -> Settings -> API -> Create New Token (the newer KGAT_... bearer-token style)
mkdir -p ~/.kaggle
printf '%s' "<YOUR_TOKEN_HERE>" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

pip install kaggle
kaggle --version     # if "kaggle: command not found", the install went to a
                      # user Scripts dir not on PATH - find it and prepend it,
                      # e.g. on Windows:
                      # export PATH="$HOME/AppData/Roaming/Python/Python312/Scripts:$PATH"
                      # or just invoke as `python -m kaggle` everywhere below

kaggle kernels list -m   # confirms auth; lists his own kernels
```

Also confirm his account is **phone-verified**
(Settings → Phone Verification) — required for GPU + internet-enabled
kernels, without which pushed kernels fail outright.

His Kaggle username goes in every `--user` flag below.

**2. Check whether Chaitra's Phase 4 run has landed.**
He can't query her private kernel directly, but the only thing that matters
is whether the *results* have been pushed to GitHub:

```bash
git pull origin main   # always do this first, every session
```

- If `results/phase4_rope_latest.json` (and `earliest`, `attn_weighted`)
  exist **and their content's `budget` field is `0.2`** (check with
  `python -c "import json; print(json.load(open('results/phase4_rope_latest.json'))['records'][0]['budget'])"`,
  should print `0.2`) → the run finished and (if Chaitra's agent session was
  still active) probably already got committed with a gate result. Read the
  commit log (`git log --oneline -10`) to see what happened. If for some
  reason it's not yet gated:
  ```bash
  python scripts/check_results.py gate --phase 4 --model qwen2.5-1.5b --budget 0.2
  ```
  Read the **p-value line**, not just the printed means. If it exits 0
  (real, significant winner):
  ```bash
  python scripts/freeze_rope_mode.py --model qwen2.5-1.5b --budget 0.2 --apply
  git add configs/defaults.yaml && git commit -m "Freeze RoPE mode: <winner> (p=<value>, budget=0.2)"
  git push origin main
  ```
  Then move to Phase 5. If it exits 2 (still no significant difference even at
  n=20/mode) — that's a real, reportable finding on its own. Pick
  `attn_weighted` as a principled (but undefended) default, document it in the
  writeup as "not empirically distinguished," and move to Phase 5 without
  freezing.
- If those files don't exist yet, or exist with `budget: 0.1` or `0.3` — the
  new run hasn't landed. **Don't block on it.** Do step 3 instead.

**3. Parallel-safe work that doesn't depend on Phase 4's outcome.**
`streaming_llm` and `snapkv_unified` don't use `rope_position_mode` at all (no
clustering), so their numbers can't be invalidated by whatever Phase 4
decides. Get real progress on Phase 5's LongBench half now:

```bash
python -m pytest -q   # always, before pushing any kernel

python scripts/kaggle_kernel.py generate --phase 5 --user <dhruv-username> \
  --target phase5-longbench --skip-gate
python scripts/kaggle_kernel.py push --phase 5
python scripts/kaggle_kernel.py status --phase 5 --user <dhruv-username>
# ... wait, poll every few minutes ...
rm -rf .kaggle_output/phase5   # belt-and-suspenders, see bug #3
python scripts/kaggle_kernel.py pull --phase 5 --user <dhruv-username>
```

`--skip-gate` is there because the real `gate5` needs all four conditions
including `sr_kv`/`centroid_merge`, and those two are worth re-running once
Phase 4 freezes a mode (though the effect measured so far, even before it was
ruled non-significant, was only ~0.04-0.07 — likely low-cost to redo just
those rows later if needed). Running `sr_kv`/`centroid_merge` now too, with
the current placeholder `attn_weighted`, is a reasonable bet if there's spare
time — just flag in the writeup which rows might get superseded.

**4. If there's time left after that**, the big one is the full NIAH factorial
(`make phase5` via `kaggle_kernel.py generate --phase 5 --user ...` without
`--target`/`--skip-gate`). Given the ceiling-effect risk discussed above,
consider a quick per-budget probe (same idea as `phase4-scan`, adapted) before
committing the full grid to `BUDGET=0.3` — or just run it and treat a ceiling
tie as a real, reportable finding rather than a failure to hide.

**5. Non-negotiables, every phase:**
- `python -m pytest -q` before every kernel push (2 CPU minutes; catches a
  broken commit before it burns GPU quota).
- `rm -rf .kaggle_output/phaseN` before re-pulling the same phase.
- Never edit `src/` to make a gate pass. Never loosen a gate threshold. A bad
  result is the deliverable, not a bug to hide — this project's gates
  (`scripts/check_results.py`) are explicitly designed to surface negative
  findings (exit code 2) rather than let them slide through as a quiet pass.
- Commit and push after every phase, with a commit message that states the
  actual numbers and what they mean (see the commit log for the pattern used
  tonight) — so Chaitra can read `git log` in the morning and know exactly
  what happened without re-deriving it from raw JSON.
- When generating a kernel for a phase that's specifically *his* to run
  (rather than resuming one of Chaitra's), pass `--user <dhruv-username>` —
  `kaggle_kernel.py` bakes the username into the kernel id and into which
  earlier-phase kernels it mounts for resume (`kernel_sources`). Don't mix
  the two accounts' phase numbers; if he starts fresh, his Phase 5 should
  chain from *his own* Phase 1-4 (or just skip the `kernel_sources`/resume
  chaining entirely for a from-scratch Phase 5 run, since Phase 5 doesn't
  actually need Phase 1-4's *outputs* mounted, only their code fixes, which
  come from `git pull`, not from kernel mounting).

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
| 1 | `make phase1 phase1-4bit` | `make gate1` | **PASS** (real GPU, 1.000 acc both bf16 and 4-bit) |
| 2 | `make phase2` | `make gate2` | **PASS** (real GPU, expected StreamingLLM/SnapKV split) |
| 3 | `make phase3` | `make gate3` | **PASS** (real GPU, conservation + budget hold, centroids verified) |
| 4 | `make phase4-scan` then `make phase4` → `make freeze-rope` | `make gate4` | **IN PROGRESS** — see live handoff above |
| 5 | `make phase5 phase5-longbench` | `make gate5` | not started — LongBench half is the priority, see strategic note above |
| 6 | `make phase6-sweep phase6-3b` | `make gate6` | not started (MODEL3B is now qwen2.5-3b) |
| 7 | `make phase7` | `make gate7` | not started |

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
