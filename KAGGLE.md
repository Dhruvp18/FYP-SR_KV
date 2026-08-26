# Running SR-KV on Kaggle

You have no local GPU, so everything from Phase 1 onward runs on Kaggle. This
is the operational guide: how to get the code there, how to survive session
limits, and the exact command for each phase.

Local machine still does useful work: the whole test suite (70 tests) runs on
CPU with no downloads, so validate changes locally with `python -m pytest -q`
before burning GPU quota.

---

## 0. One-time setup

### 0.1 Get the code onto Kaggle

**Option A - GitHub (recommended, makes iteration painless)**

Locally, once:

```bash
cd sr-kv
git init
git add -A
git commit -m "SR-KV: phases 0-3"
git branch -M main
git remote add origin https://github.com/<you>/sr-kv.git
git push -u origin main
```

Then in every Kaggle notebook, first cell:

```python
!git clone -q https://github.com/<you>/sr-kv.git /kaggle/working/sr-kv
%cd /kaggle/working/sr-kv
```

If the repo is private, use a token: `https://<token>@github.com/<you>/sr-kv.git`
and keep it in Kaggle **Add-ons → Secrets**, never pasted in a cell.

**Option B - upload as a Kaggle Dataset**

Zip the `sr-kv` folder, create a Kaggle Dataset from it, attach it to the
notebook, then copy it into the writable working directory:

```python
!cp -r /kaggle/input/sr-kv-code/sr-kv /kaggle/working/
%cd /kaggle/working/sr-kv
```

Option B needs a re-upload on every code change, which is why A is better.

### 0.2 Notebook settings (right-hand sidebar)

- **Accelerator**: `GPU T4 x2` (SR-KV uses one GPU; two just means you got the
  better queue) or `GPU P100`.
- **Internet**: **On**. Needed to pip-install and to pull models from Hugging
  Face. Requires a phone-verified account.
- **Persistence**: "Files only" or "Variables and Files" so `/kaggle/working`
  survives between sessions of the *same* notebook.

### 0.3 Dependencies

```python
!pip install -q -U "transformers>=5.0" accelerate bitsandbytes
```

`torch` is already in the image - do not reinstall it. The cache classes are
written against the transformers 5.x `Cache`/`AttentionInterface` API and
`src/compat.py` raises a clear error on anything older, so this upgrade is not
optional.

For gated models (Llama-3.2 is gated) add your HF token as a Kaggle Secret
named `HF_TOKEN` and:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Qwen models are not gated, so Phases 1-5 need no token at all.

---

## 1. Surviving the session limits

Kaggle sessions are capped (roughly 12h) and can be killed without warning, and
you have a weekly GPU quota (check the exact figure in the sidebar - it is
around 30h/week). Three habits make that survivable:

**Use "Save Version → Save & Run All (Commit)" for anything long.** A committed
run executes detached; you can close the browser. Interactive sessions die when
your connection does.

**Everything is already resumable.** `eval/run.py` appends each finished task to
`<output>.jsonl` and fsyncs it. Re-running the identical command skips whatever
is already in that file. So the recovery procedure after any kill is: run the
same command again.

**Carry results forward between sessions.** `/kaggle/working` persists per
notebook, but to move results across notebooks or accounts, commit the notebook
and attach its output as a data source to the next one:

```python
# in the next notebook, before running anything
!mkdir -p /kaggle/working/sr-kv/results
!cp /kaggle/input/<previous-notebook-output>/results/*.jsonl /kaggle/working/sr-kv/results/ 2>/dev/null || true
```

Then the resume logic sees the earlier work and continues from there.

**Or just commit results to git.** `results/` is intentionally not gitignored,
so the simplest cross-session resume is:

```python
%cd /kaggle/working/sr-kv
!git config user.email "you@example.com" && git config user.name "you"
!git add results
!git commit -q -m "results: phase 5 partial"
!git push -q
```

Next session's `git clone` brings the finished tasks with it and the run picks
up from there.

**Split across accounts/workers with shards.** If you have collaborators, each
runs the same command with a different shard index:

```bash
python eval/run.py ... --shard 0 --num_shards 4   # worker 1
python eval/run.py ... --shard 1 --num_shards 4   # worker 2
```

Shards are exhaustive and non-overlapping (there is a test for it). Collect
every worker's `.jsonl` into one `results/` directory at the end; the plotting
code de-duplicates.

---

## 2. Phase-by-phase commands

Run these from `/kaggle/working/sr-kv`. Each is also a `make` target.

### Phase 1 - does the harness work on a real model?

```bash
python -m pytest -q                        # 70 CPU tests, ~2 min, no GPU needed

# the sanity check that matters: a trivial retrieval the model must ace
python eval/run.py --method full --model qwen2.5-1.5b --task niah \
  --context_len 512 --depths 50 --n_samples 10 --max_new_tokens 16 \
  --output results/phase1_sanity.json
```

**Pass condition: accuracy > 0.9.** If it is not, the harness is broken, not the
model - a 1.5B instruct model retrieves a magic number from 512 tokens without
difficulty. Check the prompt formatting first (`eval/memory.py::build_prompt`).

Exercise the 4-bit path at least once, so the fallback is known to work:

```bash
python eval/run.py --method full --model llama3.2-3b --precision 4bit \
  --task niah --context_len 512 --depths 50 --n_samples 3 \
  --output results/phase1_4bit.json
```

### Phase 2 - baselines behave the way the papers say

```bash
python eval/run.py --method full,streaming_llm,snapkv --model qwen2.5-1.5b \
  --task niah --context_len 4096 --depths 0,25,50,75,100 --budget 0.3 \
  --n_samples 5 --output results/phase2_baselines.json
```

**Expected shape of the result** (this is a qualitative ordering check, not a
numeric match to the papers): StreamingLLM should collapse at mid-sequence
depths - it keeps only sinks plus a recent window, so a needle at 50% is simply
gone - while SnapKV should hold up much better, because it scores before it
evicts. If StreamingLLM does *well* at depth 50, something is wrong with the
depth placement, not with StreamingLLM.

### Phase 3 - the unified class at full length

```bash
python eval/run.py --method sr_kv,centroid_merge,snapkv_unified \
  --model qwen2.5-1.5b --task niah --context_len 8192 \
  --depths 0,50,100 --budget 0.3 --n_samples 3 \
  --output results/phase3_8k.json
```

Every record carries `conservation_ok` and `budget_used_pct_max`; both are
asserted in the CPU tests and recorded here so an 8k run cannot silently
violate them.

### Phase 4 - choose the RoPE position convention (do this before any big sweep)

```bash
make phase4 MODEL=qwen2.5-1.5b BUDGET=0.3 SAMPLES=5
python scripts/freeze_rope_mode.py --model qwen2.5-1.5b        # dry run, prints the table
python scripts/freeze_rope_mode.py --model qwen2.5-1.5b --apply
git add configs/defaults.yaml && git commit -m "Phase 4: freeze RoPE mode"
```

`freeze_rope_mode.py` refuses to freeze anything if all three modes are at
chance, because that means the bug is upstream in clustering and freezing a
"winner" would just freeze noise. Every later phase reads the frozen value from
`configs/defaults.yaml`; a test enforces that no script hardcodes a mode.

### Phase 5 - the factorial matrix

```bash
make phase5 MODEL=qwen2.5-1.5b BUDGET=0.3 SAMPLES=3
make phase5-longbench MODEL=qwen2.5-1.5b BUDGET=0.3

make check-complete MODEL=qwen2.5-1.5b BUDGET=0.3     # lists any missing cell
make check-ablation MODEL=qwen2.5-1.5b                # the SR-KV sanity assertion
```

`check-ablation` exits 2 and prints a **FLAG** if SR-KV (full) comes out below
*both* ablated conditions. That is not a failure to hide - it is either a bug or
a real negative result, and both belong in the report.

This is the longest phase: 4 conditions x 4 context lengths x 5 depths x 3
samples = 240 generations, plus the uncompressed baseline. Commit it rather
than running interactively, and shard it if you have more than one account.

### Phase 6 - hyperparameters, then transfer to 3B

```bash
# sweep on 1.5B, sharded across workers
for a in 0.5 1.0 2.0; do for b in 0.0 0.3 0.6; do
  python eval/run.py --method sr_kv --model qwen2.5-1.5b --task niah \
    --context_len 4096,8192 --depths 0,25,50,75,100 --budget 0.3 \
    --alpha $a --beta $b --n_samples 3 \
    --shard 0 --num_shards 1 \
    --output results/phase6_sweep_a${a}_b${b}.json
done; done

# then take the winner to 3B WITHOUT re-sweeping - the question is whether it transfers
make phase6-3b BUDGET=0.3
```

### Phase 7 - figures

```bash
make report_artifacts
```

One command regenerates every figure from `results/`, so a last-minute rerun
before the deadline is a single line. Figures land in `figures/`.

---

## 3. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `RuntimeError: SR-KV requires transformers>=5.0` | The pip install cell did not run, or the kernel restarted. Re-run it. |
| `CUDA out of memory` on the uncompressed baseline | Expected at 16k on a T4. The baseline is the biggest run in the sweep. Drop to 4-bit (`--precision 4bit`) or reduce the max context, and **say so in the report** - do not quietly drop the cell. |
| Run died mid-sweep | Re-run the identical command. It resumes from the `.jsonl`. |
| `NotImplementedError: rope_type='dynamic'` | The model uses length-dependent RoPE scaling, which breaks centroid re-positioning (CLAUDE.md A5). Qwen2.5 and Llama-3.2 are both fine; a different model may not be. |
| Accuracy is 0 everywhere, all methods including `full` | The harness, not the method. Run the Phase 1 sanity check at 512 tokens. |
| All three RoPE modes score the same at chance | Bug in clustering or centroid construction, upstream of position assignment. Fix before Phase 5. |

## 4. Quota budgeting

Rough guide for planning against ~30 GPU-hours/week:

- Phase 1 sanity: minutes.
- Phase 2 at 4k: under an hour.
- Phase 4 (3 modes at 8k): 1-2 hours.
- Phase 5 full matrix: the expensive one. Prefill dominates, so the 16k cells
  cost more than everything else combined. Run 2k/4k/8k first, add 16k once the
  rest is complete.
- Phase 6 sweep: scales with the grid; shard it.

If you run short, the documented fallback is Qwen2.5-0.5B + 1.5B at 4k/8k only.
Take that decision explicitly and write it into the limitations section - a
stated constraint is worth more than a silently reduced scope.
