# Cache Lens backend

FastAPI + WebSocket backend for the live demo in `../webapp/`. Runs the
repo's real `src/caches/*` classes against a real Qwen2.5-0.5B-Instruct model
on CPU (no GPU required), answering a real question from the project's own
eval task (`eval/gist_mcq.py`) with all 5 method lanes interleaved token by
token, two-pass - the passage compresses with zero knowledge the question
exists, exactly like the real eval.

## Run it

```bash
# from the repo root, with the project venv active
make demo-backend-test   # CLI smoke test, tiny random model + synthetic corpus, no downloads, ~1s
make demo-backend        # the real server on :8000 (downloads qwen2.5-0.5b + Paul Graham essays once)
```

Then in a second terminal: `make demo-web` (or `cd webapp && npm run dev`),
and open `http://localhost:3000`.

`demo-backend` runs without `--reload` on purpose: `uvicorn --reload`'s
WatchFiles watcher silently missed edits to `server/*.py` more than once
during development on this machine (no error, no log line - it just kept
serving the old code, which looked exactly like a data bug from the
frontend). If you add `--reload` back for convenience, watch the server log
for a `WatchFiles detected changes in ...` line after every edit before
trusting what you see in the browser - if it's missing, restart manually.

## What's real, what's simplified

- **Real**: the cache classes (`SRKVCache`, `StreamingLLMCache`), the
  `make_cache` factory, `attach_cache`, the model, the eval task itself
  (`eval.gist_mcq.build_samples`/`score`), greedy decoding with the model's
  actual `repetition_penalty`. Every stat shown
  (`get_stats()`/`check_conservation()`) is read straight off the cache, never
  recomputed. The two-pass protocol (`decode_loop.py::run_gist`) mirrors
  `eval.gist_mcq.measure()`'s own discipline exactly, including its
  `cache.t_now + 1` position-continuation fix - validated against it directly
  (see Tests, below).
- **Simplified for the demo**: generation is greedy (`do_sample=False`) for
  determinism, the passage is much shorter than real Phase 8 runs
  (`server/config.py`'s `GIST_CONTEXT_LEN_*`, 200-800 vs. 2048-16384) to stay
  in the tens-of-seconds range on CPU, and only 3 of the 9
  `configs/defaults.yaml` scoring knobs are exposed to the UI (`budget`,
  passage length, and which variant) - the rest are fixed at the real
  experiment defaults, not hand-tuned for the demo.
- **One run at a time**: `src/attn_patch.py`'s `_ACTIVE_CACHE` is a
  module-level global by design (single-threaded, batch-1 — see its own
  docstring). `model_singleton.RUN_LOCK` enforces this; a second
  `start_gist_run` while one is in flight gets an immediate `error`, not a
  queue.

## Files

- `gist_source.py` — thin wrapper around `eval.gist_mcq.build_samples`,
  picking a fresh real question each request.
- `decode_loop.py` — `run_gist()`, the manual, interleaved, round-robin,
  two-pass multi-lane decode loop (replaces `model.generate()`, which has no
  way to yield control mid-generation without a custom streamer, and runs
  the passage and question as one combined pass, which `eval/gist_mcq.py`'s
  own docstring explains lets compression "see" the question before it
  compresses - defeating the task).
- `telemetry.py` — derives which original tokens are currently individually
  cached, folded into some centroid, or hard-evicted, from each cache's own
  public bookkeeping. See its module docstring for why this is a structural
  read (alive vs. not, plus whether the method clusters) rather than trying
  to track which specific centroid absorbed which tokens - the latter isn't
  reconstructable in general once `SRKVCache`'s default
  `recluster_centroids=True` is accounted for.
- `lanes.py` — the 5 lane configs, built through `src.caches.make_cache`,
  plus `USES_CLUSTERING` (read from the real flags, for `telemetry.py`).
- `ws_handler.py` / `app.py` / `protocol.py` — the WebSocket wiring.
- `tools/decode_loop_cli.py` — standalone validation script; also useful for
  a quick non-browser check (`--real` for the actual model, `--variant`,
  `--context_len`).
- `tests/test_decode_loop.py` — the same checks as the CLI, as pytest
  (`pytest server/tests/`; not wired into the root `make test`, which is the
  research suite proper). Includes an oracle test against
  `eval.gist_mcq.measure()` itself and a `score()`-agreement test.

## Known simplifications worth knowing about

- `SRKVCacheBase`'s own default `min_budget_tokens` is 32 (a real research
  choice, protecting short sequences from being over-compressed). For demo
  passages in the 200-800 token range at 10-50% budget, `budget * passage_len`
  can still land below 32, so that floor would silently override the budget
  slider - every compressed lane converging on the same size regardless of
  what was requested. `lanes.py` overrides it to 8 for the demo only
  (`configs/defaults.yaml`, which real experiments read, is untouched). If
  lanes still look identical at a low budget, check the passage is long
  enough that this floor isn't dominating.
- `telemetry.py` cannot report which *specific* centroid a given original
  token is folded into - only whether it's folded into *some* centroid. This
  is a structural limit of the real cache's own bookkeeping
  (`positions`/`slot_weights`/`is_centroid` give per-slot counts, not
  per-slot membership), sharpened by `recluster_centroids=True` meaning a
  centroid isn't even a stable entity across steps in the first place (see
  `telemetry.py`'s module docstring for the full reasoning and the earlier,
  wrong approach it replaced).
