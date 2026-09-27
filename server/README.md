# Cache Lens backend

FastAPI + WebSocket backend for the live demo in `../webapp/`. Runs the
repo's real `src/caches/*` classes against a real Qwen2.5-0.5B-Instruct model
on CPU (no GPU required), interleaving all 5 method lanes token by token.

## Run it

```bash
# from the repo root, with the project venv active
make demo-backend-test   # CLI smoke test, tiny random model, no downloads, ~1s
make demo-backend        # the real server on :8000 (downloads qwen2.5-0.5b once, ~1GB)
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
  `make_cache` factory, `attach_cache`, the model, greedy decoding with the
  model's actual `repetition_penalty`. Every stat shown
  (`get_stats()`/`check_conservation()`) is read straight off the cache, never
  recomputed.
- **Simplified for the demo**: generation is greedy (`do_sample=False`) for
  determinism, and only 5 of the 9 `configs/defaults.yaml` scoring knobs are
  exposed to the UI (`budget`) — the rest are fixed at the real experiment
  defaults, not hand-tuned for the demo.
- **One run at a time**: `src/attn_patch.py`'s `_ACTIVE_CACHE` is a
  module-level global by design (single-threaded, batch-1 — see its own
  docstring). `model_singleton.RUN_LOCK` enforces this; a second `start_run`
  while one is in flight gets an immediate `error`, not a queue.

## Files

- `decode_loop.py` — the manual, interleaved, round-robin multi-lane decode
  loop (replaces `model.generate()`, which has no way to yield control
  mid-generation without a custom streamer).
- `telemetry.py` — reconstructs which original tokens got folded into which
  centroid, by diffing the cache's own public bookkeeping across steps. Never
  guesses at scoring; only observes.
- `lanes.py` — the 5 lane configs, built through `src.caches.make_cache`.
- `ws_handler.py` / `app.py` / `protocol.py` — the WebSocket wiring.
- `tools/decode_loop_cli.py` — standalone validation script; also useful for
  a quick non-browser check of a prompt (`--real` for the actual model).
- `tests/test_decode_loop.py` — the same checks as the CLI, as pytest
  (`pytest server/tests/`; not wired into the root `make test`, which is the
  research suite proper).

## Known simplifications worth knowing about

- `telemetry.py`'s centroid membership tracking assumes a plain token's
  position never repeats and a centroid's position only drifts by roughly one
  token-width per step — true of every method in this repo today. If a future
  cache variant repositions a *plain* token, the diffing heuristic would need
  the same is_centroid-aware fix that hard-eviction lanes needed (see the
  comment in `telemetry.py`'s `diff()`).
- `SRKVCacheBase`'s own default `min_budget_tokens` is 32 (a real research
  choice, protecting short sequences from being over-compressed). For demo
  prompts in the 40-150 token range at 10-50% budget, `budget * prompt_len`
  is routinely below 32, so that floor silently overrides the budget slider -
  every compressed lane converges on exactly 32 regardless of what was
  requested, which reads as "all four methods behave identically" when they
  were never actually tested at the requested compression. `lanes.py`
  overrides it to 8 for the demo only (`configs/defaults.yaml`, which real
  experiments read, is untouched). If lanes still look identical at a low
  budget, check the prompt is long enough that this floor isn't dominating.
