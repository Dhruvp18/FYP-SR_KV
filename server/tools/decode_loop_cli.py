#!/usr/bin/env python
"""Stage-1 validation for the live-demo decode loop - no FastAPI, no Next.js.

Runs the manual round-robin loop (`server.decode_loop.run`) against the repo's
existing CPU-only tiny random-weights model (`src.models.build_tiny_model`),
the same no-network, no-download philosophy as `eval/run.py --tiny --allow_cpu`.

Checks, every run:
  * `cache.check_conservation()` holds for every lane at every step.
  * `cache.get_stats()` never drops its 4 contract keys.
  * the "full" lane's greedy token ids exactly match
    `eval.memory.generate_and_measure(...)`'s output on the identical
    prompt/cache - proof the hand-rolled loop is faithful to `model.generate()`
    before trusting it on the other 4 (compressed) lanes.

Usage:
    python server/tools/decode_loop_cli.py --tiny                  (default, fast, CI-safe)
    python server/tools/decode_loop_cli.py --real --budget 0.3     (real Qwen2.5-0.5B, CPU, slow)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src  # noqa: E402  (applies env guards before transformers, per src/_env.py)
from src.caches import make_cache  # noqa: E402
from src.models import build_tiny_model, build_tiny_tokenizer, load_model  # noqa: E402

from server.decode_loop import run  # noqa: E402
from server.lanes import DEFAULT_OVERRIDES  # noqa: E402

STATS_KEYS = {"n_tokens_cached", "n_tokens_evicted", "n_centroids", "budget_used_pct"}


def _check_full_lane_oracle(model, tokenizer, prompt, budget, max_new_tokens, manual_ids):
    """The `full` lane must match `eval.memory.generate_and_measure` exactly."""
    from eval.memory import generate_and_measure

    oracle_cache = make_cache("full", model=model, budget=budget, **DEFAULT_OVERRIDES)
    result = generate_and_measure(model, tokenizer, prompt, oracle_cache, max_new_tokens=max_new_tokens)
    oracle_text = result["generated_text"]
    manual_text = tokenizer.decode(manual_ids, skip_special_tokens=True)
    ok = oracle_text == manual_text
    print(f"[oracle] full-lane match: {ok}")
    if not ok:
        print(f"  generate():    {oracle_text!r}")
        print(f"  manual loop:   {manual_text!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="use the real qwen2.5-0.5b model instead of the tiny one")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog and then")
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--quiet", action="store_true", help="only print the summary, not every step event")
    args = ap.parse_args()

    if args.real:
        print("Loading qwen2.5-0.5b on CPU (allow_cpu=True, explicit local-demo choice)...")
        model, tokenizer = load_model("qwen2.5-0.5b", allow_cpu=True)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        model = build_tiny_model()
        tokenizer = build_tiny_tokenizer()
        prompt = args.prompt

    conservation_failures: list[str] = []
    stats_failures: list[str] = []
    step_count = 0

    def emit(event: dict) -> None:
        nonlocal step_count
        if event["type"] == "step":
            step_count += 1
            if not STATS_KEYS.issubset(event["stats"]):
                stats_failures.append(f"{event['lane']} step {event['step']}: bad stats keys {event['stats'].keys()}")
            if not event["conservation_ok"]:
                conservation_failures.append(f"{event['lane']} step {event['step']} phase={event['phase']}")
        if not args.quiet:
            print(json.dumps(event, default=str))

    started = time.perf_counter()
    result = run(
        model,
        tokenizer,
        prompt,
        budget=args.budget,
        max_new_tokens=args.max_new_tokens,
        overrides=None,
        emit=emit,
    )
    elapsed = time.perf_counter() - started

    print(f"\n--- summary ({elapsed:.1f}s, {step_count} step events) ---")
    for row in result["summary"]:
        print(f"  {row['lane']:16s} cached={row['final_stats']['n_tokens_cached']:3d} "
              f"evicted={row['final_stats']['n_tokens_evicted']:3d} "
              f"centroids={row['final_stats']['n_centroids']:2d} "
              f"conservation_ok={row['conservation_ok']}  text={row['generated_text']!r}")

    ok = True
    if conservation_failures:
        ok = False
        print(f"\nFAIL: conservation identity broke {len(conservation_failures)} time(s):")
        for f in conservation_failures[:10]:
            print(f"  {f}")
    if stats_failures:
        ok = False
        print(f"\nFAIL: get_stats() contract violated {len(stats_failures)} time(s):")
        for f in stats_failures[:10]:
            print(f"  {f}")

    full_ids = result["lanes"]["full"].generated_ids
    oracle_ok = _check_full_lane_oracle(model, tokenizer, prompt, args.budget, args.max_new_tokens, full_ids)
    ok = ok and oracle_ok

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
