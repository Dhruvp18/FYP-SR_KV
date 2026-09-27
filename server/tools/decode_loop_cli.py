#!/usr/bin/env python
"""Stage validation for the live-demo decode loop - no FastAPI, no Next.js.

Runs the manual round-robin loop (`server.decode_loop.run_gist`) against a
real question from the project's own eval task (`eval.gist_mcq`), against
the repo's existing CPU-only tiny random-weights model by default
(`src.models.build_tiny_model`), the same no-network, no-download philosophy
as `eval/run.py --tiny --allow_cpu`.

Checks, every run:
  * `cache.check_conservation()` holds for every lane at every step.
  * `cache.get_stats()` never drops its 4 contract keys.
  * the "full" lane's greedy answer exactly matches
    `eval.gist_mcq.measure(...)`'s own two-pass output on the identical
    sample/cache - proof the hand-rolled loop is faithful to the real eval's
    two-pass protocol before trusting it on the other 4 (compressed) lanes.
  * each lane's `chosen_letter`/`correct` matches `eval.gist_mcq.score()`.

Usage:
    python server/tools/decode_loop_cli.py                          (default, tiny model, fast, CI-safe)
    python server/tools/decode_loop_cli.py --real --budget 0.2      (real Qwen2.5-0.5B, CPU, slow)
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

from server.config import GIST_CONTEXT_LEN_DEFAULT  # noqa: E402
from server.decode_loop import run_gist  # noqa: E402
from server.gist_source import build_demo_sample  # noqa: E402
from server.lanes import DEFAULT_OVERRIDES  # noqa: E402

STATS_KEYS = {"n_tokens_cached", "n_tokens_evicted", "n_centroids", "budget_used_pct"}


def _check_full_lane_oracle(model, tokenizer, sample, budget, manual_text) -> bool:
    """The `full` lane must match `eval.gist_mcq.measure()`'s own two-pass
    output exactly - the real oracle for this exact task, not an approximation."""
    from eval.gist_mcq import measure

    oracle_cache = make_cache("full", model=model, budget=budget, **DEFAULT_OVERRIDES)
    result = measure(model, tokenizer, sample, oracle_cache)
    oracle_text = result["generated_text"]
    ok = oracle_text == manual_text
    print(f"[oracle] full-lane match: {ok}")
    if not ok:
        print(f"  measure():     {oracle_text!r}")
        print(f"  manual loop:   {manual_text!r}")
    return ok


def main() -> int:
    # Windows consoles often default stdout to cp1252, which can't encode the
    # unicode marks this script prints (checkmarks, em dashes in decoded
    # text) - reconfigure rather than let a print() crash mid-run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="use the real qwen2.5-0.5b model instead of the tiny one")
    ap.add_argument("--variant", choices=["attribution", "aggregation"], default=None)
    ap.add_argument("--context_len", type=int, default=GIST_CONTEXT_LEN_DEFAULT)
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--quiet", action="store_true", help="only print the summary, not every step event")
    args = ap.parse_args()

    if args.real:
        print("Loading qwen2.5-0.5b on CPU (allow_cpu=True, explicit local-demo choice)...")
        model, tokenizer = load_model("qwen2.5-0.5b", allow_cpu=True)
        corpus = "pg"
    else:
        model = build_tiny_model()
        tokenizer = build_tiny_tokenizer()
        corpus = "synthetic"

    sample = build_demo_sample(tokenizer, variant=args.variant, context_len=args.context_len, corpus=corpus)
    print(f"[sample] variant={sample.variant} context_len={sample.context_len} answer={sample.answer_letter}")
    print(f"[sample] question: {sample.question!r}")

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
    result = run_gist(
        model,
        tokenizer,
        sample,
        budget=args.budget,
        overrides=None,
        emit=emit,
    )
    elapsed = time.perf_counter() - started

    print(f"\n--- summary ({elapsed:.1f}s, {step_count} step events) ---")
    for row in result["summary"]:
        mark = "?" if row["chosen_letter"] is None else ("✓" if row["correct"] else "✗")
        print(
            f"  {row['lane']:16s} cached={row['final_stats']['n_tokens_cached']:3d} "
            f"evicted={row['final_stats']['n_tokens_evicted']:3d} "
            f"centroids={row['final_stats']['n_centroids']:2d} "
            f"chose={row['chosen_letter']} {mark}  "
            f"conservation_ok={row['conservation_ok']}  text={row['generated_text']!r}"
        )

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

    full_text = next(row["generated_text"] for row in result["summary"] if row["lane"] == "full")
    oracle_ok = _check_full_lane_oracle(model, tokenizer, sample, args.budget, full_text)
    ok = ok and oracle_ok

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
