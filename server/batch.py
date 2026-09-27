"""A small, live demo of the real Phase 8 gist_mcq benchmark task.

The real H3a (attribution) / H3b (aggregation) experiment
(PREREGISTRATION.md's addendum, `Makefile:391-419`'s `phase8-gist` target,
confirmed against the actual record counts in
`results/phase8_gist_{attr,agg}_qwen2.5-1.5b.jsonl`) ran context_len=8192,
budgets 0.1 and 0.15, n=50 questions per (method, budget) cell, on GPU, at a
measured ~21-27s/sample - 700 sequential samples, roughly 4-5 hours on GPU
alone. Not interactive, and this repo's live demo is CPU-only.

This module runs the *same* task design, the *same* both variants, and the
*same* real scoring (`eval.gist_mcq.score()`, via `decode_loop.run_gist` -
unchanged, unreimplemented), but at a single fixed budget chosen for a clear
live demo (`BATCH_BUDGET`, not one of the pre-registered 0.1/0.15 budgets)
and far fewer samples, so a full run finishes in minutes instead of hours.
Every consumer of this table must say plainly that it's a small demo, not
the pre-registered statistical test - the numbers are for "does the
direction look right," not statistical significance.

A "batch" is nothing but the existing single-question flow
(`decode_loop.run_gist`) called in a loop: nothing about answering one
question changes here, only what happens around it - events get tagged with
which cell they belong to, and `run_complete` (normally the terminal event)
is intercepted and folded into a running table instead of ending the run.
"""

from __future__ import annotations

from typing import Callable

from .decode_loop import run_gist
from .gist_source import build_demo_sample
from .lanes import LANE_METHODS

BATCH_VARIANTS = ("attribution", "aggregation")
#: a single fixed budget, picked for a clear live demo - NOT one of
#: PREREGISTRATION.md's pre-registered 0.1/0.15 budgets. Keep any copy that
#: describes this table honest about that.
BATCH_BUDGET = 0.2
#: the real test used 50 per (method, budget) cell; scaled down for a live,
#: CPU-interactive demo. One budget now instead of two, so this is 10
#: questions per *variant* (20 total), not per cell.
BATCH_N_PER_CELL_DEFAULT = 10

EmitFn = Callable[[dict], None]


def _fresh_table() -> dict:
    return {variant: {method: {"correct": 0, "total": 0} for method in LANE_METHODS} for variant in BATCH_VARIANTS}


def run_gist_batch(
    model,
    tokenizer,
    *,
    context_len: int,
    emit: EmitFn,
    should_cancel: Callable[[], bool] = lambda: False,
    n_per_cell: int = BATCH_N_PER_CELL_DEFAULT,
) -> dict:
    table = _fresh_table()
    total = len(BATCH_VARIANTS) * n_per_cell
    done = 0
    cancelled = False

    for variant in BATCH_VARIANTS:
        for sample_idx in range(n_per_cell):
            if should_cancel():
                cancelled = True
                break

            sample = build_demo_sample(tokenizer, variant=variant, context_len=context_len, corpus="pg")
            cell = {"variant": variant, "budget": BATCH_BUDGET, "sample_idx": sample_idx, "n_per_cell": n_per_cell}

            def wrapped_emit(event: dict, cell: dict = cell) -> None:
                if event["type"] == "run_complete":
                    for row in event["summary"]:
                        tally = table[cell["variant"]][row["lane"]]
                        tally["total"] += 1
                        if row["correct"]:
                            tally["correct"] += 1
                    emit({"type": "batch_sample_complete", **cell, "summary": event["summary"]})
                else:
                    emit({**event, "batch_cell": cell})

            run_gist(
                model, tokenizer, sample, budget=BATCH_BUDGET, overrides=None, emit=wrapped_emit, should_cancel=should_cancel
            )
            done += 1
            emit({"type": "batch_progress", "done": done, "total": total, "table": table})
        if cancelled:
            break

    emit({"type": "batch_complete", "table": table, "cancelled": cancelled})
    return {"table": table, "cancelled": cancelled}
