"""Validate a sweep before anything is plotted or written up.

    python scripts/check_results.py gate --phase 5 --model qwen2.5-1.5b
    python scripts/check_results.py completeness --model qwen2.5-1.5b --budget 0.3
    python scripts/check_results.py ablation --model qwen2.5-1.5b

`gate --phase N` is the machine-checkable pass condition for phase N, and is
what an automation agent calls after every Kaggle run to decide whether to
continue:

    exit 0  proceed to the next phase
    exit 1  the phase failed or is incomplete - do not proceed
    exit 2  a result that needs a human look before proceeding

`completeness` lists every missing (condition, context, depth) cell rather than
letting `make_plots.py` quietly draw a figure from half a sweep.

`ablation` runs the Phase 5 sanity assertion: SR-KV (full) should not be worse
than *both* SnapKV-style hard eviction and centroid-merge-without-recency at
the same budget, averaged over NIAH depths. If it is, that is either a bug or a
real negative result - both worth knowing, neither worth hiding - so it is
surfaced as a FLAG with the numbers attached.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_plots import load_records  # noqa: E402
from src.rope_positions import POSITION_MODES  # noqa: E402

FACTORIAL = ["streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"]
# 16384 is deliberately absent. Measured on real T4s, qwen2.5-1.5b in bf16
# peaks at 5.11 GiB at ctx=4096 (n=210) and 10.96 GiB at ctx=8192 (n=603),
# so 16384 lands around 16.8 GiB against a 14.56 GiB card - and an external
# run did OOM there, on the first `full` task. Compression does not save it:
# those 8192 figures are already with a compressed cache, because the peak is
# prefill attention over the full sequence, which every method pays before any
# eviction runs (see CLAUDE.md A2).
#
# This is a recorded scope reduction, not a loosened gate: the pass conditions
# are untouched, only the grid is narrowed to what the hardware can measure,
# and the NIAH claim is therefore "the factorial at 2k-8k". Worth stating
# plainly in the writeup, because 16k is where KV compression matters most -
# this is a hardware limit, not evidence about the method.
DEFAULT_CONTEXTS = [2048, 4096, 8192]
DEFAULT_DEPTHS = [0, 25, 50, 75, 100]
#: phase6-3b's own method/context list (includes "full"; Makefile's own spec)
PHASE6_METHODS = ["full", "streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"]
# 8192 is deliberately absent. qwen2.5-3b does not fit at 8192 in bf16 on a
# 14.56 GiB T4 (the scan OOM'd asking for 4.05 GiB with 3.91 free), and 4-bit,
# which does fit, hangs the GPU intermittently - three runs stalled after 31,
# 37 and 4 tasks with no error, no log and a process that SIGKILL could not
# reap. bf16 at 2048/4096 is the configuration this hardware can actually
# measure, so the 3B transfer claim is scoped to 2k-4k and says so rather than
# carrying an 8192 column that no run can fill.
PHASE6_CONTEXTS = [2048, 4096]
DEFAULT_LB_TASKS = ["narrativeqa", "qasper", "gov_report", "triviaqa"]

#: a 1.5B instruct model must clear this on a 512-token retrieval
SANITY_ACCURACY = 0.9
#: below this, "the best RoPE mode" is indistinguishable from noise
CHANCE_CEILING = 0.05


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def binary_counts(values) -> tuple[int, int] | None:
    """``(n_correct, n)`` when every value is 0.0/1.0, else None.

    NIAH is exact-match, so its accuracies are binary and a proper
    significance test applies. LongBench F1/ROUGE are continuous and need a
    different test, so callers fall back to reporting the spread for those.
    """
    values = list(values)
    if values and all(v in (0.0, 1.0) for v in values):
        return int(sum(values)), len(values)
    return None


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]. No scipy needed.

    Used to answer the question the mean on its own cannot: is the gap
    between the best and worst mode bigger than sampling noise at this
    sample size? With n=25 per arm, one flipped record moves the mean by
    0.04, so an "0.04 spread" can be a single sample and nothing more.
    """
    from math import comb

    n = a + b + c + d
    row1, row2, col1 = a + b, c + d, a + c
    if min(row1, row2, col1, n - col1) < 0 or n == 0:
        return 1.0

    def prob(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)

    observed = prob(a)
    total = sum(
        prob(x)
        for x in range(max(0, col1 - row2), min(row1, col1) + 1)
        if prob(x) <= observed + 1e-12
    )
    return min(total, 1.0)


#: a difference this likely under the null is not a result worth freezing
SIGNIFICANCE_ALPHA = 0.05


def _rows(records, **match):
    return [r for r in records if all(r.get(k) == v for k, v in match.items() if v is not None)]


# ---------------------------------------------------------------------------
# completeness and ablation
# ---------------------------------------------------------------------------
def check_completeness(records, *, model, budgets, methods, contexts, depths, n_samples,
                       lb_tasks) -> list[str]:
    """Human-readable description of every missing cell."""
    have: dict[tuple, int] = defaultdict(int)
    for r in records:
        if model and r.get("model") != model:
            continue
        if "error" in r:
            continue
        if r.get("context_len") is not None:
            have[(r["method"], r["budget"], r["context_len"], r["depth"])] += 1
        elif r.get("lb_task"):
            have[(r["method"], r["budget"], r["lb_task"], None)] += 1

    missing: list[str] = []
    for method in methods:
        method_budgets = [1.0] if method == "full" else budgets
        for budget in method_budgets:
            for context in contexts:
                for depth in depths:
                    got = have.get((method, budget, context, depth), 0)
                    if got < n_samples:
                        missing.append(
                            f"NIAH  {method:<16} budget={budget:<5} ctx={context:<6} "
                            f"depth={depth:<4} {got}/{n_samples} samples"
                        )
            for task in lb_tasks:
                if have.get((method, budget, task, None), 0) == 0:
                    missing.append(f"LB    {method:<16} budget={budget:<5} task={task} 0 samples")
    return missing


def check_ablation(records, *, model) -> list[dict]:
    """SR-KV against its two ingredient-ablated siblings, per budget."""
    rows = [r for r in records
            if (not model or r.get("model") == model)
            and r.get("context_len") is not None and "error" not in r]

    by: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        by[(r["budget"], r["method"])].append(r["accuracy"])

    findings = []
    for budget in sorted({b for b, _ in by}):
        sr = _mean(by.get((budget, "sr_kv"), []))
        hard = _mean(by.get((budget, "snapkv_unified"), []))
        merge = _mean(by.get((budget, "centroid_merge"), []))
        if any(v != v for v in (sr, hard, merge)):  # NaN => a condition is missing
            findings.append({"budget": budget, "status": "INCOMPLETE", "sr_kv": sr,
                             "snapkv_unified": hard, "centroid_merge": merge})
            continue
        findings.append({
            "budget": budget,
            "status": "FLAG" if (sr < hard and sr < merge) else "ok",
            "sr_kv": sr,
            "snapkv_unified": hard,
            "centroid_merge": merge,
            "margin_vs_best_ablation": sr - max(hard, merge),
        })
    return findings


# ---------------------------------------------------------------------------
# phase gates
# ---------------------------------------------------------------------------
def gate_phase1(records, *, model) -> tuple[int, list[str]]:
    """A trivial retrieval the model must ace, or the harness is wrong."""
    rows = [r for r in _rows(records, model=model, method="full")
            if r.get("context_len") == 512 and "error" not in r]
    if len(rows) < 5:
        return 1, [f"only {len(rows)} sanity records at ctx=512, expected >= 5. Run `make phase1`."]

    accuracy = _mean(r["accuracy"] for r in rows)
    lines = [f"uncompressed NIAH accuracy at ctx=512: {accuracy:.3f} over {len(rows)} samples"]
    if accuracy <= SANITY_ACCURACY:
        lines.append(
            f"FAIL: below the {SANITY_ACCURACY} gate. At 512 tokens this is the harness, not the "
            "model. Check prompt formatting (eval/memory.py::build_prompt) first, then that the "
            "needle survives tokenizer round-tripping (eval/niah.py::build_samples)."
        )
        return 1, lines
    return 0, lines + ["PASS"]


def gate_phase2(records, *, model) -> tuple[int, list[str]]:
    """StreamingLLM must fail mid-sequence where SnapKV does not."""
    mid = (25, 50, 75)
    scores = {}
    for method in ("full", "streaming_llm", "snapkv"):
        rows = [r for r in _rows(records, model=model, method=method)
                if r.get("context_len") == 4096 and r.get("depth") in mid and "error" not in r]
        if not rows:
            return 1, [f"no 4k records for {method}; run `make phase2`."]
        scores[method] = _mean(r["accuracy"] for r in rows)

    lines = ["mid-depth (25/50/75%) accuracy at 4k: "
             + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())]
    if scores["streaming_llm"] >= scores["snapkv"]:
        lines.append(
            "FAIL: StreamingLLM did not underperform SnapKV mid-sequence. StreamingLLM keeps only "
            "sinks plus a recent window, so a needle at 50% depth should be gone. If it is not, "
            "the needle is probably not where the depth says it is - check eval/niah.py depth "
            "placement before trusting any other number in the project."
        )
        return 1, lines
    return 0, lines + ["PASS: scored eviction beats structural eviction mid-sequence, as expected"]


def gate_phase3(records, *, model) -> tuple[int, list[str]]:
    """The long-context run held its invariants the whole way through."""
    rows = [r for r in _rows(records, model=model)
            if r.get("context_len") == 8192 and "error" not in r
            and r.get("method") in ("sr_kv", "centroid_merge", "snapkv_unified")]
    if not rows:
        return 1, ["no 8k records; run `make phase3`."]

    lines = [f"{len(rows)} records at 8k across {len({r['method'] for r in rows})} conditions"]
    broken = [r["task_id"] for r in rows if not r.get("conservation_ok", False)]
    over = [r["task_id"] for r in rows if r.get("budget_used_pct_max", 0) > 100.01]
    if broken:
        lines.append(f"FAIL: token accounting broke on {len(broken)} task(s): {broken[:3]}")
    if over:
        lines.append(f"FAIL: cache exceeded its budget on {len(over)} task(s): {over[:3]}")
    if broken or over:
        return 1, lines
    return 0, lines + ["PASS: conservation held and the budget was never exceeded"]


def _resolve_phase4_arms(rows, *, run_keys=None):
    """Keep one run_key per rope mode: the Phase 4 arm, not a neighbouring sweep.

    Returns ``(rows, resolution)`` where resolution is ``None``, ``("note",
    lines)`` for an automatic exclusion worth printing, or ``("refuse", lines)``.

    When a mode is ambiguous, the modes that are *not* ambiguous define what a
    Phase 4 arm looks like: the three arms of one sweep cover the same task
    grid (same contexts, depths and sample indices), differing only in rope
    mode. So the arm whose task_id set matches the unambiguous arms' is the
    real one, and anything else is a different experiment. If nothing matches,
    or every mode is ambiguous, this refuses rather than guessing.
    """
    by_mode: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_mode[r["rope_position_mode"]][r.get("run_key")].append(r)

    if run_keys:
        wanted = set(run_keys)
        kept = [r for r in rows if r.get("run_key") in wanted]
        if not kept:
            return kept, ("refuse", [f"no records with run_key in {sorted(wanted)}."])
        return kept, None

    ambiguous = {m: keys for m, keys in by_mode.items() if len(keys) > 1}
    if not ambiguous:
        return rows, None

    grids = {
        m: frozenset(r.get("task_id") for r in next(iter(keys.values())))
        for m, keys in by_mode.items() if len(keys) == 1
    }
    if not grids or len(set(grids.values())) != 1:
        detail = "; ".join(
            f"{m}: " + ", ".join(f"{k}={len(v)} records" for k, v in sorted(keys.items()))
            for m, keys in sorted(ambiguous.items())
        )
        return rows, ("refuse", [
            f"refusing to aggregate across run_keys [{detail}]: a rope mode with more than one "
            "run_key means two different experiments were merged (Phase 6's alpha/beta sweep "
            "writes sr_kv records at Phase 4's model, budget and default rope mode). No "
            "unambiguous arm is available to identify Phase 4's own grid, so pass "
            "--run-key <hash> once per mode to say which is which.",
        ])

    reference = next(iter(grids.values()))
    chosen: dict[str, str] = {}
    notes: list[str] = []
    for mode, keys in by_mode.items():
        matching = [k for k, v in keys.items()
                    if frozenset(r.get("task_id") for r in v) == reference]
        if len(matching) != 1:
            detail = ", ".join(f"{k}={len(v)} records" for k, v in sorted(keys.items()))
            return rows, ("refuse", [
                f"refusing to aggregate across run_keys for {mode} [{detail}]: "
                f"{len(matching)} of them cover Phase 4's task grid, so which one is the "
                "Phase 4 arm is genuinely ambiguous. Pass --run-key <hash> to pick.",
            ])
        chosen[mode] = matching[0]
        dropped = sorted(k for k in keys if k != matching[0])
        if dropped:
            notes.append(
                f"  {mode}: using run_key {matching[0]} ({len(keys[matching[0]])} records); "
                f"excluded {len(dropped)} run_key(s) on a different task grid "
                f"({', '.join(dropped)}) - a different experiment, not extra samples"
            )

    kept = [r for r in rows if chosen.get(r["rope_position_mode"]) == r.get("run_key")]
    return kept, ("note", ["scoped to one run_key per mode:"] + notes)


def gate_phase4(records, *, model, budgets=None, scoring=None, run_keys=None) -> tuple[int, list[str]]:
    """All three conventions ran, and the best one beats the worst by more than noise."""
    rows = [
        r for r in records
        if r.get("method") in ("sr_kv", "centroid_merge") and r.get("rope_position_mode")
        and (not model or r.get("model") == model) and "error" not in r
    ]

    # Different budgets are different experiments. Averaging a saturated
    # budget=0.3 sweep (every record 1.0) together with a budget=0.1 sweep
    # produces a blended number that describes neither, and silently changes
    # which mode "wins" depending on what happens to be sitting in results/.
    present = sorted({r.get("budget") for r in rows}, key=lambda b: (b is None, b))
    if budgets:
        rows = [r for r in rows if r.get("budget") in set(budgets)]
        present = sorted({r.get("budget") for r in rows}, key=lambda b: (b is None, b))
    if len(present) > 1:
        return 1, [
            f"refusing to aggregate across budgets {present}: these are separate experiments, "
            "and averaging them changes which mode appears to win. Re-run with "
            f"--budget {present[0]} (repeatable) to pick one."
        ]

    # The same reasoning one level down, and the same bug a second time.
    # Phase 6's alpha/beta sweep writes `method=sr_kv` records at the very
    # model, budget and (default) rope mode Phase 4 used, so the rope-mode
    # filter above silently folded 270 sweep records into the attn_weighted
    # arm alone - turning a 100-vs-100 comparison into 100-vs-370 and moving
    # attn_weighted from 0.770 to 0.700. That is not a larger Phase 4 sample,
    # it is a different experiment sharing a directory. Phase 4 varied exactly
    # one knob; anything that varies another is out of scope for this gate.
    def _knobs(r):
        return (r.get("alpha"), r.get("beta"), r.get("lam"))

    settings = sorted({_knobs(r) for r in rows}, key=lambda t: tuple((v is None, v) for v in t))
    if scoring:
        rows = [r for r in rows if _knobs(r) == tuple(scoring)]
        settings = sorted({_knobs(r) for r in rows},
                          key=lambda t: tuple((v is None, v) for v in t))
        if not rows:
            return 1, [f"no records with (alpha, beta, lam) == {tuple(scoring)}."]
    if len(settings) > 1:
        shown = ", ".join(f"alpha={a} beta={b} lam={l}" for a, b, l in settings)
        a, b, l = settings[0]
        return 1, [
            f"refusing to aggregate across scoring settings [{shown}]: Phase 4 held alpha/beta/lam "
            "fixed and varied only rope_position_mode, so records from a hyperparameter sweep "
            "(Phase 6) are a different experiment, not extra samples. Re-run with "
            f"--alpha {a} --beta {b} --lam {l} to pick one."
        ]

    # Pinning the knobs is still not enough on its own: Phase 6's sweep has a
    # cell at exactly Phase 4's defaults (alpha=1.0, beta=0.3, lam=0.001), so
    # 30 more records land in attn_weighted and nowhere else. What actually
    # separates the two is `run_key` - the hash of the settings a single
    # `eval/run.py` invocation was launched with, which differs because Phase 4
    # passed --rope_position_mode explicitly and Phase 6 passed --alpha/--beta.
    # A Phase 4 arm is one run_key covering one task grid; resume and sharding
    # preserve that, so "one mode, several run_keys" always means two
    # experiments were merged, never one experiment run twice.
    rows, resolution = _resolve_phase4_arms(rows, run_keys=run_keys)
    if resolution and resolution[0] == "refuse":
        return 1, resolution[1]

    scores: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        scores[r["rope_position_mode"]].append(r["accuracy"])

    missing = [m for m in POSITION_MODES if not scores.get(m)]
    if missing:
        return 1, [f"missing results for mode(s) {missing}; run `make phase4`."]

    lines_prefix = list(resolution[1]) if resolution and resolution[0] == "note" else []

    means = {m: _mean(scores[m]) for m in POSITION_MODES}
    lines = lines_prefix + [
        f"budget={present[0]}: " + ", ".join(f"{m}={means[m]:.3f}" for m in POSITION_MODES)
    ]
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    if means[best] <= CHANCE_CEILING:
        lines.append(
            "FAIL: every mode is at or below chance. Three modes collapsing together points at "
            "clustering or centroid construction, upstream of position assignment. Fix that "
            "before freezing a winner - otherwise you freeze noise."
        )
        return 1, lines

    # Is the gap real, or one flipped sample? At n=25 a single record moves a
    # mean by 0.04, so a bare spread cannot answer that on its own.
    cb, cw = binary_counts(scores[best]), binary_counts(scores[worst])
    if cb and cw and best != worst:
        p = fisher_exact_2x2(cb[0], cb[1] - cb[0], cw[0], cw[1] - cw[0])
        lines.append(
            f"best={best} ({cb[0]}/{cb[1]}) vs worst={worst} ({cw[0]}/{cw[1]}): "
            f"Fisher exact p={p:.3f}"
        )
        if p >= SIGNIFICANCE_ALPHA:
            lines.append(
                f"NO SIGNIFICANT DIFFERENCE (p={p:.3f} >= {SIGNIFICANCE_ALPHA}). The modes are "
                "not distinguishable at this sample size, so there is no measured winner to "
                "freeze. Either raise --n_samples until the comparison has power, or move the "
                "budget to where accuracy is mid-range (a sweep saturated at 1.0 or floored "
                "near 0 cannot separate the conventions), or pick a default on principled "
                "grounds and document it as 'not empirically distinguished'."
            )
            return 2, lines

    spread = means[best] - min(means.values())
    lines.append(f"winner: {best} (spread across modes {spread:.3f})")
    if spread < 0.02:
        lines.append("note: modes are within 2 points - report as 'no strong effect', not a win")
    return 0, lines + ["PASS: run `make freeze-rope` to write the winner into configs/defaults.yaml"]


def gate_phase5(records, *, model, budgets, n_samples) -> tuple[int, list[str]]:
    """Full grid present, and SR-KV is not below both of its ablations."""
    missing = check_completeness(
        records, model=model, budgets=budgets, methods=FACTORIAL,
        contexts=DEFAULT_CONTEXTS, depths=DEFAULT_DEPTHS, n_samples=n_samples, lb_tasks=[],
    )
    if missing:
        lines = [f"INCOMPLETE: {len(missing)} missing NIAH cell(s)"]
        lines += [f"  {m}" for m in missing[:10]]
        if len(missing) > 10:
            lines.append(f"  ... and {len(missing) - 10} more")
        lines.append("Re-run `make phase5` - it resumes and fills only the gaps.")
        return 1, lines

    findings = check_ablation(records, model=model)
    lines = ["NIAH grid complete"]
    for f in findings:
        lines.append(
            f"  budget={f['budget']}: sr_kv={f['sr_kv']:.3f} "
            f"hard_evict={f['snapkv_unified']:.3f} centroid_merge={f['centroid_merge']:.3f}"
            f" -> {f['status']}"
        )
    if any(f["status"] == "FLAG" for f in findings):
        lines.append(
            "FLAG: SR-KV came out below BOTH ablations. Do not report this quietly. Check the "
            "RoPE position assignment and the centroid re-clustering first; if the code is "
            "right, this is a real negative finding and belongs in the report with an "
            "explanation, not omitted."
        )
        return 2, lines
    return 0, lines + ["PASS"]


def gate_phase6(records, *, model, budgets, n_samples=3) -> tuple[int, list[str]]:
    """The 1.5B-tuned config transferred to 3B without OOM or schema drift.

    Filters to phase6-3b's own (method, context_len) shape before counting
    anything, rather than "any record for this model" - a plain
    `_rows(records, model=model)` filter passed on 3 completely unrelated
    records (Phase 1's qwen2.5-3b 4-bit sanity check, ctx=512) as if they
    were phase6-3b evidence, because they happened to share a model name and
    the required schema fields. Confirmed on a real run: gate6 printed PASS
    while phase6-3b itself had never been run.
    """
    rows = [
        r for r in _rows(records, model=model)
        if r.get("method") in PHASE6_METHODS and r.get("context_len") in PHASE6_CONTEXTS
    ]

    # Precision is not a detail here. Phase 6 ran 4-bit first because bf16 OOM'd
    # at 8192, then moved to bf16 once 4-bit proved to hang; 57 four-bit records
    # already exist. 4-bit and bf16 answer the same cell with different numbers,
    # so counting them together would report a complete grid built from two
    # different experiments - the Phase 4 contamination bug, one axis over.
    precisions = sorted({r.get("precision") for r in rows if "error" not in r},
                        key=lambda p: (p is None, p))
    if len(precisions) > 1:
        return 1, [
            f"refusing to mix precisions {precisions}: these are different experiments, "
            "not extra samples of one. Keep the superseded precision out of results/ "
            "(results/diagnostics/ is excluded from load_records) and re-run the gate."
        ]

    missing = check_completeness(
        rows, model=model, budgets=budgets, methods=PHASE6_METHODS,
        contexts=PHASE6_CONTEXTS, depths=DEFAULT_DEPTHS, n_samples=n_samples, lb_tasks=[],
    )
    if missing:
        lines = [f"INCOMPLETE: {len(missing)} missing phase6-3b cell(s)"]
        lines += [f"  {m}" for m in missing[:10]]
        if len(missing) > 10:
            lines.append(f"  ... and {len(missing) - 10} more")
        lines.append("Re-run `make phase6-3b` - it resumes and fills only the gaps.")
        return 1, lines

    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    lines = [f"{len(ok)} successful records for {model}, {len(errors)} failed"]
    if errors:
        lines.append(
            f"FAIL: {len(errors)} task(s) failed ({sorted({r.get('error') for r in errors})}). "
            "Retry with --precision 4bit and record the precision change in the report rather "
            "than dropping the cells."
        )
        return 1, lines

    required = {"accuracy", "max_memory_allocated", "tokens_per_sec", "cache_stats", "budget"}
    drifted = [r.get("task_id") for r in ok if not required.issubset(r)]
    if drifted:
        lines.append(
            f"FAIL: {len(drifted)} record(s) miss required fields, so plotting would need "
            "per-model special-casing: " + str(drifted[:3])
        )
        return 1, lines

    for budget in budgets:
        lines.append(f"  budget={budget}: {len([r for r in ok if r.get('budget') == budget])} records")
    return 0, lines + ["PASS: same schema as the 1.5B runs, no failures"]


def gate_phase7(figures_dir: Path) -> tuple[int, list[str]]:
    """Every figure family exists and none is a silently empty canvas."""
    figures = sorted(figures_dir.glob("*.png"))
    if not figures:
        return 1, [f"no figures in {figures_dir}; run `make phase7`."]

    names = {p.name for p in figures}
    families = {
        "pareto": any(n.startswith("pareto_") for n in names),
        "niah_heatmap": any(n.startswith("niah_heatmap_") for n in names),
        "ablation": any(n.startswith("ablation_") for n in names),
        "rope_modes": "rope_position_modes.png" in names,
    }
    lines = [f"{len(figures)} figure(s) in {figures_dir}"]
    missing = [k for k, present in families.items() if not present]
    empty = [p.name for p in figures if p.stat().st_size <= 8000]
    if missing:
        lines.append(f"FAIL: missing figure families: {missing}")
    if empty:
        lines.append(f"FAIL: these look like empty plots (<8KB): {empty}")
    if missing or empty:
        return 1, lines
    return 0, lines + ["PASS: all four families present and non-trivially sized"]


GATES = {1: "harness sanity", 2: "baseline failure patterns", 3: "8k invariants",
         4: "RoPE ablation", 5: "factorial matrix", 6: "3B transfer", 7: "figures"}


def run_gate(phase: int, records, *, model, budgets, n_samples, figures_dir,
             explicit_budgets=None, scoring=None, run_keys=None):
    """`explicit_budgets` is what the user actually typed, or None.

    Phase 4 needs that distinction: the other gates are happy with the [0.3]
    default, but silently filtering a RoPE sweep to a budget nobody asked for
    would hide the whole sweep. It auto-detects instead, and only refuses when
    more than one budget is genuinely present.
    """
    if phase == 1:
        return gate_phase1(records, model=model)
    if phase == 2:
        return gate_phase2(records, model=model)
    if phase == 3:
        return gate_phase3(records, model=model)
    if phase == 4:
        return gate_phase4(records, model=model, budgets=explicit_budgets, scoring=scoring,
                           run_keys=run_keys)
    if phase == 5:
        return gate_phase5(records, model=model, budgets=budgets, n_samples=n_samples)
    if phase == 6:
        return gate_phase6(records, model=model, budgets=budgets, n_samples=n_samples)
    if phase == 7:
        return gate_phase7(figures_dir)
    raise ValueError(f"no gate defined for phase {phase}; known: {sorted(GATES)}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["completeness", "ablation", "gate"])
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget", type=float, action="append", default=None)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--context", type=int, action="append", default=None)
    parser.add_argument("--depth", type=int, action="append", default=None)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--lb-task", action="append", default=None)
    parser.add_argument("--skip-longbench", action="store_true")
    parser.add_argument("--phase", type=int, default=None, help="gate: which phase to check")
    # Phase 4 only: pin the scoring knobs it held fixed, so a Phase 6 sweep
    # sharing results/ cannot be counted as extra Phase 4 samples.
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--run-key", action="append", default=None,
                        help="gate 4: pin which run_key is each mode's arm")
    args = parser.parse_args(argv)

    records = load_records(Path(args.results_dir), include_errors=(args.command == "gate"))
    print(f"loaded {len(records)} records from {args.results_dir}")

    if args.command == "gate":
        if args.phase is None:
            parser.error("gate needs --phase N")
        code, lines = run_gate(
            args.phase, records,
            model=args.model,
            budgets=args.budget or [0.3],
            n_samples=args.n_samples,
            figures_dir=Path(args.figures_dir),
            explicit_budgets=args.budget,
            scoring=(
                (args.alpha, args.beta, args.lam)
                if None not in (args.alpha, args.beta, args.lam) else None
            ),
            run_keys=args.run_key,
        )
        print(f"\n=== PHASE {args.phase} GATE ({GATES.get(args.phase, '?')}) ===")
        for line in lines:
            if line:
                print(line)
        verdict = {0: "pass", 1: "failed / incomplete", 2: "needs a human look"}[code]
        print(f"=== exit {code}: {verdict} ===")
        return code

    if args.command == "completeness":
        missing = check_completeness(
            records,
            model=args.model,
            budgets=args.budget or [0.3],
            methods=args.method or FACTORIAL,
            contexts=args.context or DEFAULT_CONTEXTS,
            depths=args.depth or DEFAULT_DEPTHS,
            n_samples=args.n_samples,
            lb_tasks=[] if args.skip_longbench else (args.lb_task or DEFAULT_LB_TASKS),
        )
        if missing:
            print(f"\nINCOMPLETE - {len(missing)} missing cell(s):\n")
            for line in missing:
                print("  " + line)
            print("\nRun the missing cells before plotting or writing up.")
            return 1
        print("\nCOMPLETE - every expected cell has results.")
        return 0

    findings = check_ablation(records, model=args.model)
    if not findings:
        print("no NIAH records to compare")
        return 1

    print(f"\n{'budget':<8}{'sr_kv':>9}{'hard-evict':>13}{'centroid':>11}{'margin':>10}  status")
    flagged = False
    for f in findings:
        margin = f.get("margin_vs_best_ablation", float("nan"))
        print(f"{f['budget']:<8}{f['sr_kv']:>9.3f}{f['snapkv_unified']:>13.3f}"
              f"{f['centroid_merge']:>11.3f}{margin:>10.3f}  {f['status']}")
        flagged |= f["status"] == "FLAG"

    if flagged:
        print(
            "\nFLAG: SR-KV (full) came out below BOTH ablated conditions. Do not quietly report\n"
            "this. Either it is a bug (check RoPE position assignment and centroid re-clustering\n"
            "first) or it is a real negative finding about the recency+merge combination, which\n"
            "belongs in the report with an explanation."
        )
        return 2
    print("\nablation ordering is sane (SR-KV is not below both ablations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
