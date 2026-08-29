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
DEFAULT_CONTEXTS = [2048, 4096, 8192, 16384]
DEFAULT_DEPTHS = [0, 25, 50, 75, 100]
DEFAULT_LB_TASKS = ["narrativeqa", "qasper", "gov_report", "triviaqa"]

#: a 1.5B instruct model must clear this on a 512-token retrieval
SANITY_ACCURACY = 0.9
#: below this, "the best RoPE mode" is indistinguishable from noise
CHANCE_CEILING = 0.05


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


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


def gate_phase4(records, *, model) -> tuple[int, list[str]]:
    """All three conventions ran, and at least one is above chance."""
    scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("method") in ("sr_kv", "centroid_merge") and r.get("rope_position_mode"):
            if (not model or r.get("model") == model) and "error" not in r:
                scores[r["rope_position_mode"]].append(r["accuracy"])

    missing = [m for m in POSITION_MODES if not scores.get(m)]
    if missing:
        return 1, [f"missing results for mode(s) {missing}; run `make phase4`."]

    means = {m: _mean(scores[m]) for m in POSITION_MODES}
    lines = [", ".join(f"{m}={means[m]:.3f}" for m in POSITION_MODES)]
    best = max(means, key=means.get)
    if means[best] <= CHANCE_CEILING:
        lines.append(
            "FAIL: every mode is at or below chance. Three modes collapsing together points at "
            "clustering or centroid construction, upstream of position assignment. Fix that "
            "before freezing a winner - otherwise you freeze noise."
        )
        return 1, lines

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


def gate_phase6(records, *, model, budgets) -> tuple[int, list[str]]:
    """The 1.5B-tuned config transferred to 3B without OOM or schema drift."""
    rows = _rows(records, model=model)
    if not rows:
        return 1, [f"no records for {model}; run `make phase6-3b`."]

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


def run_gate(phase: int, records, *, model, budgets, n_samples, figures_dir):
    if phase == 1:
        return gate_phase1(records, model=model)
    if phase == 2:
        return gate_phase2(records, model=model)
    if phase == 3:
        return gate_phase3(records, model=model)
    if phase == 4:
        return gate_phase4(records, model=model)
    if phase == 5:
        return gate_phase5(records, model=model, budgets=budgets, n_samples=n_samples)
    if phase == 6:
        return gate_phase6(records, model=model, budgets=budgets)
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
