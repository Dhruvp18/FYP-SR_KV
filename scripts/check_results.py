"""Validate a finished sweep before anything is plotted or written up.

    python scripts/check_results.py completeness --model qwen2.5-1.5b --budget 0.3
    python scripts/check_results.py ablation     --model qwen2.5-1.5b

`completeness` fails loudly, listing every missing (condition, context, depth)
cell, rather than letting `make_plots.py` quietly draw a figure from half a
sweep.

`ablation` runs the Phase 5 sanity assertion: SR-KV (full) should not be worse
than *both* SnapKV-style hard eviction and centroid-merge-without-recency, at
the same budget, averaged over NIAH depths. If it is, that is either a bug or a
real negative result - both worth knowing, neither worth hiding - so this
surfaces it as a FLAG with the numbers attached instead of failing silently or
pretending it did not happen.
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

FACTORIAL = ["streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"]
DEFAULT_CONTEXTS = [2048, 4096, 8192, 16384]
DEFAULT_DEPTHS = [0, 25, 50, 75, 100]
DEFAULT_LB_TASKS = ["narrativeqa", "qasper", "gov_report", "triviaqa"]


def _mean(values) -> float:
    values = [v for v in values]
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
def check_completeness(records, *, model, budgets, methods, contexts, depths, n_samples,
                       lb_tasks) -> list[str]:
    """Return a list of human-readable descriptions of missing cells."""
    have = defaultdict(int)
    for r in records:
        if model and r.get("model") != model:
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
                got = have.get((method, budget, task, None), 0)
                if got == 0:
                    missing.append(f"LB    {method:<16} budget={budget:<5} task={task} 0 samples")
    return missing


def check_ablation(records, *, model) -> list[dict]:
    """Compare SR-KV against its two ingredient-ablated siblings per budget."""
    rows = [r for r in records if (not model or r.get("model") == model)]
    rows = [r for r in rows if r.get("context_len") is not None]

    by = defaultdict(list)
    for r in rows:
        by[(r["budget"], r["method"])].append(r["accuracy"])

    findings = []
    budgets = sorted({b for b, _ in by})
    for budget in budgets:
        sr = _mean(by.get((budget, "sr_kv"), []))
        hard = _mean(by.get((budget, "snapkv_unified"), []))
        merge = _mean(by.get((budget, "centroid_merge"), []))
        if any(v != v for v in (sr, hard, merge)):  # NaN => missing condition
            findings.append({"budget": budget, "status": "INCOMPLETE",
                             "sr_kv": sr, "snapkv_unified": hard, "centroid_merge": merge})
            continue
        worse_than_both = sr < hard and sr < merge
        findings.append({
            "budget": budget,
            "status": "FLAG" if worse_than_both else "ok",
            "sr_kv": sr,
            "snapkv_unified": hard,
            "centroid_merge": merge,
            "margin_vs_best_ablation": sr - max(hard, merge),
        })
    return findings


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["completeness", "ablation"])
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget", type=float, action="append", default=None)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--context", type=int, action="append", default=None)
    parser.add_argument("--depth", type=int, action="append", default=None)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--lb-task", action="append", default=None)
    parser.add_argument("--skip-longbench", action="store_true")
    args = parser.parse_args(argv)

    records = load_records(Path(args.results_dir))
    print(f"loaded {len(records)} records from {args.results_dir}")

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
            "\nFLAG: SR-KV (full) came out below BOTH ablated conditions at the budget(s)\n"
            "above. Do not quietly report this. Either it is a bug (check the RoPE\n"
            "position assignment and the centroid re-clustering first) or it is a real\n"
            "negative finding about the recency+merge combination, which belongs in the\n"
            "report with an explanation."
        )
        return 2
    print("\nablation ordering is sane (SR-KV is not below both ablations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
