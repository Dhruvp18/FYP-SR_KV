"""Paired, pre-registered analysis of the consensus-update pilot.

Implements exactly the criterion fixed in
experiments/consensus_update_protocol.md and nothing else: SUPPORTED requires
a dose-response in n_reinforcements (null at n1, a significant sr_kv-over-
recency_hard_evict win at n3 AND n5, with the n5 effect larger than n3's),
holding at both budgets. Anything else prints NOT SUPPORTED. This always
exits 0 - a null is a result to report, not a pipeline failure, same
discipline as scripts/analyse_phase8.py.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS = (0.1, 0.2)
VARIANTS = ("consensus_n1", "consensus_n3", "consensus_n5")
BASELINES = ("recency_hard_evict", "snapkv_unified", "streaming_llm", "centroid_merge", "full")


def paired_interval(treatment, control, *, seed=270903):
    ids = sorted(set(treatment) & set(control))
    if set(treatment) != set(control) or not ids:
        raise ValueError("Comparisons require identical conversation IDs")
    d = np.array([treatment[i] - control[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return dict(n=len(ids), difference=float(d.mean()), ci_low=float(lo), ci_high=float(hi))


def analyse(rows):
    groups = defaultdict(list)
    identities = defaultdict(set)
    for row in rows:
        identities[(row["variant"], row["sample_idx"])].add(row["context_sha256"])
        groups[(row["method"], row["budget"], row["variant"])].append(row)
    if any(len(hashes) != 1 for hashes in identities.values()):
        raise ValueError("Methods disagreed on the passage for the same (variant, sample_idx)")

    summary, scores = {}, {}
    for (method, budget, variant), records in sorted(groups.items()):
        if len(records) != len({r["sample_idx"] for r in records}):
            raise ValueError("Duplicate or missing answers for a cell")
        key = f"{method}/b{budget}/{variant}"
        cell_scores = {r["sample_idx"]: float(r["accuracy"]) for r in records}
        scores[(method, budget, variant)] = cell_scores
        summary[key] = dict(
            n=len(cell_scores), accuracy=float(np.mean(list(cell_scores.values()))),
            mean_seconds=float(np.mean([r["seconds"] for r in records])),
            mean_compression_seconds=float(np.mean([r["compression_seconds"] for r in records])),
            peak_gpu_gib=max(r["max_memory_allocated"] for r in records) / 1024**3,
        )

    comparisons = {}
    for budget in BUDGETS:
        for variant in VARIANTS:
            treatment = scores.get(("sr_kv", budget, variant))
            if treatment is None:
                continue
            for baseline in BASELINES:
                b = 1.0 if baseline == "full" else budget
                control = scores.get((baseline, b, variant))
                if control is not None:
                    comparisons[f"sr_kv minus {baseline}/{variant}/b{budget}"] = paired_interval(
                        treatment, control)
            # secondary: does clustering ALONE (no recency) also win here?
            cm = scores.get(("centroid_merge", budget, variant))
            snap = scores.get(("snapkv_unified", budget, variant))
            if cm is not None and snap is not None:
                comparisons[f"centroid_merge minus snapkv_unified/{variant}/b{budget}"] = (
                    paired_interval(cm, snap))

    def win(budget, variant):
        c = comparisons.get(f"sr_kv minus recency_hard_evict/{variant}/b{budget}")
        return c is not None and c["ci_low"] > 0, c

    def control_collapsed(budget, variant):
        c = comparisons.get(f"sr_kv minus recency_hard_evict/{variant}/b{budget}")
        return c is not None and c["ci_low"] <= 0 <= c["ci_high"], c

    per_budget_verdict = {}
    for budget in BUDGETS:
        n1_ok, c_n1 = control_collapsed(budget, "consensus_n1")
        n3_ok, c_n3 = win(budget, "consensus_n3")
        n5_ok, c_n5 = win(budget, "consensus_n5")
        dose_shape = bool(c_n3 and c_n5 and c_n5["difference"] > c_n3["difference"])
        per_budget_verdict[f"b{budget}"] = dict(
            n1_control_collapsed=n1_ok, n3_win=n3_ok, n5_win=n5_ok,
            dose_response_shape=dose_shape,
            supported=bool(n1_ok and n3_ok and n5_ok and dose_shape),
        )

    overall_supported = all(v["supported"] for v in per_budget_verdict.values())

    # Diagnostic flag, not part of the verdict (protocol: report, don't
    # silently fold into the headline): does clustering alone explain a
    # comparable amount of the win, independent of recency?
    clustering_alone_hits = sum(
        1 for k, c in comparisons.items()
        if k.startswith("centroid_merge minus snapkv_unified/") and c["ci_low"] > 0
    )

    return dict(
        summary=summary,
        paired_comparisons=comparisons,
        per_budget_verdict=per_budget_verdict,
        supported=bool(overall_supported),
        clustering_alone_significant_wins=clustering_alone_hits,
        note=("Exploratory pilot, deliberately designed to favour SR-KV "
              "(experiments/consensus_update_protocol.md). Not evidence of "
              "general or production superiority even if SUPPORTED."),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    result = analyse(rows)
    path = args.results.with_name(args.results.stem + "_analysis.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print()
    print("VERDICT:", "SUPPORTED" if result["supported"] else "NOT SUPPORTED")


if __name__ == "__main__":
    main()
