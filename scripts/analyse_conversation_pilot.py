"""Paired, conversation-level analysis of the frozen exploratory pilot."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def paired_interval(treatment, control, *, seed=260928):
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
        identities[row["sample_idx"]].add(row["context_sha256"])
        groups[(row["method"], row["budget"])].append(row)
    if any(len(hashes) != 1 for hashes in identities.values()):
        raise ValueError("Methods/questions were evaluated on different conversations")
    summary, conversation_scores = {}, {}
    for (method, budget), records in sorted(groups.items()):
        conv = defaultdict(list)
        for r in records:
            conv[r["sample_idx"]].append(r)
        if any(len(v) != 2 or {r["variant"] for r in v} != {"current_request", "old_constraint"}
               for v in conv.values()):
            raise ValueError("Incomplete or duplicate conversation/question pair")
        key = f"{method}/b{budget}"
        scores = {i: float(np.mean([r["accuracy"] for r in rr])) for i, rr in conv.items()}
        conversation_scores[(method, budget)] = scores
        summary[key] = dict(
            n_conversations=len(conv), accuracy=float(np.mean(list(scores.values()))),
            both_correct=float(np.mean([all(r["accuracy"] == 1 for r in rr) for rr in conv.values()])),
            current_request=float(np.mean([r["accuracy"] for r in records if r["variant"] == "current_request"])),
            old_constraint=float(np.mean([r["accuracy"] for r in records if r["variant"] == "old_constraint"])),
            mean_seconds=float(np.mean([r["seconds"] for r in records])),
            mean_compression_seconds=float(np.mean([r["compression_seconds"] for r in records])),
            peak_gpu_gib=max(r["max_memory_allocated"] for r in records) / 1024**3,
        )
    comparisons = {}
    for budget in (0.1, 0.2):
        treatment = conversation_scores.get(("sr_kv", budget))
        if treatment is None:
            continue
        for baseline in ("recency_hard_evict", "snapkv_unified", "streaming_llm", "full"):
            b = 1.0 if baseline == "full" else budget
            control = conversation_scores.get((baseline, b))
            if control is not None:
                comparisons[f"sr_kv minus {baseline}/b{budget}"] = paired_interval(treatment, control)
    encouraged = True
    for b in (0.1, 0.2):
        c = comparisons.get(f"sr_kv minus recency_hard_evict/b{b}", {})
        a, ref = summary.get(f"sr_kv/b{b}", {}), summary.get(f"recency_hard_evict/b{b}", {})
        encouraged &= (c.get("n", 0) == 50 and c.get("ci_low", -1) > 0
                       and all(a.get(v, -1) >= ref.get(v, 2) for v in ("current_request", "old_constraint")))
    return dict(summary=summary, paired_comparisons=comparisons,
                exploratory_encouragement=bool(encouraged),
                note="Exploratory synthetic pilot; not evidence of general or production superiority.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    result = analyse(rows)
    path = args.results.with_name(args.results.stem + "_analysis.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
