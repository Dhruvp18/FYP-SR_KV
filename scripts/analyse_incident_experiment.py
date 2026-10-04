"""Validate complete streaming records and compare methods paired by stream."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = ("full", "streaming_llm", "snapkv_unified", "recency_hard_evict", "centroid_merge", "sr_kv")
CLASSES = ("database", "payments", "auth", "queue", "NONE")


def analyse(rows, n_streams):
    expected_keys = {(m, s, c) for m in METHODS for s in range(n_streams) for c in range(3)}
    keys = [(r["method"], r["stream_id"], r["checkpoint"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Incomplete or duplicate method/stream/checkpoint grid")
    hashes = defaultdict(set)
    labels = defaultdict(set)
    for r in rows:
        hashes[r["stream_id"]].add(r["content_hash"])
        labels[r["stream_id"], r["checkpoint"]].add(r["expected"])
        if not r["conservation_ok"]:
            raise ValueError("Cache accounting failed")
        if r["method"] != "full" and (r["capacity"] != 512 or r["cache_stats"]["n_tokens_cached"] != 512):
            raise ValueError("Fixed-capacity check failed")
        if r["accuracy"] != float(r["predicted"] == r["expected"]):
            raise ValueError("Invalid saved score")
    if any(len(v) != 1 for v in hashes.values()) or any(len(v) != 1 for v in labels.values()):
        raise ValueError("Methods did not receive identical streams and labels")
    summary, by_stream = {}, {}
    for method in METHODS:
        rr = [r for r in rows if r["method"] == method]
        by_stream[method] = {s: float(np.mean([r["accuracy"] for r in rr if r["stream_id"] == s])) for s in range(n_streams)}
        active = [r for r in rr if r["expected"] != "NONE"]
        healthy = [r for r in rr if r["expected"] == "NONE"]
        f1 = []
        for cls in CLASSES:
            tp = sum(r["expected"] == cls and r["predicted"] == cls for r in rr)
            fp = sum(r["expected"] != cls and r["predicted"] == cls for r in rr)
            fn = sum(r["expected"] == cls and r["predicted"] != cls for r in rr)
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        summary[method] = dict(
            n=len(rr), accuracy=float(np.mean([r["accuracy"] for r in rr])), macro_f1=float(np.mean(f1)),
            missed_alert_rate=sum(r["predicted"] == "NONE" for r in active) / len(active),
            false_alert_rate=sum(r["predicted"] not in ("NONE", "INVALID") for r in healthy) / len(healthy),
            wrong_service_rate=sum(r["predicted"] in CLASSES[:-1] and r["predicted"] != r["expected"] for r in active) / len(active),
            invalid_rate=sum(r["predicted"] == "INVALID" for r in rr) / len(rr),
            per_checkpoint={str(c): float(np.mean([r["accuracy"] for r in rr if r["checkpoint"] == c])) for c in range(3)},
            per_scenario={s: float(np.mean([r["accuracy"] for r in rr if r["scenario"] == s])) for s in sorted({r["scenario"] for r in rr})},
            mean_ingest_seconds=float(np.mean([r["ingest_seconds"] for r in rr])),
            mean_probe_seconds=float(np.mean([r["probe_seconds"] for r in rr])),
            peak_gpu_gib=max(r["peak_gpu_bytes"] for r in rr) / 1024**3,
            max_retained_kv_mib=max(r["retained_kv_bytes"] for r in rr) / 1024**2,
        )
    rng = np.random.default_rng(51003)
    indices = rng.integers(0, n_streams, size=(10000, n_streams))
    comparisons = {}
    for method in METHODS:
        if method == "sr_kv":
            continue
        differences = np.array([by_stream["sr_kv"][s] - by_stream[method][s] for s in range(n_streams)])
        boot = differences[indices].mean(axis=1)
        comparisons[method] = dict(difference=float(differences.mean()),
                                    ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                                    ci_adjusted=np.quantile(boot, [0.00625, 0.99375]).tolist())
    wins_all = all(comparisons[m]["ci_adjusted"][0] > 0 for m in METHODS if m not in ("full", "sr_kv"))
    return dict(n_records=len(rows), n_streams=n_streams, summary=summary, comparisons=comparisons,
                exploratory_advantage=bool(n_streams == 24 and wins_all and summary["sr_kv"]["accuracy"] >= 0.8),
                scope="Synthetic incident lifecycle identification; not causal diagnosis or production validation.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("sanity", "main"), required=True)
    p.add_argument("--output-dir", default="results/incident_stream_v1")
    args = p.parse_args()
    root = Path(args.output_dir)
    rows = [json.loads(line) for f in sorted(root.glob(f"{args.stage}_part*.jsonl")) for line in f.read_text().splitlines() if line.strip()]
    n = 4 if args.stage == "sanity" else 24
    result = analyse(rows, n)
    (root / f"{args.stage}_analysis.json").write_text(json.dumps(result, indent=2))
    if args.stage == "sanity":
        passed = result["summary"]["full"]["accuracy"] >= 10 / 12
        (root / "sanity_gate.json").write_text(json.dumps({"pass": passed, "full_accuracy": result["summary"]["full"]["accuracy"]}))
        if not passed:
            print("SANITY STOP: full-cache model does not solve the task reliably", flush=True)
            return 2
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
