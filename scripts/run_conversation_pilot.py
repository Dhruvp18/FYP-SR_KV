"""Run the fixed two-question pilot; resumes only an identical dataset/config."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # noqa: F401,E402
import torch
import yaml
from eval import conversation_pilot as dataset
from eval.gist_mcq import measure
from scripts.checkpoint_utils import ResultStore, free_cuda_memory, run_key
from src.caches import make_cache
from src.models import load_model

METHODS = ("full", "streaming_llm", "snapkv_unified", "recency_hard_evict", "sr_kv")


def summarize(records):
    groups = {}
    for r in records:
        key = (r["method"], r["budget"], r["variant"])
        groups.setdefault(key, []).append(r)
    return {f"{m}/b{b}/{v}": {
        "n": len(rows), "accuracy": sum(r["accuracy"] for r in rows) / len(rows),
        "mean_seconds": sum(r["seconds"] for r in rows) / len(rows),
        "peak_gpu_gib": max(r["max_memory_allocated"] for r in rows) / 1024**3,
    } for (m, b, v), rows in groups.items()}


def check_records(records, n_samples):
    expected = n_samples * 2 * 9
    if len(records) != expected:
        raise ValueError(f"Expected {expected} answers, found {len(records)}")
    for r in records:
        if not r["conservation_ok"] or not r["pre_question_conservation_ok"]:
            raise ValueError("Cache accounting failed")
        if r["method"] != "full":
            slots = r["pre_question_cache_stats"]["n_tokens_cached"]
            if slots > r["pre_question_budget_tokens"] or slots >= r["passage_tokens"]:
                raise ValueError("Compression did not meet its pre-question budget")
    full = [r for r in records if r["method"] == "full"]
    return {
        "full_accuracy": sum(r["accuracy"] for r in full) / len(full),
        "full_by_variant": {v: sum(r["accuracy"] for r in full if r["variant"] == v) / n_samples
                            for v in dataset.VARIANTS},
        "records": len(records),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("sanity", "pilot"), required=True)
    p.add_argument("--output-dir", default="results/conversation_pilot_v1")
    args = p.parse_args(argv)
    n, seed = (5, 260926) if args.stage == "sanity" else (50, 260927)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        gate = json.loads((out / "sanity_gate.json").read_text())
        if not gate.get("pass"):
            raise ValueError("Sanity gate has not passed; refusing to run pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the measured experiment")
    torch.manual_seed(seed)
    model, tokenizer = load_model("qwen2.5-1.5b", precision="fp16", context_len=4096)
    defaults = yaml.safe_load((ROOT / "configs/defaults.yaml").read_text())
    policy = {k: defaults[k] for k in (
        "alpha", "beta", "lam", "obs_window", "pool_kernel", "n_sink",
        "centroid_frac", "cluster_mode", "rope_position_mode") if k in defaults}
    samples = dataset.build_samples(tokenizer, n_samples=n, seed=seed)
    dataset_text = "\n".join(json.dumps(s.to_dict(), sort_keys=True) for s in samples) + "\n"
    data_hash = hashlib.sha256(dataset_text.encode()).hexdigest()
    dataset_path = out / f"{args.stage}_dataset.jsonl"
    if dataset_path.exists() and dataset_path.read_text(encoding="utf-8") != dataset_text:
        raise ValueError("Dataset changed under an existing run")
    dataset_path.write_text(dataset_text, encoding="utf-8")
    store = ResultStore(out / f"{args.stage}.json")
    metadata = dict(stage=args.stage, dataset_sha256=data_hash, policy=policy,
                    model="qwen2.5-1.5b", precision="fp16", torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), seed=seed,
                    source_sha256=hashlib.sha256(b"".join(
                        f.read_bytes() for f in sorted((ROOT / "src").rglob("*.py")))).hexdigest())
    # Excluded warmup, on unrelated text and an uncompressed cache.
    warm = tokenizer("A short warmup.", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        model(**warm)
    torch.cuda.synchronize()
    started = time.perf_counter()
    total = n * 18
    for method in METHODS:
        for budget in ([1.0] if method == "full" else [0.1, 0.2]):
            key = run_key(**metadata, method=method, budget=budget)
            for sample in samples:
                if store.is_done(sample.task_id, key):
                    continue
                cache = make_cache(method, model=model, budget=budget, **policy)
                # A new cache per question ensures answer 1 never leaks into 2.
                result = measure(model, tokenizer, sample, cache)
                record = dict(task_id=sample.task_id, run_key=key, method=method,
                              budget=budget, sample_idx=sample.sample_idx, variant=sample.variant,
                              seed=seed, model="qwen2.5-1.5b", precision="fp16",
                              dataset_sha256=data_hash, context_sha256=sample.context_sha256,
                              expected_letter=sample.answer_letter, options=sample.options,
                              correction_depth=sample.correction_depth,
                              constraint_depth=sample.constraint_depth,
                              accuracy=dataset.score(sample, result["generated_text"]), **result)
                store.append(record)
                del cache
                free_cuda_memory()
                done = len(store.records)
                if done % 5 == 0:
                    print(f"[{args.stage}] {done}/{total} answers; {method} b={budget}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    checks = check_records(store.records, n)
    store.finalize(metadata=metadata, summary=summarize(store.records))
    if args.stage == "sanity":
        passed = checks["full_accuracy"] >= 0.9 and min(checks["full_by_variant"].values()) >= 0.8
        (out / "sanity_gate.json").write_text(json.dumps({"pass": passed, **checks}, indent=2))
        print("SANITY " + ("PASS" if passed else "STOP"), checks, flush=True)
        if not passed:
            return 2
    print(json.dumps(summarize(store.records), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
