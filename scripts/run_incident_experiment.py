"""Fixed-capacity streaming incident experiment, one disjoint worker shard."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # noqa: F401,E402
import torch
import yaml
from eval.incident_stream import CAPACITY, CHUNK, METHODS, build_streams, measure_stream
from scripts.checkpoint_utils import ResultStore, free_cuda_memory, run_key
from src.caches import make_cache
from src.models import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("sanity", "main"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--output-dir", default="results/incident_stream_v1")
    args = p.parse_args()
    if not 0 <= args.shard < args.num_shards:
        p.error("Invalid shard")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "main" and not json.loads((out / "sanity_gate.json").read_text())["pass"]:
        raise RuntimeError("Sanity gate must pass before main")
    n, seed = (4, 31003) if args.stage == "sanity" else (24, 41003)
    streams = build_streams(n=n, seed=seed)
    dataset_text = "\n".join(json.dumps(s.to_dict(), sort_keys=True) for s in streams) + "\n"
    data_hash = hashlib.sha256(dataset_text.encode()).hexdigest()
    dataset_path = out / f"{args.stage}_dataset.jsonl"
    if dataset_path.exists() and dataset_path.read_text(encoding="utf-8") != dataset_text:
        raise RuntimeError("Frozen dataset mismatch")
    if not dataset_path.exists():
        dataset_path.write_text(dataset_text, encoding="utf-8")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model, tokenizer = load_model("qwen2.5-1.5b", precision="fp16", context_len=8192)
    defaults = yaml.safe_load((ROOT / "configs/defaults.yaml").read_text())
    policy = {k: defaults[k] for k in ("alpha", "beta", "lam", "obs_window", "pool_kernel",
                                      "n_sink", "centroid_frac", "cluster_mode", "rope_position_mode")}
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "eval/incident_stream.py", Path(__file__)]
    metadata = dict(stage=args.stage, seed=seed, dataset_sha256=data_hash, capacity=CAPACITY,
                    chunk_size=CHUNK, policy=policy, model="qwen2.5-1.5b", precision="fp16",
                    model_commit=getattr(model.config, "_commit_hash", None),
                    torch=torch.__version__, transformers=importlib.metadata.version("transformers"),
                    gpu=torch.cuda.get_device_name(0),
                    source_sha256=hashlib.sha256(b"".join(f.read_bytes() for f in source_paths)).hexdigest())
    store = ResultStore(out / f"{args.stage}_part{args.shard}.json")
    with torch.no_grad():
        model(**tokenizer("Warmup.", return_tensors="pt").to(next(model.parameters()).device))
    for stream in streams[args.shard::args.num_shards]:
        for method in METHODS:
            key = run_key(**metadata, method=method)
            ids = [f"incident/{seed}/{stream.stream_id}/{checkpoint}" for checkpoint in range(3)]
            if all(store.is_done(task_id, key) for task_id in ids):
                continue
            capacity = None if method == "full" else CAPACITY
            cache = make_cache(method, model=model, budget=1.0, max_capacity=capacity, **policy)
            for result in measure_stream(model, tokenizer, stream, cache):
                record = dict(task_id=ids[result["checkpoint"]], run_key=key, method=method,
                              stream_id=stream.stream_id, scenario=stream.scenario,
                              content_hash=stream.content_hash, dataset_sha256=data_hash,
                              seed=seed, **result)
                store.append(record)
                print(f"[{args.stage} shard={args.shard}] saved={len(store.records)} stream={stream.stream_id} "
                      f"method={method} checkpoint={result['checkpoint']}", flush=True)
            del cache
            free_cuda_memory()
    store.finalize(metadata=metadata)
    print("WORKER COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
