"""Single CLI entrypoint for every SR-KV experiment.

    python eval/run.py --method sr_kv --model qwen2.5-1.5b \
        --context_len 8192 --budget 0.3 --output results/srkv_8k_b30.json

Sharding for parallel workers / multiple Kaggle accounts:

    python eval/run.py ... --shard 2 --num_shards 4

Everything is resumable: results land in `<output>.jsonl` as they finish, and
re-running the same command skips whatever is already there.

This file contains no eviction logic and never touches key/value tensors - it
builds a cache by name, hands it to `model.generate()` and reads `get_stats()`
(see CLAUDE.md "Forbidden").
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src  # noqa: E402,F401  (environment guards, must precede transformers)
import torch  # noqa: E402

from eval import longbench, niah  # noqa: E402
from eval.memory import build_prompt, generate_and_measure  # noqa: E402
from scripts.checkpoint_utils import (  # noqa: E402
    ResultStore,
    free_cuda_memory,
    run_key,
    shard_tasks,
)
from src.caches import METHODS, make_cache  # noqa: E402


def _csv(kind):
    def parse(value: str):
        return [kind(v) for v in str(value).split(",") if v != ""]

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an SR-KV experiment")
    p.add_argument("--method", type=_csv(str), default=None,
                   help=f"comma-separated; one of {sorted(METHODS)} (or supply via --config)")
    p.add_argument("--model", default="qwen2.5-1.5b")
    p.add_argument("--task", default="niah", choices=["niah", "longbench"])
    p.add_argument("--context_len", type=_csv(int), default=[4096])
    p.add_argument("--budget", type=_csv(float), default=[0.3])
    p.add_argument("--output", default=None,
                   help="results/<name>.json; defaults to the --config stem")
    p.add_argument("--config", default=None,
                   help="YAML from configs/ supplying model/method/budget/context_len")

    p.add_argument("--depths", type=_csv(int), default=list(niah.DEPTHS))
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--niah_corpus", default="synthetic", choices=["synthetic", "pg"])
    p.add_argument("--longbench_tasks", type=_csv(str),
                   default=["narrativeqa", "qasper", "gov_report", "triviaqa"])
    p.add_argument("--max_new_tokens", type=int, default=32)

    # policy hyperparameters (Phase 6 sweeps these)
    p.add_argument("--rope_position_mode", default=None,
                   help="latest|earliest|attn_weighted; defaults to configs/defaults.yaml")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--obs_window", type=int, default=None)
    p.add_argument("--pool_kernel", type=int, default=None)
    p.add_argument("--n_centroids", type=int, default=None)
    p.add_argument("--centroid_frac", type=float, default=None)
    p.add_argument("--cluster_mode", default=None, choices=[None, "kmeans", "temporal_chunk"])
    p.add_argument("--n_sink", type=int, default=None)

    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "4bit"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="stop after N tasks (smoke tests)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow_cpu", action="store_true", help="CPU harness tests only")
    p.add_argument("--tiny", action="store_true",
                   help="use a randomly-initialised 2-layer model (harness self-test)")
    p.add_argument("--continue_on_oom", action="store_true",
                   help="record the OOM and move on instead of raising")
    return p


#: args whose CLI type is a comma-separated list, so a scalar in YAML must be wrapped
_LIST_ARGS = {"method", "context_len", "budget", "depths", "longbench_tasks"}


def apply_config_file(parser, argv, args):
    """Merge a configs/*.yaml into the parsed args; explicit CLI flags win.

    Implemented by re-parsing with the YAML installed as parser defaults, so
    "was this flag actually passed?" is answered by argparse rather than by
    comparing against sentinel values.
    """
    if not args.config:
        return args
    import yaml

    path = Path(args.config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    known = {a.dest for a in parser._actions}
    overrides = {}
    for key, value in raw.items():
        if key not in known:
            continue
        if key in _LIST_ARGS and not isinstance(value, list):
            value = [value]
        overrides[key] = value
    overrides.setdefault("output", f"results/{path.stem}.json")

    parser.set_defaults(**overrides)
    return parser.parse_args(argv)


def load_defaults() -> dict:
    path = REPO_ROOT / "configs" / "defaults.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return {}


def cache_overrides(args, defaults: dict) -> dict:
    """CLI flags win; then configs/defaults.yaml; then the class defaults."""
    out: dict = {}
    for key in ("alpha", "beta", "lam", "obs_window", "pool_kernel", "n_centroids",
                "centroid_frac", "cluster_mode", "n_sink", "rope_position_mode"):
        value = getattr(args, key)
        if value is None:
            value = defaults.get(key)
        if value is not None:
            out[key] = value
    return out


def build_task_list(args) -> list[dict]:
    """The full, ordered work list. Pure function of the arguments.

    Ordering is fixed so that `shard_tasks` partitions deterministically and a
    resumed run rebuilds exactly the same list.
    """
    tasks: list[dict] = []
    for method in args.method:
        for budget in args.budget:
            if args.task == "niah":
                for context_len in args.context_len:
                    for depth in args.depths:
                        for sample_idx in range(args.n_samples):
                            tasks.append({
                                "method": method,
                                "budget": budget,
                                "context_len": context_len,
                                "depth": depth,
                                "sample_idx": sample_idx,
                                "task_id": f"niah/ctx{context_len}/depth{depth}/s{sample_idx}",
                            })
            else:
                for lb_task in args.longbench_tasks:
                    for sample_idx in range(args.n_samples):
                        tasks.append({
                            "method": method,
                            "budget": budget,
                            "lb_task": lb_task,
                            "sample_idx": sample_idx,
                            "task_id": f"longbench/{lb_task}/s{sample_idx}",
                        })
    return tasks


def _load_model(args):
    if args.tiny:
        from src.models import build_tiny_model, build_tiny_tokenizer

        model = build_tiny_model()
        return model, build_tiny_tokenizer()

    from src.models import load_model

    return load_model(
        args.model,
        precision=args.precision,
        context_len=max(args.context_len),
        allow_cpu=args.allow_cpu,
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = apply_config_file(parser, argv, parser.parse_args(argv))
    if not args.method:
        parser.error("--method is required (or supply one via --config)")
    if not args.output:
        parser.error("--output is required (or supply one via --config)")
    defaults = load_defaults()
    overrides = cache_overrides(args, defaults)
    torch.manual_seed(args.seed)

    for method in args.method:
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r}; known: {sorted(METHODS)}")

    tasks = shard_tasks(build_task_list(args), args.shard, args.num_shards)
    if args.limit:
        tasks = tasks[: args.limit]

    store = ResultStore(Path(args.output))
    keys = {
        id(task): run_key(
            method=task["method"],
            model=args.model if not args.tiny else "tiny",
            budget=task["budget"],
            task=args.task,
            max_new_tokens=args.max_new_tokens,
            corpus=args.niah_corpus,
            **overrides,
        )
        for task in tasks
    }
    todo = [t for t in tasks if not store.is_done(t["task_id"], keys[id(t)])]
    print(
        f"[run] {len(tasks)} tasks in shard {args.shard}/{args.num_shards}; "
        f"{len(tasks) - len(todo)} already done, {len(todo)} to go",
        flush=True,
    )
    if not todo:
        path = store.finalize(metadata=_metadata(args), summary=summarize(store.records))
        print(f"[run] nothing to do; wrote {path}")
        return 0

    model, tokenizer = _load_model(args)

    # samples are built once and reused across methods/budgets, so every
    # condition sees byte-identical prompts
    if args.task == "niah":
        samples = {
            s.task_id: s
            for s in niah.build_samples(
                tokenizer,
                context_lengths=sorted({t["context_len"] for t in todo}),
                depths=sorted({t["depth"] for t in todo}),
                n_samples=args.n_samples,
                corpus=args.niah_corpus,
                seed=args.seed,
            )
        }
        score_fn = niah.score
    else:
        samples = {
            s.task_id: s
            for s in longbench.build_samples(
                tasks=sorted({t["lb_task"] for t in todo}), n_samples=args.n_samples
            )
        }
        score_fn = longbench.score

    started = time.time()
    for i, task in enumerate(todo, 1):
        sample = samples[task["task_id"]]
        max_new = getattr(sample, "max_new_tokens", args.max_new_tokens)
        prompt = build_prompt(tokenizer, sample.context, sample.question)

        cache = make_cache(task["method"], model=model, budget=task["budget"], **overrides)
        try:
            result = generate_and_measure(
                model, tokenizer, prompt, cache, max_new_tokens=max_new
            )
        except torch.cuda.OutOfMemoryError as exc:
            free_cuda_memory()
            record = {
                **task,
                "run_key": keys[id(task)],
                "error": "cuda_oom",
                "error_detail": str(exc)[:500],
            }
            store.append(record)
            print(f"[run] CUDA OOM on {task['task_id']} ({task['method']})", flush=True)
            if not args.continue_on_oom:
                store.finalize(metadata=_metadata(args), summary=summarize(store.records))
                raise
            continue

        record = {
            **task,
            # the policy hyperparameters are recorded per row, not only in the
            # run metadata, so a sweep can be grouped by them at analysis time
            **overrides,
            "run_key": keys[id(task)],
            "model": "tiny" if args.tiny else args.model,
            "precision": getattr(model, "srkv_precision", "tiny"),
            "accuracy": score_fn(sample, result["generated_text"]),
            **{k: v for k, v in result.items() if k != "generated_text"},
            "generated_text": result["generated_text"][:400],
        }
        store.append(record)
        del cache
        free_cuda_memory()

        if i % 5 == 0 or i == len(todo):
            rate = i / max(time.time() - started, 1e-9)
            print(f"[run] {i}/{len(todo)} done ({rate:.2f} task/s)", flush=True)

    path = store.finalize(metadata=_metadata(args), summary=summarize(store.records))
    print(f"[run] wrote {path}")
    print(json.dumps(summarize(store.records), indent=2))
    return 0


def _metadata(args) -> dict:
    return {
        "argv": sys.argv,
        "args": vars(args),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def summarize(records: list[dict]) -> dict:
    """Mean accuracy / memory / throughput per (method, budget, context_len)."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if "error" in r:
            continue
        key = (r.get("method"), r.get("budget"), r.get("context_len") or r.get("lb_task"))
        groups.setdefault(key, []).append(r)

    out = {}
    for (method, budget, ctx), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(rows)
        out[f"{method}|budget={budget}|{ctx}"] = {
            "n": n,
            "accuracy": sum(r["accuracy"] for r in rows) / n,
            "max_memory_allocated": max(r["max_memory_allocated"] for r in rows),
            "tokens_per_sec": sum(r["tokens_per_sec"] for r in rows) / n,
            "n_tokens_cached": max(r["cache_stats"]["n_tokens_cached"] for r in rows),
            "budget_used_pct_max": max(r["budget_used_pct_max"] for r in rows),
        }
    n_errors = sum(1 for r in records if "error" in r)
    if n_errors:
        out["_errors"] = n_errors
    return out


if __name__ == "__main__":
    raise SystemExit(main())
