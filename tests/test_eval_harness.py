"""Phase 1 / 5 / 6 exit tests for the eval harness.

Runs through `eval/run.py` itself - the same entrypoint every experiment uses -
with `--tiny`, so the harness is exercised end to end on CPU with no downloads.
Accuracy is meaningless with random weights; what is tested here is that the
harness produces the right record schema, resumes correctly after being killed,
and shards exhaustively.
"""

from __future__ import annotations

import json

import pytest

from eval.run import build_parser, build_task_list, main, summarize
from scripts.checkpoint_utils import ResultStore, run_key, shard_tasks


def _args(*extra):
    return build_parser().parse_args(list(extra))


def _base_cmd(output, **over):
    cmd = [
        "--tiny", "--allow_cpu",
        "--method", over.get("method", "full,snapkv,sr_kv"),
        "--task", "niah",
        "--context_len", "384",
        "--budget", "0.25",
        "--depths", over.get("depths", "0,50,100"),
        "--n_samples", over.get("n_samples", "1"),
        "--max_new_tokens", "4",
        "--obs_window", "8",
        "--n_centroids", "4",
        "--output", str(output),
    ]
    for key in ("shard", "num_shards", "limit"):
        if key in over:
            cmd += [f"--{key}", str(over[key])]
    return cmd


# ---------------------------------------------------------------------------
# result schema
# ---------------------------------------------------------------------------
def test_run_produces_the_expected_result_schema(tmp_path):
    output = tmp_path / "run.json"
    assert main(_base_cmd(output)) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_records"] == 9  # 3 methods x 3 depths x 1 sample
    for record in payload["records"]:
        for field in (
            "accuracy",
            "max_memory_allocated",
            "tokens_per_sec",
            "prompt_tokens",
            "generated_tokens",
            "cache_stats",
            "budget_used_pct_max",
            "conservation_ok",
        ):
            assert field in record, f"missing {field}"
        assert set(record["cache_stats"]) == {
            "n_tokens_cached",
            "n_tokens_evicted",
            "n_centroids",
            "budget_used_pct",
        }
        assert record["conservation_ok"] is True
    assert payload["summary"], "summary block is empty"
    assert payload["metadata"]["args"]["task"] == "niah"


def test_noop_cache_reports_zero_evictions_through_the_harness(tmp_path):
    output = tmp_path / "noop.json"
    assert main(_base_cmd(output, method="full")) == 0
    records = json.loads(output.read_text(encoding="utf-8"))["records"]
    assert records
    for record in records:
        assert record["cache_stats"]["n_tokens_evicted"] == 0
        assert record["cache_stats"]["n_centroids"] == 0


def test_compressed_runs_stay_within_budget_through_the_harness(tmp_path):
    output = tmp_path / "budget.json"
    assert main(_base_cmd(output, method="snapkv,sr_kv,streaming_llm")) == 0
    records = json.loads(output.read_text(encoding="utf-8"))["records"]
    for record in records:
        assert record["budget_used_pct_max"] <= 100.0 + 1e-6
        assert record["cache_stats"]["n_tokens_cached"] < record["prompt_tokens"]


# ---------------------------------------------------------------------------
# resume after an interrupted session
# ---------------------------------------------------------------------------
def test_interrupted_run_resumes_without_redoing_or_duplicating(tmp_path):
    """Kill after N of M tasks; the resumed run must finish exactly M-N more."""
    output = tmp_path / "resume.json"
    jsonl = output.with_suffix(".jsonl")

    total = len(build_task_list(_args(*_base_cmd(output))))
    assert total == 9

    # --- session 1: dies after 4 tasks
    assert main(_base_cmd(output, limit=4)) == 0
    first_lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(first_lines) == 4

    # --- session 2: same command, no limit
    assert main(_base_cmd(output)) == 0
    all_lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(all_lines) == total, "resume duplicated or skipped work"
    assert all_lines[:4] == first_lines, "resume redid work that was already done"

    task_ids = [(json.loads(line)["task_id"], json.loads(line)["method"]) for line in all_lines]
    assert len(set(task_ids)) == total, "duplicate (task, method) pairs in the log"

    # --- session 3: nothing left to do
    assert main(_base_cmd(output)) == 0
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == total


def test_result_store_ignores_a_repeated_append(tmp_path):
    store = ResultStore(tmp_path / "s.json")
    key = run_key(method="sr_kv", budget=0.3)
    store.append({"task_id": "t1", "run_key": key, "accuracy": 1.0})
    store.append({"task_id": "t1", "run_key": key, "accuracy": 0.0})
    assert store.n_done() == 1
    assert store.records[0]["accuracy"] == 1.0


def test_result_store_survives_a_truncated_final_line(tmp_path):
    """A session killed mid-write leaves half a line; the rest must still load."""
    path = tmp_path / "trunc.json"
    store = ResultStore(path)
    store.append({"task_id": "t1", "run_key": "k", "accuracy": 1.0})
    with store.jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write('{"task_id": "t2", "run_k')

    reloaded = ResultStore(path)
    assert reloaded.n_done() == 1
    assert reloaded.is_done("t1", "k")


def test_different_hyperparameters_are_not_confused_for_completed_work(tmp_path):
    """Same task id, different alpha => different work, must not resume-skip."""
    output = tmp_path / "hp.json"
    assert main(_base_cmd(output, method="sr_kv") + ["--alpha", "1.0"]) == 0
    n_first = len(output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines())
    assert main(_base_cmd(output, method="sr_kv") + ["--alpha", "2.0"]) == 0
    n_second = len(output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines())
    assert n_second == 2 * n_first


# ---------------------------------------------------------------------------
# sharding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_shards", [1, 2, 3, 4, 7])
def test_shards_are_exhaustive_and_non_overlapping(tmp_path, num_shards):
    args = _args(*_base_cmd(tmp_path / "x.json", depths="0,25,50,75,100", n_samples="3"))
    full = build_task_list(args)
    assert len(full) == 3 * 5 * 3  # methods x depths x samples

    recombined = []
    for shard in range(num_shards):
        recombined.extend(shard_tasks(full, shard, num_shards))

    assert len(recombined) == len(full), "shards lost or duplicated tasks"
    key = lambda t: (t["method"], t["task_id"])  # noqa: E731
    assert sorted(map(key, recombined)) == sorted(map(key, full))
    assert len(set(map(key, recombined))) == len(full)


def test_shard_bounds_are_validated():
    with pytest.raises(ValueError):
        shard_tasks([1, 2, 3], shard=3, num_shards=3)
    with pytest.raises(ValueError):
        shard_tasks([1, 2, 3], shard=0, num_shards=0)


def test_sharded_runs_write_to_the_same_store_without_collisions(tmp_path):
    output = tmp_path / "sharded.json"
    for shard in range(3):
        assert main(_base_cmd(output, shard=shard, num_shards=3)) == 0
    lines = output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 9
    assert len({(json.loads(l)["task_id"], json.loads(l)["method"]) for l in lines}) == 9


def test_summarize_groups_by_method_and_budget():
    records = [
        {"method": "sr_kv", "budget": 0.3, "context_len": 4096, "accuracy": 1.0,
         "max_memory_allocated": 10, "tokens_per_sec": 5.0, "budget_used_pct_max": 100.0,
         "cache_stats": {"n_tokens_cached": 100}},
        {"method": "sr_kv", "budget": 0.3, "context_len": 4096, "accuracy": 0.0,
         "max_memory_allocated": 20, "tokens_per_sec": 7.0, "budget_used_pct_max": 100.0,
         "cache_stats": {"n_tokens_cached": 100}},
        {"method": "snapkv", "budget": 0.3, "context_len": 4096, "error": "cuda_oom"},
    ]
    summary = summarize(records)
    assert summary["sr_kv|budget=0.3|4096"]["accuracy"] == 0.5
    assert summary["sr_kv|budget=0.3|4096"]["n"] == 2
    assert summary["_errors"] == 1
