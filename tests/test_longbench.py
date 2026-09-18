"""Tests for the LongBench loader's parsing logic (no network).

`_longbench_data_dir()` is the only network-touching piece (downloads and
extracts THUDM/LongBench's data.zip); `build_samples()`'s parsing is tested
here against a fake local directory instead, keeping this CPU/offline like
the rest of the suite.
"""

from __future__ import annotations

import json

import eval.longbench as longbench


def _write_task_jsonl(data_dir, task: str, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"{task}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_samples_parses_the_real_jsonl_schema(tmp_path, monkeypatch):
    """Schema matches THUDM/LongBench's own LongBench.py generator exactly
    (verified by reading it): input/context/answers/length/dataset/language/
    all_classes/_id per line - this only exercises the fields build_samples
    actually reads.
    """
    _write_task_jsonl(tmp_path, "triviaqa", [
        {"input": "What is X?", "context": "long context here", "answers": ["X"],
         "length": 3, "dataset": "triviaqa", "language": "en", "all_classes": [], "_id": "1"},
        {"input": "What is Y?", "context": "another context", "answers": ["Y", "Y2"],
         "length": 2, "dataset": "triviaqa", "language": "en", "all_classes": [], "_id": "2"},
    ])
    monkeypatch.setattr(longbench, "_longbench_data_dir", lambda: tmp_path)

    samples = longbench.build_samples(tasks=["triviaqa"], n_samples=25)
    assert len(samples) == 2
    assert samples[0].question == "What is X?"
    assert samples[0].context == "long context here"
    assert samples[0].answers == ["X"]
    assert samples[1].answers == ["Y", "Y2"]
    assert all(s.task == "triviaqa" for s in samples)
    assert [s.sample_idx for s in samples] == [0, 1]


def test_build_samples_respects_n_samples_and_max_context_chars(tmp_path, monkeypatch):
    _write_task_jsonl(tmp_path, "gov_report", [
        {"input": "q", "context": "x" * 100, "answers": ["a"], "length": 1,
         "dataset": "gov_report", "language": "en", "all_classes": [], "_id": str(i)}
        for i in range(5)
    ])
    monkeypatch.setattr(longbench, "_longbench_data_dir", lambda: tmp_path)

    samples = longbench.build_samples(tasks=["gov_report"], n_samples=2, max_context_chars=10)
    assert len(samples) == 2
    assert all(len(s.context) == 10 for s in samples)


def test_build_samples_rejects_an_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setattr(longbench, "_longbench_data_dir", lambda: tmp_path)
    try:
        longbench.build_samples(tasks=["not_a_real_task"], n_samples=1)
        assert False, "expected KeyError"
    except KeyError:
        pass
