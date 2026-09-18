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


class _WordTokenizer:
    """Minimal whitespace tokenizer, just enough to test truncation by count."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_truncate_by_tokens_keeps_both_ends_not_just_the_start():
    """LongBench's own convention: relevant info can be anywhere, so keep
    both ends of the budget rather than truncating from one side only.
    """
    text = " ".join(f"w{i}" for i in range(100))
    truncated = longbench._truncate_by_tokens(_WordTokenizer(), text, max_tokens=10)
    words = truncated.split()
    assert len(words) == 10
    assert words[0] == "w0", "must keep the start"
    assert words[-1] == "w99", "must keep the end, not just the start"


def test_truncate_by_tokens_is_a_noop_under_the_budget():
    text = "short document"
    assert longbench._truncate_by_tokens(_WordTokenizer(), text, max_tokens=100) == text


def test_build_samples_truncates_oversized_context_by_tokens(tmp_path, monkeypatch):
    """This is the actual OOM fix: Phase 5's real run tried to allocate 43.89
    GiB on a 14.56 GiB T4 because an untruncated narrativeqa document was fed
    to `full` (uncompressed) - and prefill computes full self-attention over
    the whole document before any cache eviction runs, so every method would
    OOM the same way, not just the baseline.
    """
    long_context = " ".join(f"w{i}" for i in range(1000))
    _write_task_jsonl(tmp_path, "narrativeqa", [
        {"input": "q", "context": long_context, "answers": ["a"], "length": 1000,
         "dataset": "narrativeqa", "language": "en", "all_classes": [], "_id": "1"},
    ])
    monkeypatch.setattr(longbench, "_longbench_data_dir", lambda: tmp_path)

    samples = longbench.build_samples(
        tasks=["narrativeqa"], n_samples=1,
        tokenizer=_WordTokenizer(), max_context_tokens=50,
    )
    assert len(samples[0].context.split()) == 50


def test_build_samples_rejects_an_unknown_task(tmp_path, monkeypatch):
    monkeypatch.setattr(longbench, "_longbench_data_dir", lambda: tmp_path)
    try:
        longbench.build_samples(tasks=["not_a_real_task"], n_samples=1)
        assert False, "expected KeyError"
    except KeyError:
        pass
