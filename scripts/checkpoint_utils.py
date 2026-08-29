"""Incremental result storage, so a killed Kaggle session loses one task.

Every finished task is appended to a JSONL sidecar and flushed to disk
immediately - not buffered until the end of the run. Re-running the same
command with the same `--output` reads that sidecar, skips what is already
there, and continues. The aggregated `.json` is rewritten from the sidecar
whenever the run ends (cleanly or not, if `finalize` is reached).

Resume is keyed on `(task_id, run_key)` where `run_key` identifies the
method/model/budget/hyperparameters. Two runs that differ in any of those are
different work and will not be confused for each other even if they share an
output path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def run_key(**fields: Any) -> str:
    """Stable short hash of the settings that define a unit of work."""
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


@dataclass
class ResultStore:
    """Append-only result log with resume support."""

    output_path: Path

    def __post_init__(self):
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_path.with_suffix(".jsonl")
        self._records: list[dict] = []
        self._seen: set[tuple[str, str]] = set()
        self._load()

    def _load(self) -> None:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # a session killed mid-write leaves one truncated line;
                    # everything before it is still good
                    continue
                self._records.append(record)
                self._seen.add((record.get("task_id", ""), record.get("run_key", "")))

    # -- reads ----------------------------------------------------------
    @property
    def records(self) -> list[dict]:
        return list(self._records)

    def is_done(self, task_id: str, key: str) -> bool:
        return (task_id, key) in self._seen

    def n_done(self) -> int:
        return len(self._records)

    def pending(self, tasks: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return [(tid, key) for tid, key in tasks if not self.is_done(tid, key)]

    # -- writes ---------------------------------------------------------
    def append(self, record: dict) -> None:
        """Write one result and force it to disk before returning."""
        key = (record.get("task_id", ""), record.get("run_key", ""))
        if key in self._seen:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._records.append(record)
        self._seen.add(key)

    def finalize(self, metadata: dict | None = None, summary: dict | None = None) -> Path:
        """Write the aggregated `.json` that plotting and analysis read."""
        payload = {
            "metadata": metadata or {},
            "summary": summary or {},
            "n_records": len(self._records),
            "records": self._records,
        }
        tmp = self.output_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.output_path)
        return self.output_path


def shard_tasks(tasks: list, shard: int, num_shards: int) -> list:
    """Deterministic, exhaustive, non-overlapping partition by position.

    Striding rather than slicing keeps every shard's mix of context lengths
    roughly even, so no single worker gets all the 16k cells.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    return tasks[shard::num_shards]


def free_cuda_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
