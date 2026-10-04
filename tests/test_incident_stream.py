import copy

import pytest
import torch

from eval.incident_stream import METHODS, IncidentStream, build_streams, ingest, measure_stream, probe
from scripts.analyse_incident_experiment import analyse
from src.caches import make_cache
from src.compat import get_kv
from src.models import build_tiny_model, build_tiny_tokenizer


def test_labels_follow_lifecycle_and_dataset_has_no_sample_count_drift():
    streams = build_streams()
    assert [s.to_dict() for s in streams[:4]] == [s.to_dict() for s in build_streams(4)]
    assert len({s.content_hash for s in streams}) == 24
    for stream in streams:
        active = set()
        for events, expected in zip(stream.event_trace, stream.expected):
            for event in events:
                if event["event"] == "error":
                    active.add(event["service"])
                else:
                    active.remove(event["service"])
            assert active == (set() if expected == "NONE" else {expected})


@pytest.mark.parametrize("method", METHODS)
def test_incremental_ingest_obeys_fixed_cap_and_probe_does_not_change_live_cache(method):
    model = build_tiny_model()
    tok = build_tiny_tokenizer()
    cache = make_cache(method, model=model, max_capacity=None if method == "full" else 32,
                       obs_window=8, pool_kernel=3)
    ids = tok("database failed queue healthy " * 20)["input_ids"]
    ingest(model, cache, ids[:40], chunk_size=16)
    ingest(model, cache, ids[40:], chunk_size=16)
    before = copy.deepcopy(cache)
    probe(model, tok, cache, "Which service? A) database B) queue Answer:")
    assert cache.n_tokens_seen == before.n_tokens_seen == len(ids)
    assert cache.get_stats() == before.get_stats()
    assert torch.equal(get_kv(cache, 0)[0], get_kv(before, 0)[0])
    assert cache.get_seq_length() == (len(ids) if method == "full" else 32)
    assert cache.check_conservation()
    # Continue after the probe with correct positions; query tokens are absent.
    ingest(model, cache, ids[:10], chunk_size=5)
    assert cache.n_tokens_seen == len(ids) + 10
    assert cache.t_now == len(ids) + 9


def test_stream_measure_reports_three_checkpoints_and_independent_labels():
    tok = build_tiny_tokenizer()
    model = build_tiny_model()
    stream = IncidentStream(0, 1, "recovery", ["ERROR database " * 20,
                            "RECOVERY database " * 20, "INFO queue " * 20],
                            ["database", "NONE", "NONE"], [[], [], []])
    cache = make_cache("snapkv_unified", model=model, max_capacity=32, obs_window=8)
    rows = list(measure_stream(model, tok, stream, cache, chunk_size=16))
    assert [r["checkpoint"] for r in rows] == [0, 1, 2]
    assert [r["expected"] for r in rows] == stream.expected
    assert all(r["cache_stats"]["n_tokens_cached"] == 32 for r in rows)
    assert rows[2]["stream_tokens"] == sum(r["new_tokens"] for r in rows)


def fake_records():
    rows = []
    for m in METHODS:
        for s in range(4):
            for c in range(3):
                gold = "NONE" if c == 2 else "database"
                rows.append(dict(method=m, stream_id=s, checkpoint=c, content_hash=str(s),
                                 expected=gold, predicted=gold, accuracy=1.0, conservation_ok=True,
                                 capacity=None if m == "full" else 512,
                                 cache_stats={"n_tokens_cached": 512}, scenario="recovery",
                                 ingest_seconds=1, probe_seconds=1, peak_gpu_bytes=100,
                                 retained_kv_bytes=100))
    return rows


def test_analysis_rejects_missing_data_and_does_not_call_ties_a_win():
    rows = fake_records()
    result = analyse(rows, 4)
    assert not result["exploratory_advantage"]
    assert result["comparisons"]["snapkv_unified"]["ci_adjusted"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="grid"):
        analyse(rows[:-1], 4)
    rows[0]["content_hash"] = "changed"
    with pytest.raises(ValueError, match="identical"):
        analyse(rows, 4)
