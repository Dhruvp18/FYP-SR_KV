"""The Phase 8 analysis decides the project's headline claim, so it gets tests.

Written against planted data with a known answer, and - the point of the file -
written while `results/` still contains no phase8_* records at all, so none of
these thresholds can have been tuned to a number someone had already seen.

The properties worth pinning are not "does it compute a mean": they are the
three ways a pre-registered test quietly stops being one. A secondary arm
leaking into the verdict, a single lucky cell counting as support, and an
underpowered cell counting as support.
"""

from __future__ import annotations

import json

import pytest

from scripts import analyse_phase8


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Point the analysis at a temp results/ so real data cannot be involved."""
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(analyse_phase8, "REPO_ROOT", tmp_path)
    return tmp_path / "results"


def _verdict(out, name):
    """The verdict line for `name`, whitespace-normalised.

    The summary pads the hypothesis name to a fixed width, so asserting on the
    literal spacing tests the format string rather than the conclusion.
    """
    for line in out.splitlines():
        squashed = " ".join(line.split())
        if squashed.startswith(name):
            return squashed[len(name):].strip()
    return None


def _write(results_dir, name, rows):
    path = results_dir / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _ppl_rows(*, effect, n=50, contexts=(2048, 4096), budgets=(0.2, 0.3),
              secondary_effect=0.0):
    """Perplexity rows where centroid_merge beats snapkv_unified by `effect` nll.

    The noise is deterministic and paired: both arms of a pair share the same
    per-sample offset, which is exactly the structure a paired bootstrap is
    meant to exploit. Without it a 0.02 effect would vanish into between-
    document variance and the test would prove nothing.
    """
    rows = []
    for context in contexts:
        for budget in budgets:
            for i in range(n):
                shared = (i % 17) * 0.05  # per-document difficulty, same for both arms
                for method, delta in (
                    ("snapkv_unified", 0.0),
                    ("centroid_merge", -effect),
                    ("sr_kv", -secondary_effect),
                    ("full", -0.5),
                ):
                    rows.append({
                        "task": "perplexity", "method": method, "model": "m",
                        "context_len": context,
                        "budget": 1.0 if method == "full" else budget,
                        "sample_idx": i,
                        "nll": 3.0 + shared + delta + (i % 3) * 0.001,
                        "accuracy": -(3.0 + shared + delta),
                    })
    return rows


def test_a_real_effect_at_several_settings_is_supported(results_dir, capsys):
    _write(results_dir, "phase8_perplexity_m.jsonl", _ppl_rows(effect=0.10))
    analyse_phase8.main([])
    out = capsys.readouterr().out
    assert _verdict(out, "H1 (perplexity)") == "SUPPORTED"
    assert out.count("WIN") >= 4


def test_no_effect_is_not_supported(results_dir, capsys):
    _write(results_dir, "phase8_perplexity_m.jsonl", _ppl_rows(effect=0.0))
    analyse_phase8.main([])
    out = capsys.readouterr().out
    assert "NOT SUPPORTED" in out
    assert _verdict(out, "H1 (perplexity)") == "NOT SUPPORTED"


def test_one_lucky_setting_is_not_enough(results_dir, capsys):
    """MIN_SETTINGS exists because a single significant cell among many is noise."""
    rows = _ppl_rows(effect=0.0)
    for row in rows:
        if row["method"] == "centroid_merge" and row["context_len"] == 2048 \
                and row["budget"] == 0.2:
            row["nll"] -= 0.10
    _write(results_dir, "phase8_perplexity_m.jsonl", rows)
    analyse_phase8.main([])
    out = capsys.readouterr().out
    assert "WIN" in out, "the planted cell should still be flagged as significant"
    assert _verdict(out, "H1 (perplexity)") == "NOT SUPPORTED"


def test_an_underpowered_win_does_not_count(results_dir, capsys):
    """MIN_PAIRS applies per comparison, not to the total row count."""
    _write(results_dir, "phase8_perplexity_m.jsonl",
           _ppl_rows(effect=0.10, n=analyse_phase8.MIN_PAIRS - 1))
    analyse_phase8.main([])
    out = capsys.readouterr().out
    assert "WIN" in out
    assert "significant win at n>=50: 0" in out
    assert _verdict(out, "H1 (perplexity)") == "NOT SUPPORTED"


def test_a_secondary_win_cannot_become_the_verdict(results_dir, capsys):
    """sr_kv is reported alongside; promoting it would be switching hypotheses."""
    _write(results_dir, "phase8_perplexity_m.jsonl",
           _ppl_rows(effect=0.0, secondary_effect=0.40))
    analyse_phase8.main([])
    out = capsys.readouterr().out
    assert "sr_kv vs snapkv_unified" in out
    assert _verdict(out, "H1 (perplexity)") == "NOT SUPPORTED"


def test_the_uncompressed_reference_is_reported(results_dir, capsys):
    """A null between two compressors means nothing if neither cost anything.

    The `full` arm is recorded at budget=1.0, so it shares no setting with a
    compressed arm whose key is "ctx=N b=0.2". Pairing on the full setting key
    finds nothing and prints "no baseline" while the baseline sits in the same
    file - which is why the reference is matched on context alone.
    """
    _write(results_dir, "phase8_perplexity_m.jsonl", _ppl_rows(effect=0.0))
    analyse_phase8.main([])
    out = capsys.readouterr().out
    assert "cost of compression: snapkv_unified vs uncompressed full" in out
    assert "no full baseline" not in out
    # one row per compressed setting, each against its own context's reference
    body = out.split("cost of compression")[1]
    for setting in ("ctx=2048 b=0.2", "ctx=2048 b=0.3",
                    "ctx=4096 b=0.2", "ctx=4096 b=0.3"):
        assert setting in body
    # planted: full is 0.5 nll below snapkv everywhere, and "lower is better"
    # means compressing *costs* 0.5, so the diff must be negative, not positive
    assert "-0.5" in body and "WORSE" in body


def test_errored_rows_are_dropped_not_scored(results_dir, capsys):
    rows = _ppl_rows(effect=0.10)
    rows.append({"task": "perplexity", "method": "centroid_merge", "model": "m",
                 "context_len": 2048, "budget": 0.2, "sample_idx": 999,
                 "error": "cuda_oom"})
    _write(results_dir, "phase8_perplexity_m.jsonl", rows)
    analyse_phase8.main([])  # a KeyError on the missing 'nll' would fail here
    assert _verdict(capsys.readouterr().out, "H1 (perplexity)") == "SUPPORTED"


def test_h2_pairs_within_a_longbench_task_not_across_them(results_dir, capsys):
    """sample_idx restarts per lb_task, so the setting key must include it.

    Pairing on sample_idx alone would silently match narrativeqa sample 3
    against gov_report sample 3 - different documents, different score scales,
    and a difference that means nothing.
    """
    rows = []
    for lb_task, base in (("narrativeqa", 0.20), ("gov_report", 0.31)):
        for budget in (0.05, 0.1):
            for i in range(50):
                for method, delta in (("snapkv_unified", 0.0),
                                      ("centroid_merge", 0.03)):
                    rows.append({"task": "longbench", "method": method, "model": "m",
                                 "lb_task": lb_task, "budget": budget,
                                 "sample_idx": i,
                                 "accuracy": base + delta + (i % 11) * 0.01})
    _write(results_dir, "phase8_tight_longbench_m.jsonl", rows)
    analyse_phase8.main([])
    out = capsys.readouterr().out
    for setting in ("narrativeqa b=0.05", "gov_report b=0.1"):
        assert setting in out
    assert _verdict(out, "H2 (tight budgets)") == "SUPPORTED"
