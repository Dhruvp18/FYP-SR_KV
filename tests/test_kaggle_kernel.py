"""Tests for the Kaggle automation driver and the phase gates.

The driver's pure parts (metadata, notebook, argv, output merging) are tested
here; nothing in this file touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_results import (
    gate_phase1,
    gate_phase2,
    gate_phase3,
    gate_phase4,
    gate_phase6,
    gate_phase7,
    run_gate,
)
from scripts.kaggle_kernel import (
    PHASES,
    build_commands,
    build_metadata,
    build_notebook,
    build_parser,
    kaggle_argv,
    main,
    merge_outputs,
    parse_status,
    slug,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# kernel metadata / notebook generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phase", sorted(PHASES))
def test_metadata_is_valid_for_every_phase(phase):
    meta = build_metadata(phase, "dhruvp18")
    assert meta["id"] == f"dhruvp18/sr-kv-phase{phase}"
    assert meta["code_file"] == f"sr-kv-phase{phase}.ipynb"
    # Kaggle derives the kernel's *live* slug from `title`, not from `id`. A
    # title with spaces/punctuation slugifies to something else, and every
    # later status/pull call (built from slug(phase)) would then 404 against
    # a kernel that actually lives at a different URL. The title must be the
    # slug itself so the two can never diverge.
    assert meta["title"] == slug(phase)
    assert meta["kernel_type"] == "notebook"
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True
    # phase 7 only draws figures from existing results
    assert meta["enable_gpu"] is (phase != 7)


def test_metadata_chains_previous_phases_for_resume():
    meta = build_metadata(5, "dhruvp18")
    assert meta["kernel_sources"] == [f"dhruvp18/sr-kv-phase{p}" for p in (1, 2, 3, 4)]
    assert build_metadata(1, "dhruvp18")["kernel_sources"] == []

    explicit = build_metadata(5, "dhruvp18", depends_on=["dhruvp18/sr-kv-phase4"])
    assert explicit["kernel_sources"] == ["dhruvp18/sr-kv-phase4"]


def test_unknown_phase_is_rejected():
    with pytest.raises(KeyError):
        build_metadata(99, "someone")


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_commands_run_the_targets_then_the_gate(phase):
    commands = build_commands(phase, model="qwen2.5-1.5b", model3b="llama3.2-3b",
                              budget=0.3, samples=3, shard=0, num_shards=1)
    spec = PHASES[phase]
    for target in spec["targets"]:
        assert any(command.startswith(f"make {target} ") for command in commands)

    gate_index = next(i for i, c in enumerate(commands) if c.startswith(f"make {spec['gate']} "))
    target_indices = [
        i for i, c in enumerate(commands)
        if any(c.startswith(f"make {t} ") for t in spec["targets"])
    ]
    assert gate_index > max(target_indices), "the gate must run after the work, not before"

    for command in commands:
        assert "MODEL=qwen2.5-1.5b" in command and "BUDGET=0.3" in command


def test_phase4_freezes_the_rope_mode_only_after_its_gate_passes():
    commands = build_commands(4, model="qwen2.5-1.5b", model3b="llama3.2-3b",
                              budget=0.3, samples=5, shard=0, num_shards=1)
    gate_index = next(i for i, c in enumerate(commands) if "gate4" in c)
    freeze_index = next(i for i, c in enumerate(commands) if "freeze-rope" in c)
    assert freeze_index > gate_index, (
        "freezing before the gate would let a chance-level result become the frozen default"
    )


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_notebook_is_valid_json_and_thin(phase):
    nb = build_notebook(phase, repo="https://example.com/r.git", model="qwen2.5-1.5b",
                        model3b="llama3.2-3b", budget=0.3, samples=3, shard=0, num_shards=1)
    json.dumps(nb)  # must be serialisable
    assert nb["nbformat"] == 4
    assert nb["cells"], "empty notebook"

    source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "git clone" in source and "pytest" in source
    assert f"make {PHASES[phase]['gate']}" in source
    # notebooks orchestrate; they must not carry project logic
    for forbidden in ("def _compress", "topk", "cluster_and_merge", "rotate_half"):
        assert forbidden not in source, f"notebook contains project logic: {forbidden}"


def test_notebook_sets_expandable_segments_before_any_run_command():
    """PyTorch's caching allocator fragments over hundreds of varying-shaped
    tensors (LongBench's four tasks have very different sequence/generation
    lengths) - confirmed for real: a run OOM'd requesting 1.15 GiB with only
    198 MiB contiguous free, 81 successful tasks in, well after
    torch.cuda.empty_cache() had already run between every one of them.
    expandable_segments is PyTorch's own fix for this. It has to be an env
    var set before any `make`/`python eval/run.py` subprocess runs, not
    something eval/run.py sets itself, since it must apply before CUDA
    initializes in whichever process does the allocating.
    """
    nb = build_notebook(5, repo="r", model="qwen2.5-1.5b", model3b="qwen2.5-3b",
                        budget=0.3, samples=3, shard=0, num_shards=1)
    sources = ["".join(cell["source"]) for cell in nb["cells"]]
    full_text = "\n".join(sources)
    assert "expandable_segments:True" in full_text
    env_idx = next(i for i, s in enumerate(sources) if "expandable_segments" in s)
    run_idx = next(i for i, s in enumerate(sources) if "make phase5" in s)
    assert env_idx < run_idx, "allocator config must be set before any make/eval.run.py call"


def test_notebook_treats_gate_exit_2_as_not_a_crash():
    """Exit 2 ("needs a human look") must not fail the kernel.

    Phase 4's real run hit exactly this: gate4 correctly found the RoPE
    modes statistically indistinguishable (p=0.867, later p=1.000 once a
    separate dedup bug was fixed) and exited 2. The old notebook piped every
    command through one check=True sh() call, so that legitimate, informative
    exit code showed up on Kaggle as KernelWorkerStatus.ERROR - indistinguishable
    from an actual crash - even though 300 real records had already been
    produced and gate4 had done exactly its job. Exit 1 must still be fatal;
    only 2 gets the pass.
    """
    nb = build_notebook(4, repo="r", model="qwen2.5-1.5b", model3b="qwen2.5-3b",
                        budget=0.2, samples=3, shard=0, num_shards=1)
    source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "gate_rc = sh(" in source and "check=False" in source
    assert "if gate_rc == 1:" in source and "raise SystemExit" in source
    assert "elif gate_rc == 2:" in source
    # freeze-rope (the after-gate step) is only reachable when gate_rc is
    # neither 1 nor 2 - i.e. the else branch
    freeze_line = next(l for l in source.splitlines() if "make freeze-rope" in l)
    assert freeze_line.strip().startswith("sh(") and freeze_line.startswith("    "), (
        "freeze-rope must be indented under the gate's else branch, not run unconditionally"
    )


def test_notebook_restores_previous_results_before_running():
    nb = build_notebook(5, repo="r", model="m", model3b="m3", budget=0.3, samples=3,
                        shard=0, num_shards=1)
    sources = ["".join(cell["source"]) for cell in nb["cells"]]
    restore = next(i for i, s in enumerate(sources) if "/kaggle/input/" in s)
    work = next(i for i, s in enumerate(sources) if "make phase5" in s)
    assert restore < work, "results must be restored before the phase runs, or resume is pointless"


def test_targets_and_skip_gate_support_diagnostic_runs():
    """phase4-scan has no pass condition, so it must be able to skip the gate."""
    commands = build_commands(4, model="m", model3b="m3", budget=0.3, samples=3,
                              shard=0, num_shards=1, targets=["phase4-scan"], skip_gate=True)
    assert any(c.startswith("make phase4-scan ") for c in commands)
    assert not any("gate4" in c for c in commands)
    assert not any("freeze-rope" in c for c in commands), (
        "freeze-rope runs after the gate; skipping the gate must skip it too"
    )
    # the default path is unchanged
    default = build_commands(4, model="m", model3b="m3", budget=0.3, samples=3,
                             shard=0, num_shards=1)
    assert any("gate4" in c for c in default)
    assert any("freeze-rope" in c for c in default)


def test_extra_make_vars_reach_the_commands():
    commands = build_commands(4, model="m", model3b="m3", budget=0.2, samples=3,
                              shard=0, num_shards=1, extra_vars="SAMPLES4=20")
    assert all("SAMPLES4=20" in c for c in commands)


def test_sharding_reaches_the_generated_commands():
    commands = build_commands(5, model="m", model3b="m3", budget=0.3, samples=3,
                              shard=2, num_shards=4)
    assert all("SHARD=2" in c and "NSHARDS=4" in c for c in commands)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def test_kaggle_argv_shapes():
    directory = Path("kaggle/phase3")
    assert kaggle_argv("push", phase=3, directory=directory) == [
        "kaggle", "kernels", "push", "-p", str(directory),
    ]
    assert kaggle_argv("status", phase=3, user="u") == [
        "kaggle", "kernels", "status", "u/sr-kv-phase3",
    ]
    out_dir = Path("out")
    # --force matters: without it the CLI keeps any local file that looks
    # newer, so re-running a phase silently re-serves the previous run's
    # results and the gate scores data the run never produced.
    assert kaggle_argv("pull", phase=3, user="u", out_dir=out_dir) == [
        "kaggle", "kernels", "output", "u/sr-kv-phase3", "-p", str(out_dir), "--force",
    ]
    with pytest.raises(ValueError):
        kaggle_argv("delete", phase=1)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('Kernel status: "complete"', "complete"),
        ("has status running", "running"),
        ("status queued for execution", "queued"),
        ('status "error" - see log', "error"),
        ("something unexpected", "unknown"),
    ],
)
def test_parse_status(text, expected):
    assert parse_status(text) == expected


def test_generate_writes_a_pushable_directory(tmp_path):
    rc = main(["generate", "--phase", "4", "--user", "dhruvp18", "--out", str(tmp_path)])
    assert rc == 0

    directory = tmp_path / "phase4"
    meta = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "dhruvp18/sr-kv-phase4"
    # the file the metadata names must actually exist, or the push fails server-side
    assert (directory / meta["code_file"]).exists()
    json.loads((directory / meta["code_file"]).read_text(encoding="utf-8"))


def test_commands_requiring_a_username_say_so():
    with pytest.raises(SystemExit, match="--user"):
        main(["generate", "--phase", "1"])


def test_push_refuses_without_generated_metadata(tmp_path):
    with pytest.raises(SystemExit, match="generate"):
        main(["push", "--phase", "1", "--out", str(tmp_path), "--dry-run"])


def test_parser_defaults_point_at_the_project_repo():
    args = build_parser().parse_args(["generate", "--phase", "1", "--user", "u"])
    assert args.repo.endswith("FYP-SR_KV.git")
    assert args.model == "qwen2.5-1.5b" and args.budget == 0.3


def test_merge_outputs_sorts_results_and_figures(tmp_path):
    pulled = tmp_path / "pulled"
    (pulled / "results").mkdir(parents=True)
    (pulled / "figures").mkdir(parents=True)
    (pulled / "results" / "a.jsonl").write_text("{}", encoding="utf-8")
    (pulled / "figures" / "p.png").write_bytes(b"x")
    (pulled / "unrelated.txt").write_text("ignore me", encoding="utf-8")

    results, figures = tmp_path / "results", tmp_path / "figures"
    assert merge_outputs(pulled, results, figures) == 2
    assert (results / "a.jsonl").exists()
    assert (figures / "p.png").exists()
    assert not (results / "unrelated.txt").exists()


def test_merge_outputs_preserves_subdirectories(tmp_path):
    """Flattening to just the filename silently undid the fix that moved
    phase4-scan's output to results/diagnostics/: the very next pull would
    flatten .../results/diagnostics/x.json back to results_dir/x.json,
    re-introducing the exact task_id/run_key collision that move exists to
    prevent (load_records globs results/ non-recursively).
    """
    pulled = tmp_path / "pulled"
    (pulled / "results" / "diagnostics").mkdir(parents=True)
    (pulled / "results" / "diagnostics" / "scan.json").write_text("{}", encoding="utf-8")
    (pulled / "results" / "a.jsonl").write_text("{}", encoding="utf-8")

    results, figures = tmp_path / "results", tmp_path / "figures"
    assert merge_outputs(pulled, results, figures) == 2
    assert (results / "diagnostics" / "scan.json").exists()
    assert (results / "a.jsonl").exists()
    assert not (results / "scan.json").exists(), "must not flatten into the top level"


# ---------------------------------------------------------------------------
# phase gates
# ---------------------------------------------------------------------------
def _niah(method, accuracy, *, model="m", context_len=4096, depth=50, **extra):
    return {"model": model, "method": method, "accuracy": accuracy, "budget": 0.3,
            "context_len": context_len, "depth": depth, "task_id": f"t{depth}", **extra}


def test_gate1_passes_only_on_a_convincing_sanity_score():
    good = [_niah("full", 1.0, context_len=512) for _ in range(10)]
    assert gate_phase1(good, model="m")[0] == 0

    bad = [_niah("full", 0.5, context_len=512) for _ in range(10)]
    code, lines = gate_phase1(bad, model="m")
    assert code == 1 and any("harness" in line for line in lines)

    thin = [_niah("full", 1.0, context_len=512) for _ in range(2)]
    assert gate_phase1(thin, model="m")[0] == 1


def test_gate2_requires_streamingllm_to_lose_mid_sequence():
    records = []
    for depth in (25, 50, 75):
        records += [_niah("full", 1.0, depth=depth), _niah("snapkv", 0.8, depth=depth),
                    _niah("streaming_llm", 0.1, depth=depth)]
    assert gate_phase2(records, model="m")[0] == 0

    inverted = [r for r in records if r["method"] != "streaming_llm"]
    inverted += [_niah("streaming_llm", 0.95, depth=d) for d in (25, 50, 75)]
    code, lines = gate_phase2(inverted, model="m")
    assert code == 1 and any("depth placement" in line for line in lines)


def test_gate3_catches_broken_conservation_and_budget_overruns():
    ok = [_niah("sr_kv", 0.7, context_len=8192, conservation_ok=True, budget_used_pct_max=100.0)]
    assert gate_phase3(ok, model="m")[0] == 0

    leaky = [_niah("sr_kv", 0.7, context_len=8192, conservation_ok=False,
                   budget_used_pct_max=100.0)]
    assert gate_phase3(leaky, model="m")[0] == 1

    fat = [_niah("sr_kv", 0.7, context_len=8192, conservation_ok=True,
                 budget_used_pct_max=140.0)]
    code, lines = gate_phase3(fat, model="m")
    assert code == 1 and any("budget" in line for line in lines)


def test_gate4_refuses_to_anoint_a_winner_at_chance():
    at_chance = [_niah("sr_kv", 0.0, rope_position_mode=mode)
                 for mode in ("latest", "earliest", "attn_weighted")]
    code, lines = gate_phase4(at_chance, model="m")
    assert code == 1 and any("upstream" in line for line in lines)

    separated = [
        _niah("sr_kv", 0.2, rope_position_mode="latest"),
        _niah("sr_kv", 0.8, rope_position_mode="earliest"),
        _niah("sr_kv", 0.5, rope_position_mode="attn_weighted"),
    ]
    code, lines = gate_phase4(separated, model="m")
    assert code == 0 and any("winner: earliest" in line for line in lines)


def test_gate4_refuses_a_winner_that_is_one_flipped_sample():
    """A 0.04 "spread" at n=25 is one record, not a result.

    This is the real Phase 4 outcome at budget=0.1 (7/25, 6/25, 7/25). The
    mean alone reads as a clean win for `latest`; Fisher exact gives p=1.0.
    Freezing it would write a coin flip into configs/defaults.yaml, which
    every later phase then inherits.
    """
    records = []
    for mode, n_correct in (("latest", 7), ("earliest", 6), ("attn_weighted", 7)):
        for i in range(25):
            records.append(_niah("sr_kv", 1.0 if i < n_correct else 0.0,
                                 depth=i % 5, rope_position_mode=mode))
    code, lines = gate_phase4(records, model="m")
    assert code == 2, "a one-record difference must not pass as a measured winner"
    assert any("NO SIGNIFICANT DIFFERENCE" in line for line in lines)
    assert any("p=1.000" in line for line in lines)


def test_gate4_accepts_a_difference_that_is_actually_significant():
    records = []
    for mode, n_correct in (("latest", 24), ("earliest", 4), ("attn_weighted", 5)):
        for i in range(25):
            records.append(_niah("sr_kv", 1.0 if i < n_correct else 0.0,
                                 depth=i % 5, rope_position_mode=mode))
    code, lines = gate_phase4(records, model="m")
    assert code == 0
    assert any("winner: latest" in line for line in lines)


def test_gate4_refuses_to_average_across_budgets():
    """Two budgets in results/ are two experiments, not one bigger sample.

    Mixing a saturated budget=0.3 sweep into a budget=0.1 sweep changed which
    mode `max()` picked, purely from which files were on disk.
    """
    records = []
    for mode in ("latest", "earliest", "attn_weighted"):
        for i in range(5):
            records.append(_niah("sr_kv", 1.0, depth=i, rope_position_mode=mode, budget=0.3))
            records.append(_niah("sr_kv", 0.0, depth=i, rope_position_mode=mode, budget=0.1))
    code, lines = gate_phase4(records, model="m")
    assert code == 1
    assert any("refusing to aggregate across budgets" in line for line in lines)
    # naming one budget resolves it
    code_one, _ = gate_phase4(records, model="m", budgets=[0.3])
    assert code_one != 1


def test_gate4_notes_when_the_modes_are_indistinguishable():
    close = [
        _niah("sr_kv", 0.70, rope_position_mode="latest"),
        _niah("sr_kv", 0.71, rope_position_mode="earliest"),
        _niah("sr_kv", 0.705, rope_position_mode="attn_weighted"),
    ]
    code, lines = gate_phase4(close, model="m")
    assert code == 0
    assert any("no strong effect" in line for line in lines)


def test_gate5_flags_srkv_losing_to_both_ablations():
    records = []
    for context in (2048, 4096, 8192, 16384):
        for depth in (0, 25, 50, 75, 100):
            records += [
                _niah("streaming_llm", 0.3, context_len=context, depth=depth),
                _niah("snapkv_unified", 0.6, context_len=context, depth=depth),
                _niah("centroid_merge", 0.5, context_len=context, depth=depth),
                _niah("sr_kv", 0.2, context_len=context, depth=depth),
            ]
    code, lines = run_gate(5, records, model="m", budgets=[0.3], n_samples=1,
                           figures_dir=Path("."))
    assert code == 2, "a negative result must surface, not pass silently"
    assert any("FLAG" in line for line in lines)


def test_gate6_fails_on_oom_rather_than_quietly_dropping_cells():
    ok = [{"model": "llama3.2-3b", "method": "sr_kv", "budget": 0.3, "accuracy": 0.5,
           "max_memory_allocated": 1, "tokens_per_sec": 1.0, "cache_stats": {}, "task_id": "t"}]
    assert gate_phase6(ok, model="llama3.2-3b", budgets=[0.3])[0] == 0

    with_oom = ok + [{"model": "llama3.2-3b", "error": "cuda_oom", "task_id": "t2"}]
    code, lines = gate_phase6(with_oom, model="llama3.2-3b", budgets=[0.3])
    assert code == 1 and any("4bit" in line for line in lines)


def test_gate7_rejects_empty_canvases(tmp_path):
    assert gate_phase7(tmp_path)[0] == 1

    for name in ("pareto_m.png", "niah_heatmap_m_b0.3.png", "ablation_m.png",
                 "rope_position_modes.png"):
        (tmp_path / name).write_bytes(b"x" * 9000)
    assert gate_phase7(tmp_path)[0] == 0

    (tmp_path / "pareto_m.png").write_bytes(b"x" * 100)
    code, lines = gate_phase7(tmp_path)
    assert code == 1 and any("empty plot" in line for line in lines)
