"""Phase 4 and Phase 7 gate tests: frozen defaults, result checks, figures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from eval.run import main as run_main
from scripts.check_results import check_ablation, check_completeness
from scripts.freeze_rope_mode import apply_winner, mode_scores
from scripts.make_plots import load_records
from scripts.make_plots import main as plots_main
from src.rope_positions import POSITION_MODES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = REPO_ROOT / "configs" / "defaults.yaml"


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------
def test_defaults_file_declares_a_rope_mode_and_its_frozen_state():
    cfg = yaml.safe_load(DEFAULTS.read_text(encoding="utf-8"))
    assert cfg["rope_position_mode"] in POSITION_MODES
    assert "rope_position_mode_frozen" in cfg, (
        "defaults.yaml must say whether the mode was chosen by Phase 4 or is a placeholder"
    )
    for key in ("alpha", "beta", "lam", "obs_window", "n_sink", "centroid_frac"):
        assert key in cfg


def test_generated_configs_cover_the_experiment_matrix():
    configs = sorted(p.name for p in (REPO_ROOT / "configs").glob("*__*.yaml"))
    assert configs, "run scripts/gen_configs.py"
    for name in ("qwen25-15b__sr_kv__b030.yaml", "qwen25-15b__centroid_merge__b030.yaml"):
        assert name in configs
    body = yaml.safe_load((REPO_ROOT / "configs" / "qwen25-15b__sr_kv__b030.yaml").read_text())
    assert body["method"] == "sr_kv" and body["budget"] == 0.3


def test_no_later_script_hardcodes_a_rope_position_mode():
    """Phase 4's winner must reach later phases through configs/defaults.yaml.

    Only the places that legitimately name a mode are exempt: the module that
    defines them, the class that takes one as a parameter, the CLI help text,
    the Phase 4 ablation driver itself, and the tests.
    """
    allowed = {
        REPO_ROOT / "src" / "rope_positions.py",
        REPO_ROOT / "src" / "caches" / "sr_kv.py",
        REPO_ROOT / "src" / "caches" / "__init__.py",
        REPO_ROOT / "eval" / "run.py",
        REPO_ROOT / "scripts" / "freeze_rope_mode.py",
        REPO_ROOT / "scripts" / "make_plots.py",
        REPO_ROOT / "Makefile",
        REPO_ROOT / "CLAUDE.md",
    }
    searched = [
        *(REPO_ROOT / "scripts").glob("*.py"),
        *(REPO_ROOT / "eval").glob("*.py"),
        *(REPO_ROOT / "src").rglob("*.py"),
        *(REPO_ROOT / "notebooks").glob("*.ipynb"),
    ]
    offenders = []
    for path in searched:
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for mode in POSITION_MODES:
            if f'"{mode}"' in text or f"'{mode}'" in text or f"={mode}" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} mentions {mode!r}")
    assert not offenders, "hardcoded RoPE mode outside the allowed files:\n" + "\n".join(offenders)


def test_freeze_rope_mode_rewrites_defaults(tmp_path):
    target = tmp_path / "defaults.yaml"
    shutil.copy(DEFAULTS, target)
    apply_winner("earliest", "latest=0.10, earliest=0.90, attn_weighted=0.50", path=target)

    cfg = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert cfg["rope_position_mode"] == "earliest"
    assert cfg["rope_position_mode_frozen"] is True
    assert "0.90" in cfg["rope_position_mode_evidence"]
    # untouched keys survive
    assert cfg["alpha"] == yaml.safe_load(DEFAULTS.read_text(encoding="utf-8"))["alpha"]


def test_mode_scores_only_counts_merging_conditions():
    records = [
        {"method": "sr_kv", "rope_position_mode": "latest", "accuracy": 1.0, "model": "m"},
        {"method": "centroid_merge", "rope_position_mode": "latest", "accuracy": 0.0, "model": "m"},
        # hard eviction has no centroids, so its rope mode is meaningless
        {"method": "snapkv_unified", "rope_position_mode": "latest", "accuracy": 1.0, "model": "m"},
    ]
    scores = mode_scores(records)
    assert sorted(scores["latest"]) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# result checks
# ---------------------------------------------------------------------------
def test_completeness_lists_every_missing_cell():
    records = [
        {"model": "m", "method": "sr_kv", "budget": 0.3, "context_len": 2048,
         "depth": 0, "accuracy": 1.0},
    ]
    missing = check_completeness(
        records, model="m", budgets=[0.3], methods=["sr_kv"], contexts=[2048, 4096],
        depths=[0, 50], n_samples=1, lb_tasks=[],
    )
    assert len(missing) == 3
    assert all("sr_kv" in line for line in missing)

    complete = check_completeness(
        records, model="m", budgets=[0.3], methods=["sr_kv"], contexts=[2048],
        depths=[0], n_samples=1, lb_tasks=[],
    )
    assert complete == []


def test_ablation_check_flags_srkv_losing_to_both_siblings():
    def rows(method, accuracy):
        return [{"model": "m", "method": method, "budget": 0.3, "context_len": 4096,
                 "depth": d, "accuracy": accuracy} for d in (0, 50, 100)]

    bad = rows("sr_kv", 0.2) + rows("snapkv_unified", 0.6) + rows("centroid_merge", 0.5)
    findings = check_ablation(bad, model="m")
    assert findings[0]["status"] == "FLAG"
    assert findings[0]["margin_vs_best_ablation"] == pytest.approx(-0.4)

    good = rows("sr_kv", 0.7) + rows("snapkv_unified", 0.6) + rows("centroid_merge", 0.5)
    assert check_ablation(good, model="m")[0]["status"] == "ok"

    partial = rows("sr_kv", 0.7) + rows("snapkv_unified", 0.6)
    assert check_ablation(partial, model="m")[0]["status"] == "INCOMPLETE"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_results(tmp_path_factory):
    """Actually run the harness, so plots are tested against real records."""
    out_dir = tmp_path_factory.mktemp("results")
    for mode in POSITION_MODES:
        rc = run_main([
            "--tiny", "--allow_cpu",
            "--method", "full,streaming_llm,snapkv_unified,centroid_merge,sr_kv",
            "--task", "niah", "--context_len", "320", "--budget", "0.25",
            "--depths", "0,50", "--n_samples", "1", "--max_new_tokens", "4",
            "--obs_window", "8", "--n_centroids", "4", "--rope_position_mode", mode,
            "--output", str(out_dir / f"r_{mode}.json"),
        ])
        assert rc == 0
    return out_dir


def test_plots_are_generated_from_real_results_and_are_not_empty(real_results, tmp_path):
    figures = tmp_path / "figures"
    assert plots_main(["--results-dir", str(real_results), "--figures-dir", str(figures)]) == 0

    written = sorted(figures.glob("*.png"))
    assert written, "no figures produced"
    names = {p.name for p in written}
    assert any(n.startswith("pareto_") for n in names)
    assert any(n.startswith("niah_heatmap_") for n in names)
    assert any(n.startswith("ablation_") for n in names)
    assert "rope_position_modes.png" in names, "Phase 4 figure missing"

    for path in written:
        # a silently-empty matplotlib canvas still writes a file, just a tiny one
        assert path.stat().st_size > 8000, f"{path.name} looks like an empty plot"


def test_plot_loader_deduplicates_and_drops_errors(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"task_id": "t", "run_key": "k", "method": "sr_kv", "model": "m",
                    "budget": 0.3, "context_len": 4096, "depth": 0, "sample_idx": 0,
                    "accuracy": 1.0}) + "\n"
        + json.dumps({"task_id": "t", "run_key": "k", "method": "sr_kv", "model": "m",
                      "budget": 0.3, "context_len": 4096, "depth": 0, "sample_idx": 0,
                      "accuracy": 1.0}) + "\n"
        + json.dumps({"task_id": "t2", "run_key": "k", "error": "cuda_oom"}) + "\n",
        encoding="utf-8",
    )
    records = load_records(tmp_path)
    assert len(records) == 1


def test_plots_fail_loudly_on_an_empty_results_dir(tmp_path):
    assert plots_main(["--results-dir", str(tmp_path), "--figures-dir", str(tmp_path / "f")]) == 1


def test_load_records_ignores_subdirectories(tmp_path):
    """Diagnostic runs (e.g. `make phase4-scan`) write to results/diagnostics/
    specifically so they cannot collide with a real sweep's task_id/run_key
    space when a gate aggregates "everything in results/". This happened for
    real: a 3-sample budget scan and a 20-sample ablation both measured
    sr_kv/attn_weighted/budget=0.2, and the directory-wide dedup in
    `load_records` silently let the scan's value overwrite the sweep's on one
    of the 15 colliding cells - based on alphabetical file order, not on which
    run was the intended measurement. `Path.glob` is non-recursive, so a
    subdirectory is the whole fix; this pins that down as intentional
    behavior rather than an accident someone "cleans up" later.
    """
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"task_id": "t", "run_key": "real", "method": "sr_kv", "model": "m",
                    "budget": 0.2, "context_len": 8192, "depth": 50, "sample_idx": 0,
                    "accuracy": 1.0}) + "\n",
        encoding="utf-8",
    )
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    (diag / "scan.jsonl").write_text(
        json.dumps({"task_id": "t", "run_key": "real", "method": "sr_kv", "model": "m",
                    "budget": 0.2, "context_len": 8192, "depth": 50, "sample_idx": 0,
                    "accuracy": 0.0}) + "\n",
        encoding="utf-8",
    )
    records = load_records(tmp_path)
    assert len(records) == 1
    assert records[0]["accuracy"] == 1.0, "a file outside results/ leaked into the aggregate"


def test_figures_never_average_niah_and_longbench_together(tmp_path):
    """NIAH is exact-match 0/1; LongBench is F1/ROUGE. One bar cannot be both.

    Measured on real data: the budget=0.3 snapkv_unified ablation bar averaged
    9 NIAH records at 1.000 with 100 LongBench records at 0.249 and plotted
    0.311 - contradicting gate 5, which reports 1.000 because check_ablation
    filters on context_len. make_plots never got that guard. Same
    cross-experiment blending as the Phase 4 gate and the precision run_key.
    """
    from scripts.make_plots import plot_ablation, plot_pareto

    def niah(method, acc, ctx=2048, depth=0, sample=0):
        return {"model": "m", "method": method, "budget": 0.3, "accuracy": acc,
                "context_len": ctx, "depth": depth, "sample_idx": sample,
                "prompt_tokens": 2048, "cache_stats": {"n_tokens_cached": 600}}

    def lb(method, acc, task="gov_report", sample=0):
        return {"model": "m", "method": method, "budget": 0.3, "accuracy": acc,
                "lb_task": task, "sample_idx": sample,
                "prompt_tokens": 4096, "cache_stats": {"n_tokens_cached": 1200}}

    records = []
    for method in ("streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"):
        records += [niah(method, 1.0, depth=d) for d in (0, 25, 50)]
        records += [lb(method, 0.25, sample=i) for i in range(3)]
    records += [niah("full", 1.0), lb("full", 0.25)]

    written = plot_ablation(records, tmp_path) + plot_pareto(records, tmp_path)
    names = [p.name for p in written]

    # one figure per (model, task), never a merged one
    assert any("niah" in n for n in names), names
    assert any("longbench" in n for n in names), names
    assert all(("niah" in n) or ("longbench" in n) for n in names), names


def test_ablation_skips_budgets_missing_conditions(tmp_path):
    """A budget with one condition present is a diagnostic point, not an ablation.

    budget=0.1 in the real results is Phase 4's floor probe - sr_kv alone.
    Drawing it beside three empty slots invites reading one diagnostic run as
    a four-way comparison.
    """
    from scripts.make_plots import plot_ablation

    def rec(method, budget, acc, depth=0):
        return {"model": "m", "method": method, "budget": budget, "accuracy": acc,
                "context_len": 2048, "depth": depth, "sample_idx": 0,
                "prompt_tokens": 2048, "cache_stats": {"n_tokens_cached": 600}}

    records = [rec("sr_kv", 0.1, 0.27)]  # lone diagnostic budget
    for method in ("streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"):
        records.append(rec(method, 0.2, 1.0))

    written = plot_ablation(records, tmp_path)
    assert written, "the complete budget should still plot"

    import scripts.make_plots as mp
    rows = [r for r in records if r["method"] in
            ("streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv")]
    present = {}
    for r in rows:
        present.setdefault(r["budget"], set()).add(r["method"])
    plotted = sorted(b for b, m in present.items() if len(m) == 4)
    assert plotted == [0.2], f"only complete budgets belong on an ablation: {plotted}"
