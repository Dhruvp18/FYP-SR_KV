"""Regenerate every report figure from `results/`.

    python scripts/make_plots.py                  # all four figure families
    python scripts/make_plots.py --only pareto
    python scripts/make_plots.py --results-dir results --figures-dir figures

Reads the raw JSON/JSONL that `eval/run.py` writes - never synthetic data - so
a figure that comes out empty means the sweep is incomplete, and
`scripts/check_results.py completeness` will say which cells are missing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

#: display names and a stable colour per condition, so every figure agrees
CONDITION_STYLE = {
    "full": ("Uncompressed", "#444444", "o"),
    "streaming_llm": ("StreamingLLM", "#1b9e77", "s"),
    "snapkv": ("SnapKV (reference)", "#7570b3", "^"),
    "snapkv_unified": ("SnapKV-style hard evict", "#7570b3", "^"),
    "centroid_merge": ("Centroid merge (no recency)", "#d95f02", "D"),
    "recency_hard_evict": ("Recency + hard evict", "#a6761d", "v"),
    "sr_kv": ("SR-KV (full)", "#e7298a", "*"),
}


def style(method: str):
    return CONDITION_STYLE.get(method, (method, "#999999", "o"))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_records(results_dir: Path, include_errors: bool = False) -> list[dict]:
    """Every record from every result file, de-duplicated.

    Failed tasks (CUDA OOM) are dropped by default so figures never plot them,
    but the phase gates need to see them - an OOM is itself a result.
    """
    records: dict[tuple, dict] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    _add(records, json.loads(line), include_errors)
                except json.JSONDecodeError:
                    continue
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for record in payload.get("records", []):
            _add(records, record, include_errors)
    return list(records.values())


def _add(store: dict, record: dict, include_errors: bool = False) -> None:
    if "error" in record and not include_errors:
        return
    key = (
        record.get("model"),
        record.get("method"),
        record.get("budget"),
        record.get("context_len"),
        record.get("lb_task"),
        record.get("depth"),
        record.get("sample_idx"),
        record.get("run_key"),
    )
    store[key] = record


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _group(records, *keys):
    out = defaultdict(list)
    for r in records:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# (a) memory vs accuracy Pareto
# ---------------------------------------------------------------------------
def plot_pareto(records, figures_dir: Path) -> list[Path]:
    """Accuracy against KV cache size, one panel per model."""
    written = []
    by_model = _group(records, "model")
    for (model,), rows in sorted(by_model.items(), key=lambda kv: str(kv[0])):
        fig, ax = plt.subplots(figsize=(7, 5))
        for (method,), method_rows in sorted(_group(rows, "method").items()):
            label, colour, marker = style(method)
            points = []
            for (budget,), budget_rows in sorted(_group(method_rows, "budget").items()):
                cached = _mean(r["cache_stats"]["n_tokens_cached"] for r in budget_rows)
                prompt = _mean(r["prompt_tokens"] for r in budget_rows)
                points.append((100.0 * cached / max(prompt, 1), _mean(r["accuracy"] for r in budget_rows)))
            if not points:
                continue
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=marker, color=colour, label=label,
                    linewidth=1.6, markersize=8 if marker != "*" else 12)

        ax.set_xlabel("KV cache retained (% of prompt tokens)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Memory / accuracy trade-off - {model}")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
        written.append(_save(fig, figures_dir / f"pareto_{model}.png"))
    return written


# ---------------------------------------------------------------------------
# (b) NIAH heatmaps
# ---------------------------------------------------------------------------
def plot_niah_heatmaps(records, figures_dir: Path) -> list[Path]:
    """Depth x context-length accuracy grid, one panel per condition."""
    niah = [r for r in records if r.get("context_len") is not None and r.get("depth") is not None]
    written = []
    for (model, budget), rows in sorted(
        _group(niah, "model", "budget").items(), key=lambda kv: str(kv[0])
    ):
        methods = sorted({r["method"] for r in rows})
        if not methods:
            continue
        depths = sorted({r["depth"] for r in rows})
        contexts = sorted({r["context_len"] for r in rows})

        fig, axes = plt.subplots(
            1, len(methods), figsize=(3.1 * len(methods), 3.4), squeeze=False, sharey=True
        )
        for ax, method in zip(axes[0], methods):
            grid = np.full((len(depths), len(contexts)), np.nan)
            cells = _group([r for r in rows if r["method"] == method], "depth", "context_len")
            for (depth, ctx), cell_rows in cells.items():
                grid[depths.index(depth), contexts.index(ctx)] = _mean(
                    r["accuracy"] for r in cell_rows
                )
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
            ax.set_title(style(method)[0], fontsize=9)
            ax.set_xticks(range(len(contexts)))
            ax.set_xticklabels([f"{c // 1024}k" for c in contexts], fontsize=8)
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"{d}%" for d in depths], fontsize=8)
            ax.set_xlabel("context")
            for i in range(len(depths)):
                for j in range(len(contexts)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
        axes[0][0].set_ylabel("needle depth")
        fig.suptitle(f"NIAH accuracy - {model}, budget={budget}", fontsize=11)
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02)
        written.append(_save(fig, figures_dir / f"niah_heatmap_{model}_b{budget}.png"))
    return written


# ---------------------------------------------------------------------------
# (c) factorial ablation
# ---------------------------------------------------------------------------
def plot_ablation(records, figures_dir: Path) -> list[Path]:
    """The four-condition matrix, so each ingredient's contribution is visible."""
    order = ["streaming_llm", "snapkv_unified", "centroid_merge", "sr_kv"]
    written = []
    for (model,), rows in sorted(_group(records, "model").items(), key=lambda kv: str(kv[0])):
        budgets = sorted({r["budget"] for r in rows if r["method"] in order})
        if not budgets:
            continue
        fig, ax = plt.subplots(figsize=(1.6 * len(budgets) + 4, 4.5))
        width = 0.8 / max(len(order), 1)
        x = np.arange(len(budgets))
        for i, method in enumerate(order):
            label, colour, _ = style(method)
            heights = [
                _mean(
                    r["accuracy"] for r in rows
                    if r["method"] == method and r["budget"] == budget
                )
                for budget in budgets
            ]
            ax.bar(x + i * width, heights, width, label=label, color=colour)

        baseline = [r["accuracy"] for r in rows if r["method"] == "full"]
        if baseline:
            ax.axhline(_mean(baseline), color="#444444", linestyle="--", linewidth=1.2,
                       label="Uncompressed")
        ax.set_xticks(x + width * (len(order) - 1) / 2)
        ax.set_xticklabels([f"budget {b}" for b in budgets])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Ablation: recency and clustering contributions - {model}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        written.append(_save(fig, figures_dir / f"ablation_{model}.png"))
    return written


# ---------------------------------------------------------------------------
# (d) RoPE position mode comparison (Phase 4)
# ---------------------------------------------------------------------------
def plot_rope_modes(records, figures_dir: Path) -> list[Path]:
    # only the merging conditions have a meaningful centroid position
    rope = [
        r for r in records
        if r.get("rope_position_mode") and r.get("method") in ("sr_kv", "centroid_merge")
    ]
    if not rope:
        print("  [skip] no rope-mode-tagged records; run Phase 4 first")
        return []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    modes = ["earliest", "attn_weighted", "latest"]
    depths = sorted({r["depth"] for r in rope if r.get("depth") is not None})
    for mode in modes:
        rows = [r for r in rope if r["rope_position_mode"] == mode]
        if not rows:
            continue
        ys = [_mean(r["accuracy"] for r in rows if r.get("depth") == d) for d in depths]
        ax.plot(depths, ys, marker="o", label=mode, linewidth=1.6)

    ax.set_xlabel("needle depth (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Centroid RoPE position convention (Phase 4)")
    ax.grid(alpha=0.3)
    ax.legend()
    return [_save(fig, figures_dir / "rope_position_modes.png")]


PLOTS = {
    "pareto": plot_pareto,
    "niah": plot_niah_heatmaps,
    "ablation": plot_ablation,
    "rope": plot_rope_modes,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="regenerate SR-KV figures")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--only", choices=sorted(PLOTS), default=None)
    args = parser.parse_args(argv)

    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    records = load_records(results_dir)
    print(f"loaded {len(records)} records from {results_dir}")
    if not records:
        print("no records found - nothing to plot")
        return 1

    written: list[Path] = []
    for name, fn in PLOTS.items():
        if args.only and name != args.only:
            continue
        print(f"[{name}]")
        written.extend(fn(records, figures_dir))

    print(f"\n{len(written)} figure(s) written to {figures_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
