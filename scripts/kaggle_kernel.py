"""Drive the GPU phases on Kaggle without a browser.

Kaggle notebooks pushed through the API run **detached** (the same thing the UI
calls "Save & Run All"), which is what makes the whole remaining pipeline
automatable: push a kernel, poll until it finishes, pull its output, check the
phase gate locally, move to the next phase.

    python scripts/kaggle_kernel.py generate --phase 5 --user dhruvp18
    python scripts/kaggle_kernel.py push     --phase 5
    python scripts/kaggle_kernel.py status   --phase 5 --user dhruvp18
    python scripts/kaggle_kernel.py pull     --phase 5 --user dhruvp18
    python scripts/kaggle_kernel.py run      --phase 5 --user dhruvp18   # all four, polling

`run` is the one an agent calls. It ends by executing the phase's gate against
the pulled results and exits with the gate's own status, so the agent's decision
to continue is made on evidence rather than on the kernel having finished.

Requires Kaggle API credentials (`~/.kaggle/kaggle.json`, chmod 600). A Kaggle
MCP server wraps these same operations; if you have one connected, the tool
names differ but the sequence is identical - see HANDOFF.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "https://github.com/Dhruvp18/FYP-SR_KV.git"

#: phase -> the Makefile targets it runs. The Makefile stays the single source
#: of truth for what a phase actually does; this only says which targets and in
#: what order, so the two cannot drift apart.
PHASES: dict[int, dict] = {
    1: {"title": "harness sanity", "targets": ["phase1", "phase1-4bit"], "gate": "gate1",
        "note": "Uncompressed NIAH at 512 tokens must exceed 0.9 or the harness is broken."},
    2: {"title": "baselines", "targets": ["phase2"], "gate": "gate2",
        "note": "StreamingLLM must underperform SnapKV at mid-sequence depths."},
    3: {"title": "unified class at 8k", "targets": ["phase3"], "gate": "gate3",
        "note": "Conservation and budget invariants must hold across a full 8k run."},
    4: {"title": "RoPE position ablation", "targets": ["phase4"], "gate": "gate4",
        "after_gate": ["freeze-rope"],
        "note": "Freezes the winner into configs/defaults.yaml. Commit that file afterwards."},
    5: {"title": "factorial matrix + LongBench", "targets": ["phase5", "phase5-longbench"],
        "gate": "gate5",
        "note": "The long one. Shard it if you have more than one account."},
    6: {"title": "hyperparameter sweep + 3B transfer",
        "targets": ["phase6-sweep", "phase6-3b"], "gate": "gate6",
        "note": "No re-sweep on 3B: the question is whether the 1.5B config transfers."},
    7: {"title": "figures", "targets": ["phase7"], "gate": "gate7",
        "note": "Regenerates every figure from results/ in one command."},
}

POLL_SECONDS = 120
MAX_POLL_HOURS = 12


# ---------------------------------------------------------------------------
# pure builders (unit-tested)
# ---------------------------------------------------------------------------
def slug(phase: int) -> str:
    return f"sr-kv-phase{phase}"


def build_metadata(phase: int, user: str, *, depends_on: list[str] | None = None,
                   private: bool = True) -> dict:
    """kernel-metadata.json for `kaggle kernels push`."""
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase}; known: {sorted(PHASES)}")
    if depends_on is None:
        # by default a phase mounts every earlier phase's output, which is what
        # makes a later run resume rather than redo finished work
        depends_on = [f"{user}/{slug(p)}" for p in sorted(PHASES) if p < phase]
    return {
        "id": f"{user}/{slug(phase)}",
        "title": f"SR-KV Phase {phase} - {PHASES[phase]['title']}",
        "code_file": f"{slug(phase)}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        # phase 7 only reads results and draws figures, so it needs no GPU
        "enable_gpu": phase != 7,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        # previous phases' outputs get mounted under /kaggle/input, which is how
        # a later phase resumes from an earlier one's results
        "kernel_sources": list(depends_on or []),
    }


def build_commands(phase: int, *, model: str, model3b: str, budget: float, samples: int,
                   shard: int, num_shards: int) -> list[str]:
    """The make invocations this phase runs, in order."""
    spec = PHASES[phase]
    variables = (
        f"MODEL={model} MODEL3B={model3b} BUDGET={budget} "
        f"SAMPLES={samples} SHARD={shard} NSHARDS={num_shards}"
    )
    commands = [f"make {target} {variables}" for target in spec["targets"]]
    commands.append(f"make {spec['gate']} {variables}")
    commands += [f"make {target} {variables}" for target in spec.get("after_gate", [])]
    return commands


def build_notebook(phase: int, *, repo: str, model: str, model3b: str, budget: float,
                   samples: int, shard: int, num_shards: int) -> dict:
    """A thin notebook: clone, install, restore, run make targets, check the gate.

    No project logic lives here - it shells out to the Makefile, same as a local
    run would.
    """
    spec = PHASES[phase]
    commands = build_commands(phase, model=model, model3b=model3b, budget=budget,
                              samples=samples, shard=shard, num_shards=num_shards)

    def md(*lines):
        return {"cell_type": "markdown", "metadata": {}, "source": [f"{line}\n" for line in lines]}

    def code(*lines):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [f"{line}\n" for line in lines]}

    cells = [
        md(f"# SR-KV Phase {phase} - {spec['title']}",
           "",
           f"{spec['note']}",
           "",
           "Generated by `scripts/kaggle_kernel.py`. Thin by design: it clones the repo and",
           "calls Makefile targets, so what runs here is exactly what runs locally."),

        code("!nvidia-smi || echo 'no GPU (expected for phase 7)'",
             "import torch",
             "print('cuda:', torch.cuda.is_available())"),

        code("import subprocess, sys",
             "",
             "REPO = %r" % repo,
             "WORKDIR = '/kaggle/working/sr-kv'",
             "",
             "def sh(cmd, cwd=WORKDIR, check=True):",
             "    \"\"\"Run a shell command, streaming output; raise so a failure fails the kernel.\"\"\"",
             "    print('+', cmd, flush=True)",
             "    result = subprocess.run(cmd, shell=True, cwd=cwd)",
             "    if check and result.returncode != 0:",
             "        raise SystemExit(f'FAILED ({result.returncode}): {cmd}')",
             "    return result.returncode"),

        code("import os",
             "if os.path.isdir(WORKDIR):",
             "    sh('git pull -q', check=False)",
             "else:",
             "    sh(f'git clone -q {REPO} {WORKDIR}', cwd='/kaggle/working')",
             "sh('git log --oneline -1')"),

        code("# transformers 5.x is required (src/compat.py raises otherwise). torch ships with the image.",
             "sh(\"pip install -q -U 'transformers>=5.0' accelerate bitsandbytes\", cwd='/kaggle/working')",
             "import transformers; print('transformers', transformers.__version__)"),

        md("## Restore results from earlier phases",
           "",
           "Any kernel listed in `kernel_sources` is mounted under `/kaggle/input`. Copying its",
           "`.jsonl` files in is what makes this run resume instead of redoing finished work."),

        code("import glob, shutil, os",
             "os.makedirs(f'{WORKDIR}/results', exist_ok=True)",
             "restored = 0",
             "for pattern in ('/kaggle/input/*/results/*.jsonl', '/kaggle/input/*/sr-kv/results/*.jsonl'):",
             "    for src in glob.glob(pattern):",
             "        shutil.copy(src, f'{WORKDIR}/results/')",
             "        restored += 1",
             "print(f'restored {restored} result file(s)')",
             "sh('ls -la results | head -20', check=False)"),

        md("## Quick CPU test suite",
           "",
           "Two minutes, no GPU, no downloads. Cheapest possible way to catch a broken commit",
           "before spending quota on it."),
        code("sh('python -m pytest -q')"),

        md(f"## Run phase {phase}",
           "",
           "Resumable: every finished task is fsynced to `results/*.jsonl`, so if this session is",
           "killed, re-running this same kernel continues from where it stopped."),
        code(*[f"sh({command!r})" for command in commands]),

        md("## Results are the kernel output",
           "",
           "Everything under `/kaggle/working` becomes this kernel's output, so `results/` and",
           "`figures/` are pulled down by `kaggle_kernel.py pull` and mounted by the next phase."),
        code("sh('ls -la results figures 2>/dev/null | head -40', check=False)"),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kaggle_argv(action: str, *, phase: int, user: str | None = None,
                directory: Path | None = None, out_dir: Path | None = None) -> list[str]:
    """The `kaggle` CLI invocation for one action. Kept pure so it can be tested."""
    if action == "push":
        return ["kaggle", "kernels", "push", "-p", str(directory)]
    if action == "status":
        return ["kaggle", "kernels", "status", f"{user}/{slug(phase)}"]
    if action == "pull":
        return ["kaggle", "kernels", "output", f"{user}/{slug(phase)}", "-p", str(out_dir)]
    raise ValueError(f"unknown action {action!r}")


# ---------------------------------------------------------------------------
# side-effecting commands
# ---------------------------------------------------------------------------
def _run(argv: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    if shutil.which(argv[0]) is None:
        raise SystemExit(
            f"`{argv[0]}` is not installed. Run `pip install kaggle` and put your API token in "
            "~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token), chmod 600."
        )
    return subprocess.run(argv, capture_output=True, text=True)


def cmd_generate(args) -> int:
    directory = Path(args.out) / f"phase{args.phase}"
    directory.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(
        args.phase, args.user, depends_on=args.depends_on, private=not args.public
    )
    notebook = build_notebook(
        args.phase, repo=args.repo, model=args.model, model3b=args.model3b,
        budget=args.budget, samples=args.samples, shard=args.shard, num_shards=args.num_shards,
    )

    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / f"{slug(args.phase)}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {directory}/kernel-metadata.json")
    print(f"wrote {directory}/{slug(args.phase)}.ipynb")
    print(f"\nnext: python scripts/kaggle_kernel.py push --phase {args.phase}")
    return 0


def cmd_push(args) -> int:
    directory = Path(args.out) / f"phase{args.phase}"
    if not (directory / "kernel-metadata.json").exists():
        raise SystemExit(f"{directory} has no kernel-metadata.json; run `generate` first")
    result = _run(kaggle_argv("push", phase=args.phase, directory=directory), dry_run=args.dry_run)
    print(result.stdout or "", result.stderr or "")
    return result.returncode


def cmd_status(args) -> int:
    result = _run(kaggle_argv("status", phase=args.phase, user=args.user), dry_run=args.dry_run)
    print(result.stdout or "", result.stderr or "")
    return result.returncode


def parse_status(text: str) -> str:
    """Reduce `kaggle kernels status` chatter to one of running/complete/error."""
    lowered = (text or "").lower()
    for state in ("complete", "error", "cancel", "running", "queued"):
        if state in lowered:
            return "complete" if state == "complete" else state
    return "unknown"


def cmd_pull(args) -> int:
    out_dir = Path(args.download_to or (REPO_ROOT / ".kaggle_output" / f"phase{args.phase}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        kaggle_argv("pull", phase=args.phase, user=args.user, out_dir=out_dir),
        dry_run=args.dry_run,
    )
    print(result.stdout or "", result.stderr or "")
    if args.dry_run:
        return 0

    merged = merge_outputs(out_dir, REPO_ROOT / "results", REPO_ROOT / "figures")
    print(f"merged {merged} file(s) into results/ and figures/")
    return result.returncode


def merge_outputs(out_dir: Path, results_dir: Path, figures_dir: Path) -> int:
    """Copy pulled .jsonl/.json/.png back into the repo, skipping what we have."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in (".jsonl", ".json") and "results" in path.parts:
            shutil.copy(path, results_dir / path.name)
            copied += 1
        elif path.suffix == ".png" and "figures" in path.parts:
            shutil.copy(path, figures_dir / path.name)
            copied += 1
    return copied


def cmd_run(args) -> int:
    """generate -> push -> poll -> pull -> gate. The one an agent calls."""
    cmd_generate(args)
    if cmd_push(args) != 0:
        return 1
    if args.dry_run:
        print("(dry run: skipping poll)")
        return 0

    print(f"\npolling every {POLL_SECONDS}s (Kaggle sessions cap out around {MAX_POLL_HOURS}h)")
    deadline = time.time() + MAX_POLL_HOURS * 3600
    while time.time() < deadline:
        result = _run(kaggle_argv("status", phase=args.phase, user=args.user))
        state = parse_status(result.stdout + result.stderr)
        print(f"  [{time.strftime('%H:%M:%S')}] {state}", flush=True)
        if state in ("complete", "error", "cancel"):
            break
        time.sleep(POLL_SECONDS)
    else:
        print("timed out waiting for the kernel; check the Kaggle UI")
        return 1

    if state != "complete":
        print(f"kernel finished in state {state!r}; pulling output anyway for the logs")

    cmd_pull(args)

    gate = ["python", "scripts/check_results.py", "gate", "--phase", str(args.phase),
            "--model", args.model3b if args.phase == 6 else args.model,
            "--budget", str(args.budget), "--n-samples", str(args.samples)]
    print("\n+ " + " ".join(gate))
    return subprocess.run(gate, cwd=REPO_ROOT).returncode


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["generate", "push", "status", "pull", "run"])
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASES))
    parser.add_argument("--user", default=None, help="your Kaggle username")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", default=str(REPO_ROOT / "kaggle"))
    parser.add_argument("--model", default="qwen2.5-1.5b")
    parser.add_argument("--model3b", default="llama3.2-3b")
    parser.add_argument("--budget", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--depends-on", action="append", default=None,
                        help="kernel to mount for resume, e.g. user/sr-kv-phase4 (repeatable)")
    parser.add_argument("--download-to", default=None)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands, call nothing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("generate", "status", "pull", "run") and not args.user:
        raise SystemExit("--user (your Kaggle username) is required for this command")

    return {
        "generate": cmd_generate,
        "push": cmd_push,
        "status": cmd_status,
        "pull": cmd_pull,
        "run": cmd_run,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
