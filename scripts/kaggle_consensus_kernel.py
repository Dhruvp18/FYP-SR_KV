"""Generate, push, poll and pull the consensus_update_v1 Kaggle kernel.

Not integrated into kaggle_kernel.py's PHASES dict on purpose: this is an
exploratory application pilot (experiments/consensus_update_protocol.md),
same tier as conversation_pilot_v1/incident_stream_v1, not one of the eight
core phases that script drives. Reuses the same proven cell-building pattern
(sh() helper that tees to a persistent log, CUDA allocator env, --force pull)
rather than re-deriving it.

    python scripts/kaggle_consensus_kernel.py generate --user <you>
    python scripts/kaggle_consensus_kernel.py push
    python scripts/kaggle_consensus_kernel.py status --user <you>
    python scripts/kaggle_consensus_kernel.py pull   --user <you>
    python scripts/kaggle_consensus_kernel.py run    --user <you>   # all four, polling
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
SLUG = "sr-kv-consensus-update"
POLL_SECONDS = 120
MAX_POLL_HOURS = 8


def build_metadata(user: str, *, private: bool = True) -> dict:
    return {
        "id": f"{user}/{SLUG}",
        "title": SLUG,  # identical to the slug - see kaggle_kernel.py's comment on why
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def build_notebook(*, repo: str) -> dict:
    def md(*lines):
        return {"cell_type": "markdown", "metadata": {}, "source": [f"{line}\n" for line in lines]}

    def code(*lines):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [f"{line}\n" for line in lines]}

    cells = [
        md("# SR-KV consensus_update_v1 pilot",
           "",
           "Exploratory, deliberately favourable-by-design pilot - see "
           "experiments/consensus_update_protocol.md for the pre-registered hypothesis "
           "and success criterion. Runs sanity, then the pilot only if sanity passes, "
           "then the paired-bootstrap analysis, all in one GPU session (one Kaggle slot)."),

        code("!nvidia-smi || echo 'no GPU'",
             "import torch",
             "print('cuda:', torch.cuda.is_available())"),

        code("import subprocess, sys",
             "",
             "REPO = %r" % repo,
             "WORKDIR = '/kaggle/working/sr-kv'",
             "LOGFILE = '/kaggle/working/run.log'",
             "",
             "def sh(cmd, cwd=WORKDIR, check=True):",
             "    print('+', cmd, flush=True)",
             "    with open(LOGFILE, 'a') as logf:",
             "        logf.write(f'+ {cmd}\\n'); logf.flush()",
             "        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,",
             "                                stderr=subprocess.STDOUT, text=True, bufsize=1)",
             "        for line in proc.stdout:",
             "            print(line, end='', flush=True); logf.write(line); logf.flush()",
             "        proc.wait()",
             "    if check and proc.returncode != 0:",
             "        raise SystemExit(f'FAILED ({proc.returncode}): {cmd}')",
             "    return proc.returncode"),

        code("import os",
             "if os.path.isdir(WORKDIR):",
             "    sh('git pull -q', check=False)",
             "else:",
             "    sh(f'git clone -q {REPO} {WORKDIR}', cwd='/kaggle/working')",
             "sh('git log --oneline -1')"),

        code("import os",
             "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'",
             "os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'"),

        code("sh(\"pip install -q -U 'transformers>=5.0' accelerate bitsandbytes\", cwd='/kaggle/working')",
             "import transformers; print('transformers', transformers.__version__)"),

        md("## Restore any earlier progress",
           "",
           "This kernel resumes from its OWN previous output via `git pull` only - unlike the",
           "phaseN kernels it mounts no `kernel_sources`, since this pilot has no earlier phase",
           "to build on. Re-pushing after committing partial `results/consensus_update_v1/` "
           "progress is how a killed run resumes."),

        md("## Quick CPU test suite"),
        code("sh('python -m pytest -q')"),

        md("## Sanity set (5 conversations, all variants/methods/budgets)",
           "",
           "Must pass before the 50-conversation pilot runs - see "
           "experiments/consensus_update_protocol.md."),
        code("sh('python scripts/run_consensus_update.py --stage sanity')"),

        code("import json",
             "gate = json.load(open(f'{WORKDIR}/results/consensus_update_v1/sanity_gate.json'))",
             "print(json.dumps(gate, indent=2))",
             "if not gate['pass']:",
             "    raise SystemExit('Sanity gate failed - stopping before the pilot set, per protocol.')"),

        md("## Pilot set (50 conversations)"),
        code("sh('python scripts/run_consensus_update.py --stage pilot')"),

        md("## Pre-registered analysis"),
        code("sh('python scripts/analyse_consensus_update.py "
             "results/consensus_update_v1/pilot.jsonl')"),

        md("## Results are the kernel output"),
        code("sh('ls -la results/consensus_update_v1 | head -40', check=False)"),
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


def _run(argv, *, dry_run=False):
    """Run a `kaggle ...` argv, falling back to `python -m kaggle` when the
    console script isn't on PATH (confirmed the case on this machine - the
    pip install landed the package but not a PATH entry)."""
    if argv[0] == "kaggle" and shutil.which("kaggle") is None:
        argv = [sys.executable, "-m", "kaggle"] + argv[1:]
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    return subprocess.run(argv, capture_output=True, text=True)


def parse_status(text: str) -> str:
    lowered = (text or "").lower()
    for state in ("complete", "error", "cancel", "running", "queued"):
        if state in lowered:
            return "complete" if state == "complete" else state
    return "unknown"


def cmd_generate(args) -> int:
    directory = Path(args.out) / "consensus_update"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(args.user, private=not args.public)
    notebook = build_notebook(repo=args.repo)
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / f"{SLUG}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {directory}/kernel-metadata.json")
    print(f"wrote {directory}/{SLUG}.ipynb")
    return 0


def cmd_push(args) -> int:
    directory = Path(args.out) / "consensus_update"
    if not (directory / "kernel-metadata.json").exists():
        raise SystemExit(f"{directory} has no kernel-metadata.json; run `generate` first")
    result = _run(["kaggle", "kernels", "push", "-p", str(directory)], dry_run=args.dry_run)
    print(result.stdout or "", result.stderr or "")
    return result.returncode


def cmd_status(args) -> int:
    result = _run(["kaggle", "kernels", "status", f"{args.user}/{SLUG}"], dry_run=args.dry_run)
    print(result.stdout or "", result.stderr or "")
    return result.returncode


def cmd_pull(args) -> int:
    out_dir = Path(args.download_to or (REPO_ROOT / ".kaggle_output" / "consensus_update"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        ["kaggle", "kernels", "output", f"{args.user}/{SLUG}", "-p", str(out_dir), "--force"],
        dry_run=args.dry_run,
    )
    print(result.stdout or "", result.stderr or "")
    if args.dry_run:
        return 0
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in out_dir.rglob("*"):
        if not (path.is_file() and path.suffix in (".jsonl", ".json")):
            continue
        parts = path.parts
        if "consensus_update_v1" not in parts:
            continue
        # Preserve the path from consensus_update_v1 onward (same fix as
        # kaggle_kernel.py's merge_outputs - flattening to just the basename
        # let the repo's own committed sanity_v1_rejected_decoy/ record
        # silently overwrite the live run's same-named file, because both
        # cloned-repo history and fresh kernel output land under this same
        # directory name. Confirmed happening on this pilot's first pull.
        idx = parts.index("consensus_update_v1")
        dest = results_dir / Path(*parts[idx:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
        copied += 1
    print(f"merged {copied} file(s) into results/consensus_update_v1/")
    return result.returncode


def cmd_run(args) -> int:
    cmd_generate(args)
    if cmd_push(args) != 0:
        return 1
    if args.dry_run:
        print("(dry run: skipping poll)")
        return 0
    print(f"\npolling every {POLL_SECONDS}s (cap {MAX_POLL_HOURS}h)")
    deadline = time.time() + MAX_POLL_HOURS * 3600
    state = "unknown"
    while time.time() < deadline:
        result = _run(["kaggle", "kernels", "status", f"{args.user}/{SLUG}"])
        state = parse_status(result.stdout + result.stderr)
        print(f"  [{time.strftime('%H:%M:%S')}] {state}", flush=True)
        if state in ("complete", "error", "cancel"):
            break
        time.sleep(POLL_SECONDS)
    else:
        print("timed out waiting for the kernel; check the Kaggle UI")
        return 1
    cmd_pull(args)
    if state != "complete":
        print(f"kernel finished in state {state!r} - inspect results/consensus_update_v1/ and the log")
        return 1
    print("\n+ python scripts/analyse_consensus_update.py results/consensus_update_v1/pilot.jsonl")
    return subprocess.run(
        [sys.executable, "scripts/analyse_consensus_update.py",
         "results/consensus_update_v1/pilot.jsonl"],
        cwd=REPO_ROOT,
    ).returncode


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["generate", "push", "status", "pull", "run"])
    p.add_argument("--user", default=None)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--out", default=str(REPO_ROOT / "kaggle"))
    p.add_argument("--download-to", default=None)
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("generate", "status", "pull", "run") and not args.user:
        raise SystemExit("--user (your Kaggle username) is required for this command")
    return {
        "generate": cmd_generate, "push": cmd_push,
        "status": cmd_status, "pull": cmd_pull, "run": cmd_run,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
