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
    # Phase 8's "gate" is not a check_results gate. Its pass condition was
    # written down in PREREGISTRATION.md before any data existed and is
    # evaluated by scripts/analyse_phase8.py, which always exits 0: NOT
    # SUPPORTED is a result to report, not a pipeline failure. `local_gate`
    # is what cmd_run runs instead of `check_results.py gate --phase 8`,
    # which does not exist and must not be invented after seeing numbers.
    8: {"title": "pre-registered follow-ups (E1 perplexity, E2 tight budgets)",
        "targets": ["phase8-perplexity", "phase8-tight-longbench"],
        "gate": "phase8-analyse",
        "local_gate": ["python", "scripts/analyse_phase8.py"],
        "note": "Tests whether centroid-merging wins under a distributional metric (H1) "
                "or under real compression pressure (H2). Criterion fixed in "
                "PREREGISTRATION.md; do not change it after seeing results."},
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
        # Kaggle derives the kernel's *live* slug from this title, not from
        # `id` - a title with spaces/punctuation would slugify to something
        # else (e.g. "SR-KV Phase 1 - harness sanity" -> "sr-kv-phase-1-
        # harness-sanity") and every later status/pull call, which is built
        # from slug(phase), would 404. Keeping the title identical to the
        # slug is what makes `id` and the real kernel URL agree.
        "title": slug(phase),
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
                   shard: int, num_shards: int, targets: list[str] | None = None,
                   skip_gate: bool = False, extra_vars: str = "") -> list[str]:
    """The make invocations this phase runs, in order.

    `targets` overrides the phase's usual target list, and `skip_gate` drops
    the gate. Both exist for diagnostic runs like `phase4-scan`, which answers
    "which budget is even worth measuring at" and has no pass condition of its
    own - running that phase's gate against it would just fail on a sweep that
    was never meant to satisfy it.
    """
    spec = PHASES[phase]
    variables = (
        f"MODEL={model} MODEL3B={model3b} BUDGET={budget} "
        f"SAMPLES={samples} SHARD={shard} NSHARDS={num_shards}"
    )
    if extra_vars:
        variables = f"{variables} {extra_vars}"
    commands = [f"make {target} {variables}" for target in (targets or spec["targets"])]
    if not skip_gate:
        commands.append(f"make {spec['gate']} {variables}")
        commands += [f"make {target} {variables}" for target in spec.get("after_gate", [])]
    return commands


def build_notebook(phase: int, *, repo: str, model: str, model3b: str, budget: float,
                   samples: int, shard: int, num_shards: int,
                   targets: list[str] | None = None, skip_gate: bool = False,
                   extra_vars: str = "") -> dict:
    """A thin notebook: clone, install, restore, run make targets, check the gate.

    No project logic lives here - it shells out to the Makefile, same as a local
    run would.
    """
    spec = PHASES[phase]
    commands = build_commands(phase, model=model, model3b=model3b, budget=budget,
                              samples=samples, shard=shard, num_shards=num_shards,
                              targets=targets, skip_gate=skip_gate, extra_vars=extra_vars)
    # The gate command (if present) is always the one right after the phase's
    # own targets - see build_commands. Split it out so its exit code 2
    # ("needs a human look" - a negative or non-significant finding, not a
    # crash) can be handled separately from a real failure. Every other
    # command still uses the blunt check=True/raise-on-any-nonzero path.
    n_pre = len(targets if targets is not None else spec["targets"])
    run_commands, gate_command, after_commands = commands[:n_pre], None, []
    if not skip_gate:
        gate_command, after_commands = commands[n_pre], commands[n_pre + 1:]

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

        md("## CUDA allocator config",
           "",
           "Long runs with many different tensor shapes (LongBench's four tasks have very",
           "different sequence/generation lengths) fragment PyTorch's caching allocator over",
           "hundreds of tasks: `torch.cuda.empty_cache()` between tasks (already in",
           "eval/run.py) releases cached blocks but does not defragment them, so a run can OOM",
           "on a request far smaller than the free memory actually reports. Confirmed on a real",
           "run: OOM'd requesting 1.15 GiB with only 198 MiB contiguous free, 81 tasks in.",
           "`expandable_segments` is PyTorch's own documented fix for exactly this - it's what",
           "the OOM error message itself suggests. Set here (env var, inherited by every `make`/",
           "`python eval/run.py` subprocess below) rather than in eval/run.py, since it has to",
           "apply before CUDA is initialized in whichever process actually does the allocating."),
        code("import os",
             "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'",
             "os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # newer torch name"),

        code("# transformers 5.x is required (src/compat.py raises otherwise). torch ships with the image.",
             "sh(\"pip install -q -U 'transformers>=5.0' accelerate bitsandbytes\", cwd='/kaggle/working')",
             "import transformers; print('transformers', transformers.__version__)"),

        md("## Hugging Face authentication",
           "",
           "meta-llama/Llama-3.2-3B-Instruct (and llama3.2-1b) are gated repos: downloading",
           "them needs an HF token from an account that has accepted Meta's license at",
           "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct. Add that token as a Kaggle",
           "secret named `HF_TOKEN` (notebook editor -> Add-ons -> Secrets). Qwen models are not",
           "gated and work with no token."),
        code("try:",
             "    from kaggle_secrets import UserSecretsClient",
             "    hf_token = UserSecretsClient().get_secret('HF_TOKEN')",
             "except Exception:",
             "    hf_token = None",
             "if hf_token:",
             "    from huggingface_hub import login",
             "    login(token=hf_token)",
             "    print('Hugging Face: authenticated via HF_TOKEN secret')",
             "else:",
             "    print('Hugging Face: no HF_TOKEN secret found - gated models (Llama) will fail to download')"),

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
        code(*[f"sh({command!r})" for command in run_commands]),
    ]

    if gate_command is not None:
        after_lines = [f"    sh({c!r})" for c in after_commands] or ["    pass"]
        cells.append(
            md("## Gate",
               "",
               "Exit 0 = proceed (runs the after-gate step below, if any). Exit 1 = a real",
               "failure - fails the kernel, same as any other command here. Exit 2 = \"needs a",
               "human look\": a negative or statistically non-significant finding, not a crash -",
               "this is the pipeline working as designed (see CLAUDE.md / HANDOFF.md), so it does",
               "NOT fail the kernel, it just skips the after-gate step and prints why.")
        )
        cells.append(code(
            f"gate_rc = sh({gate_command!r}, check=False)",
            "if gate_rc == 1:",
            f"    raise SystemExit(f'FAILED ({{gate_rc}}): {gate_command}')",
            "elif gate_rc == 2:",
            "    print()",
            "    print('=' * 70)",
            "    print('GATE EXITED 2: needs a human look - see the gate output above.')",
            "    print('This is not a crash. Skipping the after-gate step (if any).')",
            "    print('=' * 70)",
            "else:",
            *after_lines,
        ))

    cells += [
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
        # --force is not optional. Without it the CLI skips any file whose
        # local copy looks newer ("Skipping, found more recently modified
        # local copy"), so a re-run of the same phase silently serves the
        # PREVIOUS run's results from .kaggle_output/ and the gate then scores
        # data that never came from the run you just did.
        return ["kaggle", "kernels", "output", f"{user}/{slug(phase)}",
                "-p", str(out_dir), "--force"]
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
        targets=args.target, skip_gate=args.skip_gate, extra_vars=args.make_var or "",
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
    """Copy pulled .jsonl/.json/.png back into the repo, skipping what we have.

    Preserves the path *relative to* the results/figures directory component
    (e.g. .../results/diagnostics/x.json lands at results_dir/diagnostics/
    x.json) instead of flattening to just the filename. Flattening silently
    defeated the fix that moved diagnostic-run output (e.g. phase4-scan) out
    of results/ so it can't contaminate a gate's directory-wide aggregation
    (load_records globs non-recursively) - every subsequent pull would
    flatten it right back to the top level, re-introducing exactly the
    collision that move was meant to prevent.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        parts = path.parts
        if path.suffix in (".jsonl", ".json") and "results" in parts:
            dest = results_dir / Path(*parts[parts.index("results") + 1 :])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest)
            copied += 1
        elif path.suffix == ".png" and "figures" in parts:
            dest = figures_dir / Path(*parts[parts.index("figures") + 1 :])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest)
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

    if getattr(args, "skip_gate", False):
        print("\n(--skip-gate: diagnostic run, no pass condition to check)")
        return 0

    gate = PHASES[args.phase].get("local_gate") or [
        "python", "scripts/check_results.py", "gate", "--phase", str(args.phase),
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
    # qwen2.5-3b, not llama3.2-3b: ungated and RoPE-compatible with centroid
    # merging (see the comment on Makefile's MODEL3B). Pass
    # --model3b llama3.2-3b to restore the cross-architecture comparison
    # once HF access to the gated repo is in place.
    parser.add_argument("--model3b", default="qwen2.5-3b")
    parser.add_argument("--budget", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--target", action="append", default=None,
                        help="run these Makefile targets instead of the phase's usual ones "
                             "(repeatable), e.g. --target phase4-scan")
    parser.add_argument("--skip-gate", action="store_true",
                        help="do not run the phase gate - for diagnostic runs that have no "
                             "pass condition of their own")
    parser.add_argument("--make-var", default=None,
                        help='extra make variables, e.g. --make-var "SAMPLES4=20"')
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
