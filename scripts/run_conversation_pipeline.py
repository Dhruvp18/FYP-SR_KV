"""Run validation, sanity and pilot sequentially, stopping on any failed gate."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/conversation_pilot_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MPLBACKEND"] = "Agg"
    stages = (
        ("tests", ["-m", "pytest", "-q", "--tb=short"]),
        ("sanity", ["scripts/run_conversation_pilot.py", "--stage", "sanity"]),
        ("pilot", ["scripts/run_conversation_pilot.py", "--stage", "pilot"]),
        ("analysis", ["scripts/analyse_conversation_pilot.py", str(OUT / "pilot.jsonl")]),
    )
    for stage, args in stages:
        (OUT / "pipeline_status.json").write_text(json.dumps({"stage": stage, "state": "running"}))
        stage_env = {**env, "CUDA_VISIBLE_DEVICES": "" if stage == "tests" else "0"}
        with (OUT / f"{stage}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, env=stage_env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (OUT / "pipeline_status.json").write_text(json.dumps(
                {"stage": stage, "state": "stopped", "returncode": result.returncode}))
            return result.returncode
    (OUT / "pipeline_status.json").write_text(json.dumps({"stage": "analysis", "state": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
