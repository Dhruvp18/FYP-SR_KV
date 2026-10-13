"""Run the frozen conversation pilot at 30% and 40% cache budgets."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/conversation_pilot_30_40_v1"
BUDGETS = (0.3, 0.4)
BUDGET_ARGS = [str(b) for b in BUDGETS]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MPLBACKEND"] = "Agg"
    stages = (
        ("tests", ["-m", "pytest", "-q", "--tb=short"]),
        ("sanity", ["scripts/run_conversation_pilot.py", "--stage", "sanity",
                     "--output-dir", str(OUT), "--budgets", *BUDGET_ARGS]),
        ("pilot", ["scripts/run_conversation_pilot.py", "--stage", "pilot",
                    "--output-dir", str(OUT), "--budgets", *BUDGET_ARGS]),
        ("analysis", ["scripts/analyse_conversation_pilot.py", str(OUT / "pilot.jsonl"),
                       "--budgets", *BUDGET_ARGS]),
    )
    for stage, args in stages:
        (OUT / "pipeline_status.json").write_text(json.dumps(
            {"stage": stage, "state": "running", "budgets": BUDGETS}))
        stage_env = {**env, "CUDA_VISIBLE_DEVICES": "" if stage == "tests" else "0"}
        with (OUT / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, env=stage_env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (OUT / "pipeline_status.json").write_text(json.dumps(
                {"stage": stage, "state": "stopped", "returncode": result.returncode,
                 "budgets": BUDGETS}))
            return result.returncode
    (OUT / "pipeline_status.json").write_text(json.dumps(
        {"stage": "analysis", "state": "complete", "budgets": BUDGETS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
