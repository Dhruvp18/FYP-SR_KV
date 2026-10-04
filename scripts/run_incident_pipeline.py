"""Validate, run two independent GPU shards, then analyse the fixed pilot."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/incident_stream_v1"


def status(stage, state="running", **extra):
    (OUT / "pipeline_status.json").write_text(json.dumps(dict(stage=stage, state=state, **extra)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"}
    status("tests")
    with (OUT / "tests.log").open("w") as log:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short"], cwd=ROOT,
                                env={**env, "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status("tests", "stopped", returncode=result.returncode)
        return result.returncode
    for stage in ("sanity", "main"):
        status(stage)
        workers, logs = [], []
        for shard in range(2):
            log = (OUT / f"{stage}_part{shard}.log").open("w")
            logs.append(log)
            workers.append(subprocess.Popen([
                sys.executable, "-u", "scripts/run_incident_experiment.py", "--stage", stage,
                "--shard", str(shard), "--num-shards", "2"], cwd=ROOT,
                env={**env, "CUDA_VISIBLE_DEVICES": str(shard)}, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in workers]
        for log in logs:
            log.close()
        if any(codes):
            status(stage, "stopped", returncodes=codes)
            return 1
        with (OUT / f"{stage}_analysis.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", "scripts/analyse_incident_experiment.py",
                                     "--stage", stage], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(stage + "_analysis", "stopped", returncode=result.returncode)
            return result.returncode
    status("analysis", "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
