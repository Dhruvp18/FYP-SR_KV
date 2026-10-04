"""Periodically save a live pilot locally without logging its session URL."""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url-file", required=True)
    p.add_argument("--dest", default="results/conversation_pilot_v1")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--remote", default="srkv_conversation_pilot_v1/results/conversation_pilot_v1")
    p.add_argument("--record-glob", default="pilot.jsonl")
    p.add_argument("--expected-records", type=int, default=900)
    args = p.parse_args()
    base = Path(args.url_file).read_text().strip().rstrip("/")
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    remote = args.remote
    session = requests.Session()

    def get(path):
        response = session.get(base + "/api/contents/" + quote(path, safe="/"), timeout=20)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        return response.json()

    while True:
        try:
            listing = get(remote)
            for item in listing["content"]:
                if item["type"] != "file":
                    continue
                target = (dest / item["name"]).resolve()
                if not target.is_relative_to(dest):
                    raise ValueError("Unexpected destination")
                # Frozen datasets were already restored byte-for-byte.
                if target.name.endswith("_dataset.jsonl") and target.exists():
                    continue
                data = get(item["path"])
                raw = (base64.b64decode(data["content"]) if data["format"] == "base64"
                       else data["content"].encode("utf-8"))
                if target.suffix == ".jsonl":
                    lines = raw.splitlines(keepends=True)
                    if lines and not lines[-1].endswith(b"\n"):
                        lines.pop()
                    for line in lines:
                        json.loads(line)
                    raw = b"".join(lines)
                    if target.exists() and len(raw.splitlines()) < len(target.read_bytes().splitlines()):
                        raise ValueError("Remote checkpoint has fewer rows; local copy preserved")
                tmp = target.with_suffix(target.suffix + ".download")
                tmp.write_bytes(raw)
                tmp.replace(target)
            state = json.loads((dest / "pipeline_status.json").read_text())
            n = sum(len(f.read_bytes().splitlines()) for f in dest.glob(args.record_glob))
            print(f"{time.strftime('%H:%M:%S')} backed up {n}/{args.expected_records} answers; {state}", flush=True)
            (dest / "backup_status.json").write_text(json.dumps(
                {"last_success_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 "record_count": n,
                 **({"pilot_answers": n} if args.record_glob == "pilot.jsonl" else {}),
                 "pipeline": state}, indent=2))
            if state.get("state") in ("complete", "stopped"):
                return 0
        except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
            # Request exception text can contain the credential-bearing URL.
            print(f"{time.strftime('%H:%M:%S')} backup deferred: {type(exc).__name__}", flush=True)
        time.sleep(max(5, min(args.interval, 60)))


if __name__ == "__main__":
    raise SystemExit(main())
