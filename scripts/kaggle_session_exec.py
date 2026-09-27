"""Execute a Python file on an explicitly supplied Jupyter session.

The credential-bearing URL is read from a local file, never logged or stored
in experiment artifacts. Requires requests and websocket-client locally.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import uuid
from pathlib import Path

import requests
import websocket


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url-file", required=True)
    p.add_argument("--kernel-file", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()
    base = Path(args.url_file).read_text().strip().rstrip("/")
    kernel_file = Path(args.kernel_file)
    session = requests.Session()
    if kernel_file.exists():
        kernel_id = kernel_file.read_text().strip()
    else:
        response = session.post(base + "/api/kernels", json={"name": "python3"}, timeout=30)
        if response.status_code not in (200, 201):
            raise SystemExit(f"Create kernel: HTTP {response.status_code}")
        kernel_id = response.json()["id"]
        kernel_file.write_text(kernel_id)
    ident = uuid.uuid4().hex
    url = base.replace("https://", "wss://", 1) + f"/api/kernels/{kernel_id}/channels?session_id={ident}"
    conn = websocket.create_connection(url, timeout=args.timeout, suppress_origin=True)
    message_id = uuid.uuid4().hex
    message = {
        "header": {"msg_id": message_id, "username": "srkv", "session": ident,
                   "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": Path(args.code).read_text(encoding="utf-8"),
                    "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
    }
    conn.send(json.dumps(message))
    failed = False
    try:
        while True:
            msg = json.loads(conn.recv())
            if msg.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind, content = msg.get("msg_type", msg.get("header", {}).get("msg_type")), msg["content"]
            if kind == "stream":
                print(content.get("text", ""), end="", flush=True)
            elif kind in ("execute_result", "display_data"):
                print(content.get("data", {}).get("text/plain", ""), flush=True)
            elif kind == "error":
                failed = True
                print(content.get("ename"), content.get("evalue"), flush=True)
            elif kind == "status" and content.get("execution_state") == "idle":
                break
    finally:
        conn.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
