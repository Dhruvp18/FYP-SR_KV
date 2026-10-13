"""Upload one file to an explicitly supplied Jupyter contents API session."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()
    base = Path(args.url_file).read_text().strip().rstrip("/")
    remote_parts = Path(args.remote).parts
    if not remote_parts or ".." in remote_parts:
        raise ValueError("Remote path must be a safe relative path")
    parent = ""
    for part in remote_parts[:-1]:
        parent = f"{parent}/{part}".lstrip("/")
        response = requests.put(base + "/api/contents/" + quote(parent, safe="/"),
                                json={"type": "directory"}, timeout=60)
        if response.status_code not in (200, 201):
            raise SystemExit(f"Create remote directory failed: HTTP {response.status_code}")
    payload = base64.b64encode(args.source.read_bytes()).decode("ascii")
    response = requests.put(
        base + "/api/contents/" + quote(args.remote, safe="/"),
        json={"type": "file", "format": "base64", "content": payload}, timeout=120)
    if response.status_code not in (200, 201):
        raise SystemExit(f"Upload failed: HTTP {response.status_code}")
    print(f"Uploaded {args.source.name} ({args.source.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
