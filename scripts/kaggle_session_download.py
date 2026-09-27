"""Download a named experiment directory through Jupyter's contents API."""
import argparse
import base64
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    base = Path(args.url_file).read_text().strip().rstrip("/")
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    def download(remote, local):
        response = session.get(base + "/api/contents/" + quote(remote, safe="/"), timeout=60)
        if response.status_code != 200:
            raise SystemExit(f"Contents API returned HTTP {response.status_code}")
        data = response.json()
        if data["type"] == "directory":
            local.mkdir(parents=True, exist_ok=True)
            for item in data["content"]:
                target = (local / item["name"]).resolve()
                if not target.is_relative_to(dest):
                    raise ValueError("Unexpected path outside destination")
                download(item["path"], target)
        else:
            if data["format"] == "base64":
                payload = base64.b64decode(data["content"])
            else:
                payload = data["content"].encode("utf-8")
            local.write_bytes(payload)
            print(f"Saved {local.name}: {len(payload)} bytes", flush=True)
    download(args.remote, dest)


if __name__ == "__main__":
    main()
