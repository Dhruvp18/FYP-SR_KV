"""Host the live Cache Lens backend (server/) on a Kaggle GPU, tunnelled out
with ngrok, so the Next.js frontend on your local machine can talk to a real
GPU instead of your CPU.

This is a different Kaggle workflow from `scripts/kaggle_kernel.py`: that one
pushes *detached* batch kernels ("Save & Run All") that run unattended and
produce `results/*.jsonl`. A live demo needs the opposite - an *interactive*
session that stays up while you drive it from a browser - so run this from a
notebook cell directly (`!python scripts/kaggle_demo_server.py`), not via a
committed/detached push, and keep that notebook tab open for as long as you
want the demo reachable.

One-time setup, in the Kaggle notebook:

1. Notebook settings -> Accelerator: GPU T4 x2 (or P100). Internet: On.
   (Same settings KAGGLE.md's eval workflow uses.)
2. Get a free authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
   and store it as a Kaggle Secret named NGROK_AUTHTOKEN (Add-ons -> Secrets).
3. First cell - get the code onto Kaggle (same as KAGGLE.md section 0.1):
     !git clone -q https://github.com/<you>/FYP-SR_KV.git /kaggle/working/sr-kv
     %cd /kaggle/working/sr-kv
4. Second cell - install deps (fastapi/uvicorn/websockets are already in
   requirements.txt; pyngrok is Kaggle-only, not in requirements.txt on
   purpose - the local dev workflow never needs a tunnel):
     !pip install -q -U "transformers>=5.0" accelerate fastapi "uvicorn[standard]" websockets pyngrok
5. Third cell:
     !python scripts/kaggle_demo_server.py

That cell blocks (it's serving requests) and prints the public wss:// URL to
paste into your local webapp/.env.local as NEXT_PUBLIC_WS_URL, e.g.:

     NEXT_PUBLIC_WS_URL=wss://<something>.ngrok-free.app/ws/generate

Then restart `npm run dev` locally (Next.js only reads NEXT_PUBLIC_* env vars
at dev-server start) and use the app exactly as with the local backend -
nothing in webapp/ needs to change, `useGenerationSocket.ts` already reads
this var. CORS needs no change either: the browser's request Origin is still
http://localhost:3000 regardless of where the backend physically runs.

The ngrok URL is random and changes every time this cell (re)starts, and the
GPU session itself is capped (~12h, killed without warning) - re-run this
script and paste the new URL when that happens, same "just re-run it" model
KAGGLE.md's eval workflow already uses for interrupted runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ngrok_authtoken() -> str | None:
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient  # only importable inside a Kaggle kernel

        return UserSecretsClient().get_secret("NGROK_AUTHTOKEN")
    except Exception:
        return None


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA device visible - this will still run (server/model_singleton.py "
            "allows CPU), just at local-machine speed, defeating the point of running it here. "
            "Check the notebook's Accelerator setting.",
            file=sys.stderr,
        )

    from pyngrok import conf, ngrok

    token = _ngrok_authtoken()
    if not token:
        raise SystemExit(
            "No ngrok authtoken found. Set it as a Kaggle Secret named NGROK_AUTHTOKEN "
            "(Add-ons -> Secrets), or export NGROK_AUTHTOKEN before running this script. "
            "Get a free one at https://dashboard.ngrok.com/get-started/your-authtoken"
        )
    conf.get_default().auth_token = token

    port = 8000
    tunnel = ngrok.connect(port, "http")
    ws_url = tunnel.public_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws/generate"

    print("=" * 70)
    print(f"Public URL:  {tunnel.public_url}")
    print(f"WebSocket:   {ws_url}")
    print()
    print("Paste this into webapp/.env.local on your local machine, then")
    print("restart `npm run dev`:")
    print()
    print(f"  NEXT_PUBLIC_WS_URL={ws_url}")
    print("=" * 70)

    import uvicorn

    # Blocks here for as long as the notebook session is kept alive - this IS
    # the server, not a launcher for a background one (see module docstring
    # for why this has to be an interactive cell, not a detached kernel push).
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
