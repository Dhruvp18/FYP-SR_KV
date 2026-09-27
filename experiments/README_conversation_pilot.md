# Running the conversation-state pilot

Read `conversation_pilot_protocol.md` for the fixed design and limitations.
This runner is separate from the historical phase grids; no prior results or
policy defaults are changed. The existing untracked `eval/conv_state.py` draft
is not used or modified.

On a T4 with repository dependencies installed:

```bash
python -m pytest -q
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_conversation_pilot.py --stage sanity
# Proceed only if sanity_gate.json contains "pass": true.
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_conversation_pilot.py --stage pilot
python scripts/analyse_conversation_pilot.py results/conversation_pilot_v1/pilot.jsonl
```

The runner writes the exact dataset before generation, fsyncs each result,
and supports resuming the identical command. Each question uses a fresh cache
and reprocesses the shared transcript independently. Model weights are public
Qwen2.5-1.5B-Instruct; no training or account credentials are required.

Outputs include `sanity.jsonl`, `sanity_gate.json`, `pilot.jsonl`, the two
dataset JSONL files, aggregate JSON files, and `pilot_analysis.json`.

The optional local `scripts/kaggle_session_exec.py` helper communicates with a
user-supplied Jupyter proxy using requests and websocket-client. It accepts a
URL-file and kernel-ID-file outside the repository; never commit a session
URL or credential. It creates a separate kernel, leaving the notebook kernel
alone. The measured work can run in a detached subprocess and save results
under `/kaggle/working/srkv_conversation_pilot_v1`.
