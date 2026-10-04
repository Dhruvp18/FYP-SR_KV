# Streaming incident monitoring: sanity result

Run date: 2026-10-03. Status: **sanity gate failed; main experiment not run**.

This is a new application, using chronological service logs and a fixed cache
capacity of 512 entries per layer/head. It does not repeat the earlier customer
support experiment at 10% and 20% budgets. The model receives logs in 256-token
chunks and answers at three checkpoints, with each question evaluated on a copy
of the continuing cache. The logs are synthetic, not production incident data.

## Observations

Four streams, three checkpoints each, six methods: 72 saved answers.
All methods saw the same streams and questions. The baselines are this
repository's implementations of the named approaches.

| Method | Correct / 12 | Accuracy |
|---|---:|---:|
| Full cache | 6 | 50.0% |
| SR-KV | 5 | 41.7% |
| Recency-only hard eviction | 5 | 41.7% |
| Centroid merge without recency | 4 | 33.3% |
| SnapKV-style unified hard eviction | 3 | 25.0% |
| StreamingLLM-style | 3 | 25.0% |

These are sanity results, not a successful application benchmark. SR-KV tied
recency-only on every evaluated answer. Its descriptive advantage over SnapKV
and StreamingLLM is based on only four independent streams, and the paired
confidence intervals include zero. There is no evidence of superiority over
all compressed baselines.

The preregistered sanity requirement was at least 10/12 correct with full
cache. The full model obtained only 6/12, so the pipeline stopped automatically.
The planned 24-stream, 432-answer main run was not started. No thresholds,
cache policies, or datasets were adjusted to produce a win.

## Failure diagnosis

The full-cache model missed six active incidents by answering NONE. It produced
no invalid labels. A separate post-sanity diagnostic replayed exactly the same
token sequences through native Hugging Face SDPA, with ordinary full-prompt
generation and no custom cache. Its predictions matched the streaming full-cache
runner on all 12 checkpoints, also scoring 6/12. This supports a model/task
limitation and does not suggest a prediction-level streaming defect in these
cases; it is not proof of numerical equivalence for every input.

The model was Qwen2.5-1.5B-Instruct, FP16, pinned checkpoint
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with Transformers 5.17.0 on two
Tesla T4 GPUs. The complete CPU test suite passed on Kaggle (198 tests), including
nine new streaming tests. Analysis validated the complete 72-answer grid, cache
accounting, shared stream content, and the fixed compressed-cache size.

Compressed caches retained 14 MiB of K/V tensors at each checkpoint; full cache
reached 166.9 MiB. This does not imply a comparable reduction in total GPU memory.
SR-KV was slower here: average segment ingestion was 1.66 seconds versus 0.73
for recency-only, and average probe time including cache copying was 0.95 versus
0.47 seconds. These small-run timings are secondary observations.

## Saved evidence and reproduction

- `sanity_dataset.jsonl`: all four streams, gold answers and lifecycle events.
- `sanity_part0.jsonl`, `sanity_part1.jsonl`: all 72 generated answers and measurements.
- `sanity_part0.json`, `sanity_part1.json`: finalized records and environment metadata.
- `sanity_analysis.json`, `sanity_gate.json`: scores, intervals and failed gate.
- `full_native_diagnostic.json`: all 12 native full-cache diagnostic predictions.
- `tests.log`, setup and worker logs: execution evidence.
- `../../experiments/incident_stream_protocol.md`: frozen design.

Recompute the sanity analysis from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/analyse_incident_experiment.py --stage sanity
```

Exit code 2 is expected because the sanity gate failed. The inference runner is
`scripts/run_incident_pipeline.py`; it requires a compatible CUDA environment
with two GPUs and model access. The native diagnostic is
`scripts/diagnose_incident_full.py`.

A reasonable follow-up would first test a stronger model's full-cache task
accuracy. That would be a separately documented experiment, with fresh held-out
evaluation streams and the same fair comparisons. This run does not establish
that a stronger model would give SR-KV an advantage.
