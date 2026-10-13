# Conversation-state cache-budget repeat: 30% and 40%

This is a separate repeat of the frozen conversation-state pilot described in
`conversation_pilot_protocol.md`. It preserves the five methods, Qwen2.5-1.5B
Instruct, fp16, the two-question fresh-cache procedure, seeds, 5-example sanity
set, 50-conversation pilot, scoring, and analysis. The only planned change is
the compressed-cache budgets: 0.3 and 0.4 instead of 0.1 and 0.2. Full cache
remains at 1.0. The original results are not overwritten.

Outputs go to `results/conversation_pilot_30_40_v1/`. The sanity gate must pass
before the pilot starts. The pilot uses the same deterministic datasets as the
earlier run when the tokenizer and dependencies match; dataset hashes are saved
in the result metadata for verification. This is an exploratory mechanism test,
not a confirmatory or general-superiority claim.

The completed Kaggle run used Python 3.13, PyTorch 2.11.0+cu128, Transformers
5.16.1, and a Tesla T4. The 10%/20% pilot recorded PyTorch 2.10.0+cu128 and
Transformers 5.17.0. The deterministic pilot dataset hash and `src/` source
hash match the earlier run, but the runtime versions differ; see
`results/conversation_pilot_30_40_v1/run_environment.json` when comparing
results.

Run on a CUDA GPU from the repository root:

```bash
python scripts/run_conversation_30_40_pipeline.py
```

The pipeline runs the repository tests, sanity gate, 900-answer pilot, and
conversation-level paired bootstrap analysis. It resumes only matching result
keys in its separate output directory.
