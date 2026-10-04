# Incident monitoring: fixed-capacity streaming pilot v1

Frozen before new GPU inference, 2026-10-03. This is a new exploratory task;
prior conversation/Phase 8 results and their criteria remain unchanged.

Application: identify the presently unresolved failing service as a log stream
arrives. Synthetic labelled service logs give unambiguous ground truth, not a
claim about production logs or causal root-cause diagnosis. Four scenarios:
handoff to a different incident, complete recovery, recurrence, and persistence.
Repeated stale errors, sparse new errors and unrelated routine activity appear.
Only explicit recovery closes a failure; a successful request does not.

Model: the same fixed Qwen2.5-1.5B-Instruct checkpoint used previously, fp16.
Ingest logs in chronological 256-token chunks, retaining cache state across
three successive segments. After each segment, query a COPY of the live cache;
discard query/answer state so probes never teach the continuing stream.

One fixed cap: 512 KV entries per layer/head for each compressed method,
independent of stream length. No percentage sweep. Temporary chunk tensors and
method-specific metadata are additional memory and are reported separately
from retained KV slots. Full cache has no cap and is an accuracy reference.

Methods: full, StreamingLLM-style sinks+recent window, SnapKV-style unified
hard eviction, recency-only hard eviction, centroid merge without recency,
and full SR-KV. Same frozen defaults, chunks, probes and logs. The baselines are
this repository's implementations, not a claim about all published variants.

Sanity: 4 streams (one per scenario), seed 31003; 3 checkpoints x 6 methods =
72 answers. Pass requires full-cache >=10/12 correct, all records valid,
accounting correct, and every compressed retained cache within 512 entries.
If the full model cannot do the task, stop and diagnose, without inspecting or
tuning the main set. Cache or harness defects may be repaired transparently.

Main: 24 fresh streams, seed 41003, balanced across scenarios; 432 answers.
Two identical T4s may process disjoint streams; no shared model/cache state.
Stop on errors; fsync each checkpoint; resume exact dataset/config only.

Primary metric: active-service accuracy, averaging the three checkpoints
within each stream. Unit of resampling is a STREAM, not a checkpoint. Compare
SR-KV with each compressed baseline using 10,000 paired bootstrap draws.
Report 95% CIs; use 98.75% CIs for four simultaneous primary comparisons
(Bonferroni family-wise 5%). Evidence of an exploratory advantage requires
positive adjusted CI versus ALL four compressed baselines, plus >=80% absolute
accuracy. Otherwise report the narrower differences and no overall advantage.
This small pilot can miss real small improvements; do not add examples to
chase significance after seeing it. A positive result needs independent real
logs and stronger application-specific baselines, including a rule-based state
tracker for logs with explicit lifecycle events.

Also report macro F1 over database/payments/auth/queue/NONE, wrong-service
rate, missed alerts (NONE predicted for active incidents), false alerts in
healthy periods, per-scenario accuracy, and accuracy at each checkpoint.
Timing: incremental ingest time, probe time (including cache-copy overhead),
and compression time. Peak GPU allocation includes model, transient buffers
and probe copy; it is not isolated cache memory. Retained K/V tensor bytes are
derived from the actual model dimensions/dtype and retained slots. Do not claim
less overall memory or faster responses merely from the 512-entry limit.

All datasets, generated answers, failures, settings, and analyses are retained.
No threshold, sample set, scenario weight or cache-policy tuning after main
results are visible. The experiment tests a hypothesis, not a promised win.
