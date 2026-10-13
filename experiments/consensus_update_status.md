# Experiment status

## Complete — verified 2026-10-13

Ran on Kaggle (chaitrasamant/sr-kv-consensus-update), qwen2.5-1.5b fp16, one T4.
165 sanity answers + 1,650 pilot answers, zero errors, cache accounting held
throughout. First attempt's decoy wording made the task ambiguous even for
the uncompressed `full` reference (60% accuracy, sanity gate correctly
stopped the run); fixed and re-verified (`full` = 100% on all three variants)
before the real pilot ran. See `results/consensus_update_v1/sanity_v1_rejected_decoy/`
for the discarded attempt.

**Verdict: NOT SUPPORTED, both budgets.** No dose-response in `n_reinforcements`
for `sr_kv` over `recency_hard_evict` — the pre-registered, falsifiable
prediction this pilot existed to test.

    sr_kv minus recency_hard_evict         n1      n3      n5
    budget 0.1                           +0.00   +0.04   -0.04
    budget 0.2                           +0.16   -0.10    0.00

Neither row is monotonically increasing (the dose-response shape required by
`experiments/consensus_update_protocol.md`). Where a difference did reach
significance, it was never in the predicted shape: at budget 0.2, `n3` is
negative and significant (`sr_kv` minus `snapkv_unified` = -0.30, 95% CI
[-0.42, -0.18]) — the opposite of the hypothesis — and the one `n1` gap that
*is* significant (+0.16) sits in a cell the design predicts should show
nothing (nothing to cluster at `n_reinforcements=1`), so it cannot be
attributed to clustering and is reported as unexplained noise, not support.

`centroid_merge` (clustering, no recency) beat `snapkv_unified` at exactly
one of six cells (`consensus_n1/b0.1`), again at `n1` where there is nothing
to cluster — consistent with noise, not a real clustering effect.

Every method, including hard-eviction `streaming_llm`, improved sharply from
`n1` to `n5` (e.g. `streaming_llm` at budget 0.2: 4% to 100%). Redundancy
helps robustness under compression generally; it does not differentially
favour the recency+clustering combination over hard eviction, which was the
actual claim under test.

This pilot was explicitly engineered to give SR-KV's clustering mechanism
every advantage this project could construct (see protocol doc's own framing,
same discipline as `gist_mcq`'s H3). It still did not show the predicted
effect. This is a materially stronger negative result for centroid merging
than `conversation_pilot_v1` or `incident_stream_v1` individually, because
those never isolated the specific condition (redundant reinforcement) this
one was built around.

Mean compression time per answer, for the record (not part of the verdict):
`sr_kv`/`centroid_merge` ~0.95-1.0s vs `recency_hard_evict`/`snapkv_unified`
~0.23-0.27s vs `streaming_llm` ~0.1s — clustering costs roughly 4x a hard-
eviction method's compression time here, consistent with the earlier timing
discussion.
