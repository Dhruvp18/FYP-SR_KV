# Consensus-reinforced correction: a dataset built to favour SR-KV

Written before any GPU inference for this dataset exists. This is an
exploratory pilot, same tier as `conversation_pilot_v1` and
`incident_stream_v1` - it does not replace or modify Phase 8's pre-registered
hypotheses or either prior pilot's results.

## Why this dataset, specifically

Two things are already established from committed data:

1. **Recency helps when the correct answer is "the most recent version of a
   fact that changed."** `recency_hard_evict` beat `snapkv_unified` on
   `current_request`/`accepted_solution`-style questions in
   `conversation_pilot_v1` (consistently, at both budgets) and on the
   `persistent`/`recurrence` scenarios in `incident_stream_v1`'s sanity set.
2. **Clustering has never added anything on top of that.** In both pilots,
   `sr_kv` either exactly matched `recency_hard_evict` (incident_stream: byte-
   identical answers on all 12 questions) or was *worse* than it at one budget
   (conversation_pilot: -0.10 at budget 0.2). Neither prior task ever restated
   the corrected fact more than once, so clustering never had genuinely
   redundant material to average - the one structural condition `gist_mcq`'s
   H3b already showed clustering needs.

This dataset adds that missing ingredient on purpose: the corrected value is
restated `n_reinforcements` times (1, 3, or 5) late in the passage, competing
against one individually salient DECOY mention of the superseded value. It is
explicitly engineered to give SR-KV every advantage this project knows how to
construct - same disclosed-bias framing as `gist_mcq`'s H3 addendum. A win
here establishes "the combination can win somewhere, under conditions we
understand," not "SR-KV is generally better." A loss here, on a task built
specifically to favour it, is correspondingly strong evidence against the
mechanism.

## Hypothesis

**H.** `sr_kv`'s accuracy advantage over `recency_hard_evict` grows with
`n_reinforcements`. At `n_reinforcements=1` the two should be indistinguishable
(nothing to cluster); at `n_reinforcements=5`, `sr_kv` should beat
`recency_hard_evict` by a larger, significant margin than at `n_reinforcements=3`.
This is a **dose-response** prediction, not a single flat-offset claim - the
shape of the curve across the three variants is the actual test, and is
harder to produce by chance than one significant cell.

## Success criterion - fixed now

Primary comparison: `sr_kv` minus `recency_hard_evict`, paired on `sample_idx`
(the same conversation, same decoy, same product/location draw across all
three variants), paired bootstrap, 10,000 resamples, 95% CI.

**SUPPORTED** requires all of:
- at `consensus_n1`: CI includes zero (confirms the control collapsed as
  predicted - this is a precondition, not itself evidence for SR-KV);
- at `consensus_n3` and `consensus_n5`: CI excludes zero in SR-KV's favour;
- the point estimate at `n5` is larger than at `n3` (the dose-response shape);
- n >= 50 pairs per variant, at both budgets tested (0.1, 0.2).

Anything else is **NOT SUPPORTED**. A win at a single (variant, budget) cell
with no monotonic shape across `n_reinforcements` is noise, reported as such,
not as a win.

Secondary, reported alongside: `sr_kv`/`recency_hard_evict` both vs.
`snapkv_unified`, `streaming_llm`, and `full` (the ceiling and the two methods
with no recency term at all) - same comparisons `conversation_pilot_v1`
already reports, for continuity.

## Experiment

- **Model**: qwen2.5-1.5b-instruct, fp16, same as both prior pilots.
- **Context**: 4096 tokens (same as `conversation_pilot_v1`).
- **Budgets**: 0.1, 0.2.
- **Methods**: full, streaming_llm, snapkv_unified, recency_hard_evict,
  centroid_merge, sr_kv. `centroid_merge` (clustering, no recency) is included
  specifically to check the mechanism split cleanly: if clustering alone
  (no recency) also beats `snapkv_unified` here, the advantage isn't specific
  to the recency+clustering combination the hypothesis claims.
- **Variants**: `consensus_n1`, `consensus_n3`, `consensus_n5`
  (`eval/consensus_update.py`).
- **Sample size**: n=50 conversations, same seed across all methods/budgets/
  variants (every method answers the identical passages).
- **Scoring**: four-option MCQ, letter match (`eval.gist_mcq.score`), so a
  centroid's inability to reproduce exact tokens cannot by itself penalise it
  (same fix H1 already established).
- **Sanity gate before the main run** (same discipline as both prior pilots):
  5 conversations, all variants/methods/budgets. Continue only if `full`
  scores >= 9/10 overall on each variant, every record is error-free, cache
  accounting holds, and every compressed pre-question cache actually meets
  its token budget. If `full` cannot do the task even with every reinforcement
  present, stop and diagnose before touching the 50-conversation set.

## What would NOT count as support, stated in advance

- A win only at `consensus_n5`, flat or null at `n3` - that is a single
  lucky cell, not a dose-response, and is to be reported as inconclusive.
- A win driven entirely by `centroid_merge` also beating `snapkv_unified` by
  a similar margin - that would mean clustering alone explains the result,
  not the recency+clustering combination, and the headline claim would need
  rewriting to say so.
- Tuning `n_reinforcements`, the decoy's wording, or the depths after seeing
  pilot answers. The three variants are fixed now; no fourth variant gets
  added to chase significance.

## Known threats to validity

- This is still a synthetic, templated task with repeated structure across
  conversations (shared product/location pool, shared sentence templates) -
  the same external-validity limitation `conversation_pilot_v1` already
  disclosed. A positive result motivates a follow-up on real transcripts, it
  does not stand in for one.
- The decoy's wording ("URGENT CORRECTION") is one specific way of making a
  false mention individually salient. A null result here only rules out
  clustering's advantage under *this* decoy construction, not every possible
  one.
- n=50/variant/budget is pilot-scale, same as both predecessors. It can
  detect a real, fairly large effect; it is not powered to rule out a small
  one.
