# Pre-registration — Phase 8: does centroid-merging help anywhere?

Written **before** any Phase 8 data exists. Committed first, deliberately, so
the success criterion cannot move once numbers are on screen. Git history is
the proof of ordering.

## Why this phase exists

Phases 5 and 6 produced a clean null. `centroid_merge` against
`snapkv_unified` — one boolean apart, the comparison the whole design exists
to make:

    NIAH 1.5B @ budget 0.2     1.000  vs  1.000     (delta  0.000)
    NIAH 3B    (both budgets)  0.983  vs  0.983     (delta  0.000)
    LongBench 1.5B @ 0.3       0.248  vs  0.249     (delta -0.001)

Two structural reasons that null may be an artefact of *what was measured*
rather than a property of the mechanism:

1. **Every metric so far rewards exact tokens.** NIAH is exact match;
   LongBench uses `qa_f1` and `rouge_l`, both n-gram overlap. A centroid is an
   attention-weighted average and cannot reproduce an exact token, so these
   metrics penalise it by construction. Consistent with this, `gov_report`
   (ROUGE) is the one task where merging is *significantly worse*
   (-0.0094, 95% CI [-0.0179, -0.0022], paired bootstrap n=25).
2. **Compression has been nearly free.** At budget 0.3 hard eviction costs
   only 2.3% against uncompressed. There is almost nothing for merging to
   recover.

## Hypotheses

- **H1 (metric).** Under a metric that rewards *distributional* fidelity
  rather than exact tokens, centroid-merging beats hard eviction. Perplexity
  is that metric, and is the standard KV-compression metric (StreamingLLM
  reports it).
- **H2 (pressure).** Under aggressive compression, where hard eviction
  discards information rather than redundancy, centroid-merging beats hard
  eviction. Partial signal should beat none.

## Success criterion — fixed now

For each experiment, the comparison is **`centroid_merge` vs
`snapkv_unified`**, paired on `sample_idx`, tested by paired bootstrap
(10,000 resamples, 95% CI).

A hypothesis is **SUPPORTED** only if:
- the 95% CI on the paired mean difference excludes zero **in the predicted
  direction**, and
- n >= 50 pairs per comparison, and
- it holds at **more than one** setting (>=2 budgets for H2, >=2 context
  lengths for H1).

Anything else is **NOT SUPPORTED**. A single significant cell among many is
noise, and is to be reported as such.

`sr_kv` vs `snapkv_unified` is reported alongside but is **secondary**: the
recency term is already established as harmful on depth-randomised retrieval,
and H1/H2 are about clustering.

## What gets reported either way

Every result, including negatives, and including any experiment started and
abandoned. The existing Phase 5/6 nulls stay in the writeup regardless of what
Phase 8 shows. If H1 and H2 both fail, the conclusion is that centroid-merging
does not help, stated plainly.

Explicitly ruled out:
- reporting a subgroup that was not named above,
- re-running a comparison with more samples because the first was not
  significant,
- switching the metric or the pairing after seeing results.

## Experiments

**E1 (H1) — perplexity.** qwen2.5-1.5b, contexts 2048/4096/8192, budgets
0.2/0.3, methods full/streaming_llm/snapkv_unified/centroid_merge/sr_kv,
n>=50 documents. Metric: token-level perplexity on a held-out continuation,
conditioned on the compressed cache of the prefix. Lower is better, so the
predicted direction is `centroid_merge` perplexity **below**
`snapkv_unified`.

**E2 (H2) — tight budgets.** LongBench, budgets 0.05/0.1/0.15, same four
tasks and methods as Phase 5, n=25/task (100 pairs/method).

## Known threat to validity

Perplexity and downstream task accuracy are not the same thing, and a win on
E1 alone does not establish a useful method. If E1 succeeds and E2 fails, the
honest claim is "centroid-merging preserves distributional information that
n-gram metrics do not credit", not "SR-KV is better".
