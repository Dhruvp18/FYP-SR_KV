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

---

## Addendum (2026-09-20) — H3: a task built for the mechanism, not borrowed

Written before any `phase8_gist_*` file exists. E1/E2 ask whether the null is
an artefact of *what was measured* on tasks NIAH and LongBench already
happened to include. H3 instead asks: is there *any* application where
clustering's actual mechanism - averaging cosine-similar candidate keys into
one attention-weighted centroid, per CLAUDE.md A5 / `src/clustering.py` - is
a plausible win, and can a task be built to isolate exactly that?

Two structural properties make clustering plausible over hard eviction, and
neither NIAH nor LongBench has both:

1. **The evicted content must be genuinely redundant** - similar key vectors
   for cosine-clustering to average into a faithful summary, rather than
   forcing dissimilar content together into a vector that resembles nothing.
   NIAH's needle is unique by construction; LongBench's `gov_report` is dense,
   non-redundant prose, and is already the one task where clustering measured
   *significantly worse* (-0.0094, 95% CI [-0.018, -0.002], n=25 - see above).
   This is cited as the reasoning that motivated H3, not as evidence for it:
   re-testing `gov_report` with more samples is exactly the move the base
   pre-registration rules out, so H3 is tested on a new task instead.
2. **The metric must not need exact tokens back.** Same reasoning as H1, but
   applied to a downstream judgment instead of perplexity: if the answer is
   "which of these four options", a blurry-but-present attention target can
   move the model toward the right choice without reproducing any wording.

`eval/gist_mcq.py` builds a task with both properties: a fact is restated
several times, in different phrasing, scattered non-locally through a long
document (so no single restatement is "recently attended" when compression
scores it), and the question is multiple-choice, scored on which letter comes
out - not on string overlap with any span in the text.

Two independent variants, each its own hypothesis:

- **H3a (`attribution`).** An entity is linked to an attribute (a "team
  colour") via four restatements at scattered depths, plus two single-mention
  decoy entities. Question: which team was the entity assigned to (MCQ).
  Tests whether the entity/attribute link survives when every individual
  mention is, alone, unremarkable to windowed-attention scoring.
- **H3b (`aggregation`).** A majority topic is mentioned five times and two
  minority topics once each, scattered non-locally. Question: which issue
  came up most often (MCQ). Tests whether *relative frequency* survives, not
  just existence - a centroid pulled from five similar mentions should point
  more strongly toward that topic than one pulled from a single mention,
  something hard eviction has no equivalent of (a token either survives whole
  or is gone; there is no "how many almost-evicted tokens agreed").

### Success criterion - identical structure to H1/H2, fixed now

For each variant independently: `centroid_merge` vs `snapkv_unified`, paired
on `sample_idx`, paired bootstrap (10,000 resamples, 95% CI). SUPPORTED
requires the CI to exclude zero in the predicted direction (higher accuracy
for `centroid_merge`), n>=50 pairs, at >=2 budget settings. H3a and H3b are
scored and reported independently - a win on one is not evidence for the
other, since they test different claims about what a centroid preserves.
`sr_kv` vs `snapkv_unified` is secondary, exactly as in H1/H2.

### Experiment

**E3 (H3a + H3b).** qwen2.5-1.5b, context 8192 (compression must actually
bite for either mechanism to show anything - see the "fair last shot" note
above on why 0.3 was nearly free), budgets 0.1/0.15, methods
full/snapkv_unified/centroid_merge/sr_kv, n=50/variant/budget.
`streaming_llm` is dropped for this experiment only: it is architecturally
different, not part of the clustering comparison, and this is a second
speculative experiment on top of an already-committed E1/E2, not a full
phase - `make phase8-gist`.

### Known threat to validity

This task is synthetic and designed specifically to give clustering the
conditions it should need. A win here establishes "the mechanism can win
somewhere", which is weaker than "the mechanism helps on realistic text" -
the same gap E1 already has between perplexity and downstream usefulness. A
loss here, on a task built to give the mechanism every advantage this project
could construct, would be considerably stronger evidence against it than
another null on tasks that were never designed to favour either policy.
